"""Policy-to-script USB insertion: local wrist detection, visual alignment, taught push."""

import argparse
import json
import pathlib
import sys
import time
import urllib.request

import cv2
import numpy as np

from dream_hackathon.bundle_pull import HttpArm, ToolKinematics, move_to, play, sample
from dream_hackathon.bundle_vision import servo
from dream_hackathon.orchestration import lookup


class InsertionVision:
    """Locate the taught port plane and held orange USB in one wrist image.

    Port features are expressed relative to the live plug tip and plug axis,
    so a different grasp position does not request an impossible movement of
    the wrist-mounted camera relative to its own gripper.
    """

    def __init__(self, reference, settings):
        self.reference, self.settings = reference, settings
        self.gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.anchors = np.asarray(settings['port_polygon'], np.float32)
        mask = np.zeros(self.gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, self.anchors.astype(np.int32), 255)
        x, y, w, h = settings['plug_roi']
        mask[y:y+h, x:x+w] = 0
        self.points = cv2.goodFeaturesToTrack(self.gray, 150, 0.01, 3, mask=mask)
        if self.points is None or len(self.points) < 8:
            raise ValueError('Reference port plane needs eight trackable features')
        tip, basis = self.plug(reference)
        self.target = ((self.anchors - tip) @ basis).ravel()

    def plug(self, frame):
        x, y, w, h = self.settings['plug_roi']
        crop = frame[y:y+h, x:x+w]
        mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
                           np.array(self.settings['hsv_low']), np.array(self.settings['hsv_high']))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            raise ValueError('Held orange USB is not visible')
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[label, cv2.CC_STAT_AREA] < self.settings['minimum_plug_pixels']:
            raise ValueError('Too few held USB pixels')
        yy, xx = np.nonzero(labels == label)
        pixels = np.column_stack((xx + x, yy + y)).astype(float)
        _, _, axes = np.linalg.svd(pixels - pixels.mean(axis=0), full_matrices=False)
        direction = axes[0]
        if direction @ np.asarray(self.settings['plug_axis_hint']) < 0:
            direction = -direction
        positions = pixels @ direction
        tip = pixels[positions <= np.quantile(positions, 0.15)].mean(axis=0)
        basis = np.column_stack(([direction[1], -direction[0]], direction))
        return tip, basis

    def measure(self, image):
        frame = cv2.resize(image, self.reference.shape[1::-1])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        options = dict(winSize=(25, 25), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
        live, found, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **options)
        if live is None or found is None:
            raise ValueError('Port plane not tracked')
        back, returned, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, live, None, **options)
        if back is None or returned is None:
            raise ValueError('Port plane reverse tracking failed')
        valid = found.ravel().astype(bool) & returned.ravel().astype(bool)
        valid &= np.linalg.norm(back - self.points, axis=2).ravel() < 1
        if valid.sum() < 8:
            raise ValueError('Too few consistent port-plane matches')
        homography, inliers = cv2.findHomography(self.points[valid], live[valid], cv2.RANSAC, 1.5)
        if homography is None or inliers is None or inliers.sum() < 8:
            raise ValueError('Port plane not located')
        anchors = cv2.perspectiveTransform(self.anchors.reshape(-1, 1, 2), homography).reshape(-1, 2)
        tip, basis = self.plug(frame)
        features = ((anchors - tip) @ basis).ravel()
        errors = np.linalg.norm((features - self.target).reshape(-1, 2), axis=1)
        port = cv2.perspectiveTransform(np.float32(self.settings['port_point']).reshape(1, 1, 2), homography)[0, 0]
        gap = float(np.linalg.norm(port - tip))
        status = {'known': True, 'near_opening': bool(errors.max() <= self.settings['near_error_px']
                                                    and gap <= self.settings['near_gap_px']),
                  'aligned': bool(errors.max() <= self.settings['align_tolerance_px']),
                  'error_px': float(errors.max()), 'tip_port_gap_px': gap,
                  'matched_features': int(inliers.sum()), 'features': features.tolist()}
        cv2.polylines(frame, [np.rint(anchors).astype(np.int32)], True, (0, 255, 0), 1)
        cv2.circle(frame, tuple(np.rint(tip).astype(int)), 4, (0, 128, 255), 1)
        cv2.circle(frame, tuple(np.rint(port).astype(int)), 4, (255, 0, 255), 1)
        return features, status, frame


def mean_rotation(rotations):
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def push_rows(demos, aligned_q, grip, kinematics, samples, duration):
    """Average time-normalized relative TCP paths, never absolute joint angles."""
    start = kinematics.fk(np.asarray(aligned_q))
    origins = [np.linalg.inv(kinematics.fk(np.asarray(demo['rows'][0]['q']))) for demo in demos]
    result, seed = [], np.asarray(aligned_q).copy()
    for phase in np.linspace(0, 1, samples):
        relatives = [origin @ kinematics.fk(sample(demo['rows'], phase * demo['rows'][-1]['t'])[0])
                     for demo, origin in zip(demos, origins, strict=True)]
        relative = np.eye(4)
        relative[:3, 3] = np.mean([pose[:3, 3] for pose in relatives], axis=0)
        relative[:3, :3] = mean_rotation([pose[:3, :3] for pose in relatives])
        seed = kinematics.solve(start @ relative, seed)
        result.append({'t': float(phase * duration), 'q': seed.tolist(), 'grip': float(grip)})
    return result


def execute(arm, vision, plan, kinematics, report, policy_stop=None, sleep=time.sleep):
    """Wait alongside policy, take over when near, align, then execute taught push."""
    while True:
        try:
            _, status, image = vision.measure(arm.image())
        except ValueError as error:
            report('waiting', {'known': False, 'reason': str(error)}, None)
            sleep(plan['poll_seconds'])
            continue
        report('waiting', status, image)
        if status['near_opening']:
            break
        sleep(plan['poll_seconds'])
    if policy_stop is not None:
        policy_stop()
    q, grip = arm.state()
    arm.command(q, grip)
    report('takeover', {'q': q.tolist(), 'grip': grip}, None)

    def move(target, held_grip):
        move_to(arm, target, held_grip, plan['servo']['motion_rad_s'],
                hz=plan['hz'], settle=plan['servo']['settle_s'])

    aligned = servo(arm, vision, move, plan['servo'], report)
    # Recheck the live measurement after the final correction; never jump to
    # the recording's absolute start joints or reopen the held USB.
    _, status, image = vision.measure(arm.image())
    report('aligned', {**status, 'q': aligned.tolist()}, image)
    if not status['aligned']:
        raise ValueError('Live alignment changed before the insertion push')
    rows = push_rows(plan['demonstrations'], aligned, grip, kinematics,
                     plan['push_samples'], plan['push_seconds'])
    report('push', {'rows': len(rows), 'seconds': plan['push_seconds'], 'targets': rows}, None)
    play(arm, rows, hz=plan['hz'])
    if plan.get('release_grip') is not None:
        arm.command(np.asarray(rows[-1]['q']), plan['release_grip'])
    result = {'script_complete': True, 'push_complete': True, 'insertion_verified': False,
              'release_commanded': plan.get('release_grip') is not None}
    report('complete', result, None)
    return result


def prepare(dataset, selections, output):
    """Export explicitly selected demonstration intervals and the first reference frame."""
    import pyarrow.parquet as pq

    info = json.loads((dataset / 'meta/info.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    demos = []
    for number, selection in enumerate(selections):
        episode, first, last = map(int, selection.split(':'))
        if first < 0 or last <= first:
            raise ValueError('Choose episode:start_frame:end_frame with end after start')
        fields = {'episode_chunk': episode // info['chunks_size'], 'episode_index': episode,
                  'video_key': 'observation.images.right_wrist'}
        table = pq.read_table(dataset / info['data_path'].format(**fields))
        states = np.asarray(table['observation.state'].to_pylist())
        times = np.asarray(table['timestamp'].to_pylist())
        names = info['features']['observation.state'].get('names') or []
        if (states.ndim != 2 or states.shape[1] != 7 or len(names) != 7
                or not all(name.startswith('right_') for name in names[:6])):
            raise ValueError('Select a single-right-arm recording with six joints and one gripper')
        if last >= len(times):
            raise ValueError('Selected insertion interval exceeds the episode')
        rows = [{'t': float(times[i] - times[first]), 'q': states[i, :6].tolist(), 'grip': 0.0}
                for i in range(first, last + 1)]
        demos.append({'episode': episode, 'first_frame': first, 'last_frame': last,
                      'source': str(dataset), 'column': 'observation.state', 'rows': rows})
        if number == 0:
            capture = cv2.VideoCapture(str(dataset / info['video_path'].format(**fields)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            ok, image = capture.read()
            capture.release()
            if not ok:
                raise ValueError('Could not decode the selected pre-insertion reference')
            if not cv2.imwrite(str(output / 'reference.png'), image):
                raise ValueError('Could not save reference image')
    plan = {'version': 1, 'name': 'USB-C reference alignment and taught insertion',
            'status': 'prepared offline; not executed on hardware', 'reference': 'reference.png',
            'demonstrations': demos, 'hz': 30, 'poll_seconds': 0.1, 'push_samples': 101,
            'push_seconds': float(np.mean([demo['rows'][-1]['t'] for demo in demos])),
            'release_grip': None,
            'vision': {'port_polygon': [[230, 193], [341, 179], [352, 243], [305, 257]],
                       'port_point': [316, 203], 'plug_roi': [285, 185, 65, 58],
                       'hsv_low': [0, 80, 60], 'hsv_high': [25, 255, 255],
                       'minimum_plug_pixels': 15, 'plug_axis_hint': [0, 1],
                       'near_error_px': 65, 'near_gap_px': 55, 'align_tolerance_px': 3},
            'servo': {'probe_rad': 0.006, 'step_rad': 0.01, 'gain': 0.5, 'damping': 0.1,
                      'motion_rad_s': 0.15, 'settle_s': 0.15}}
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('prepare', help='Export selected recording windows; no arm requests')
    export.add_argument('dataset', type=pathlib.Path)
    export.add_argument('--window', action='append', required=True,
                        help='episode:start_frame:end_frame; repeat to average')
    export.add_argument('--output', type=pathlib.Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('plan', type=pathlib.Path)
    run.add_argument('--server', default='http://127.0.0.1:8692')
    run.add_argument('--config', type=pathlib.Path, help='Station model/TCP config, needed for execution')
    run.add_argument('--execute', action='store_true',
                     help='Wait for near-port detection, take over, align and push')
    run.add_argument('--check-image', type=pathlib.Path, help='Offline detector evaluation; no arm requests')
    run.add_argument('--policy-stop-url', help='POST {} and await completion before commanding the arm')
    run.add_argument('--policy-status-url', help='After stop, GET this JSON until the selected busy field is false')
    run.add_argument('--policy-busy-path', default='busy', help='Dot path in status JSON, e.g. dream.right.busy')
    run.add_argument('--quiet', action='store_true',
                     help='Only final JSON on stdout; progress stays in the event file')
    run.add_argument('--output', type=pathlib.Path, default=pathlib.Path('inspection/usb-insertion-run'))
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        plan = prepare(args.dataset, args.window, args.output)
        print(json.dumps({'plan': str(args.output / 'plan.json'), 'demonstrations': len(plan['demonstrations'])}))
        return
    plan = json.loads(args.plan.read_text())
    reference = cv2.imread(str(args.plan.parent / plan['reference']))
    if reference is None:
        parser.error('Reference image could not be read')
    vision = InsertionVision(reference, plan['vision'])
    args.output.mkdir(parents=True, exist_ok=True)

    def report(phase, status, image):
        event = {'phase': phase, 'time': time.time(), **status}
        with (args.output / 'events.jsonl').open('a') as log:
            log.write(json.dumps(event) + '\n')
        if not args.quiet:
            print(json.dumps(event), file=sys.stderr, flush=True)
        if image is not None:
            cv2.imwrite(str(args.output / 'alignment.jpg'), image)

    if args.check_image:
        image = cv2.imread(str(args.check_image))
        if image is None:
            parser.error('Check image could not be read')
        _, status, annotated = vision.measure(image)
        report('check', status, annotated)
        print(json.dumps(status), flush=True)
        return
    if not args.execute:
        print(json.dumps({'prepared': True, 'demonstrations': len(plan['demonstrations']),
                          'push_seconds': plan['push_seconds'], 'execute': False}))
        return
    if args.config is None:
        parser.error('--execute requires the existing station --config for calibrated TCP retargeting')
    if args.policy_status_url and not args.policy_stop_url:
        parser.error('--policy-status-url accompanies --policy-stop-url')

    def stop_policy():
        request = urllib.request.Request(args.policy_stop_url, data=b'{}',
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            response.read()
        if args.policy_status_url:
            while True:
                with urllib.request.urlopen(args.policy_status_url) as response:
                    busy = lookup(json.load(response), args.policy_busy_path)
                if busy is False:
                    break
                if busy is not True:
                    raise ValueError('Policy busy path must resolve to a boolean')
                time.sleep(plan['poll_seconds'])

    result = execute(HttpArm(args.server), vision, plan, ToolKinematics.from_config(args.config, 'right'),
                     report, stop_policy if args.policy_stop_url else None)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
