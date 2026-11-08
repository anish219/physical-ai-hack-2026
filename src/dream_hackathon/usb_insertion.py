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
    """Reacquire the port across viewpoints and align it relative to the live held USB."""

    def __init__(self, reference, settings, views=()):
        from dream_hackathon.usb_localization import HeldUSB, PortRegistration

        self.reference, self.settings = reference, settings
        self.registration = PortRegistration(reference, settings)
        for view in views:
            self.registration.add_view(view)
        self.usb = HeldUSB(settings)
        self.anchors = np.float32(settings['port_polygon'])
        tip, basis, _ = self.usb.locate(reference)
        self.target = ((self.anchors - tip) @ basis).ravel()

    def measure(self, image):
        frame = cv2.resize(image, self.reference.shape[1::-1])
        anchors, port, port_status = self.registration.locate(frame)
        tip, basis, plug_status = self.usb.locate(frame)
        features = ((anchors - tip) @ basis).ravel()
        errors = np.linalg.norm((features - self.target).reshape(-1, 2), axis=1)
        gap = float(np.linalg.norm(port - tip))
        status = {'known': True, 'near_opening': bool(errors.max() <= self.settings['near_error_px']
                                                    and gap <= self.settings['near_gap_px']),
                  'aligned': bool(errors.max() <= self.settings['align_tolerance_px']),
                  'error_px': float(errors.max()), 'tip_port_gap_px': gap,
                  **port_status, **plug_status, 'port_point': port.tolist(), 'features': features.tolist()}
        cv2.polylines(frame, [np.rint(anchors).astype(np.int32)], True, (0, 255, 0), 1)
        cv2.arrowedLine(frame, tuple(np.rint(tip + basis[:, 1] * 20).astype(int)),
                        tuple(np.rint(tip).astype(int)), (0, 128, 255), 2)
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
        camera_image = arm.image()
        try:
            _, status, image = vision.measure(camera_image)
        except ValueError as error:
            report('waiting', {'known': False, 'reason': str(error)}, camera_image)
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
    views = [cv2.imread(str(args.plan.parent / name)) for name in plan.get('view_images', [])]
    if any(view is None for view in views):
        parser.error('A port view image could not be read')
    vision = InsertionVision(reference, plan['vision'], views)
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
