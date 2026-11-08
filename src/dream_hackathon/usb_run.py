"""One headless run: right start pose -> approach API -> continuous visual insertion -> exit."""

import argparse
import json
import pathlib
import sys
import time
import urllib.request

import cv2
import numpy as np

from dream_hackathon.bundle_pull import HttpArm, ToolKinematics, move_to
from dream_hackathon.usb_insertion import InsertionVision, execute


class ControllerAPI:
    """Reuse the existing right-only Dream worker and its configured credentials."""

    def __init__(self, url, poll_seconds=0.1, *, policy_url=None, keep_input=False):
        self.url, self.poll_seconds = url.rstrip('/'), poll_seconds
        self.policy_url, self.keep_input = policy_url, keep_input

    def request(self, path, body=None, *, base=None):
        request = urllib.request.Request((base or self.url).rstrip('/') + path,
                                         data=None if body is None else json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            return json.load(response)

    def state(self):
        return self.request('/api/state')

    def select(self):
        if not self.keep_input:
            self.request('/api/arms/right/input', {'source': 'dream'})

    def policy(self, action, body=None):
        if self.policy_url:
            return self.request('/api/right/' + action, body, base=self.policy_url)
        return self.request('/api/dream/right/' + action, body)

    def policy_state(self):
        if self.policy_url:
            return self.policy('state')
        return self.state()['dream']['right']

    def stop(self):
        self.policy('stop', {})
        while self.policy_state()['busy']:
            time.sleep(self.poll_seconds)

    def start(self, task, checkpoint):
        self.policy('connect', {})
        while True:
            state = self.policy_state()
            if state.get('observation_error'):
                raise RuntimeError(state['observation_error'])
            if state.get('observation') is not None:
                break
            time.sleep(self.poll_seconds)
        self.policy('run', {'task': task, 'checkpoint': checkpoint})


def run(backend, arm, vision, plan, kinematics, settings, report, *, sleep=time.sleep, align_only=False):
    """Run one complete attempt; all action dependencies are injectable for offline tests."""
    initial = backend.state()
    draft = initial.get('api_drafts', {}).get('right', {})
    current = initial['dream']['right']
    task = settings.get('task') or draft.get('task') or current.get('task')
    checkpoint = settings.get('checkpoint') or draft.get('checkpoint') or current.get('checkpoint')
    if not task or not checkpoint:
        raise ValueError('Set a right-arm task/checkpoint in the profile, arguments, or existing API draft')
    start = settings.get('start_q')
    if start is None:
        start = initial['arms']['right'].get('zero_joints')
    if start is None:
        raise ValueError('No saved right start pose; supply start_q or --start-q with six joint radians')
    start = np.asarray(start, float)
    if start.shape != (6,) or not np.isfinite(start).all():
        raise ValueError('Start pose must contain six finite joint radians')
    backend.select()
    backend.stop()
    _q, grip = arm.state()
    if settings.get('start_grip') is not None:
        grip = settings['start_grip']
    report('returning_to_start', {'q': start.tolist(), 'grip': float(grip)}, None)
    move_to(arm, start, grip, settings['start_speed_rad_s'], hz=plan['hz'])
    while True:
        measured, measured_grip = arm.state()
        arrived = float(np.max(np.abs(measured - start))) <= settings['start_tolerance_rad']
        if settings.get('start_grip') is not None:
            arrived = arrived and abs(measured_grip - grip) <= settings['start_grip_tolerance']
        if arrived:
            break
        sleep(settings['poll_seconds'])
    report('start_pose_reached', {'q': measured.tolist()}, None)
    report('starting_api', {'task': task, 'checkpoint': checkpoint}, None)
    backend.start(task, checkpoint)
    report('visual_checker_started', {}, None)
    policy_yielded = False

    def yield_policy():
        nonlocal policy_yielded
        if not policy_yielded:
            backend.stop()
            policy_yielded = True
            report('api_yielded', {}, None)

    try:
        result = execute(arm, vision, plan, kinematics, report, policy_stop=yield_policy,
                         align_only=align_only)
    except KeyboardInterrupt:
        # This is the operator's explicit interruption, not an automatic cutoff.
        yield_policy()
        raise
    yield_policy()
    result = {**result, 'run_complete': True, 'api_stopped': True, 'start_q': start.tolist()}
    report('run_complete', result, None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=pathlib.Path, default=pathlib.Path('configs/usb_insertion_run.json'))
    parser.add_argument('--station-config', type=pathlib.Path, default=pathlib.Path('configs/station.yaml'))
    parser.add_argument('--start-q', type=float, nargs=6)
    parser.add_argument('--start-grip', type=float)
    parser.add_argument('--task')
    parser.add_argument('--checkpoint')
    parser.add_argument('--script-input', action='store_true',
                        help='Keep Script selected; use the direct Dream worker')
    parser.add_argument('--policy-url', default='http://127.0.0.1:8084')
    parser.add_argument('--align-only', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args(argv)
    settings = json.loads(args.profile.read_text())
    for name in ('start_q', 'start_grip', 'task', 'checkpoint'):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    if settings.get('start_grip') is not None and not 0 <= settings['start_grip'] <= 1:
        parser.error('start_grip must be normalized between zero and one')
    plan_path = (args.profile.parent / settings['plan']).resolve()
    plan = json.loads(plan_path.read_text())
    reference = cv2.imread(str(plan_path.parent / plan['reference']))
    views = [cv2.imread(str(plan_path.parent / name)) for name in plan.get('view_images', [])]
    if reference is None or any(view is None for view in views):
        parser.error('Could not read the insertion reference/view images')
    vision = InsertionVision(reference, plan['vision'], views)
    kinematics = ToolKinematics.from_config(args.station_config, 'right')
    output = args.output or pathlib.Path('inspection/usb-insertion-runs') / time.strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, exist_ok=True)

    def report(phase, status, image):
        event = {'phase': phase, 'time': time.time(), **status}
        with (output / 'events.jsonl').open('a') as log:
            log.write(json.dumps(event) + '\n')
        if not args.quiet:
            print(json.dumps(event), file=sys.stderr, flush=True)
        if image is not None:
            cv2.imwrite(str(output / 'alignment.jpg'), image)

    result = run(ControllerAPI(settings['controller_url'], settings['poll_seconds'],
                               policy_url=args.policy_url if args.script_input else None,
                               keep_input=args.script_input),
                 HttpArm(settings['right_url']), vision, plan, kinematics, settings, report,
                 align_only=args.align_only)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
