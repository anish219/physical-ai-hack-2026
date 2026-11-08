"""Recorded-image detection, relative taught motion and policy/script handoff."""

import json
import pathlib

import cv2
import numpy as np
import pytest

from dream_hackathon import usb_insertion as insertion

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'recordings/usb-insertion'


def detector():
    plan = json.loads((ASSETS / 'plan.json').read_text())
    reference = cv2.imread(str(ASSETS / plan['reference']))
    return insertion.InsertionVision(reference, plan['vision'])


def test_recorded_near_frame_triggers_but_requires_further_alignment():
    vision = detector()
    _, status, _ = vision.measure(cv2.imread(str(ASSETS / 'near-frame-460.png')))
    assert status['near_opening'] and not status['aligned']
    assert status['error_px'] > vision.settings['align_tolerance_px']


def test_earlier_recorded_frame_does_not_take_over():
    _, status, _ = detector().measure(cv2.imread(str(ASSETS / 'before-frame-450.png')))
    assert not status['near_opening']


def test_reference_is_aligned_and_missing_plug_does_not_pass():
    vision = detector()
    _, status, _ = vision.measure(vision.reference)
    assert status['near_opening'] and status['aligned']
    gray = cv2.cvtColor(cv2.cvtColor(vision.reference, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    with pytest.raises(ValueError, match='USB'):
        vision.measure(gray)


def test_common_image_translation_does_not_change_plug_relative_alignment():
    vision = detector()
    shifted = cv2.warpAffine(vision.reference, np.float32([[1, 0, 2], [0, 1, 3]]), (640, 360))
    features, status, _ = vision.measure(shifted)
    np.testing.assert_allclose(features, vision.target, atol=0.6)
    assert status['aligned']


class CartesianKinematics:
    def fk(self, q):
        transform = np.eye(4)
        transform[:3, 3] = q[:3]
        transform[:3, :3] = cv2.Rodrigues(np.asarray(q[3:], float))[0]
        return transform

    def solve(self, transform, seed):
        return np.r_[transform[:3, 3], cv2.Rodrigues(transform[:3, :3])[0].ravel()]


def demo(origin, delta):
    return {'rows': [{'t': 0, 'q': [*origin, 0, 0, 0], 'grip': 90},
                     {'t': 2, 'q': (np.r_[origin, 0, 0, 0] + np.r_[delta, 0, 0, 0]).tolist(), 'grip': 100}]}


def test_average_relative_push_starts_at_live_pose_and_keeps_live_grip():
    demos = [demo([1, 2, 3], [0, 0, -0.01]), demo([-4, 7, 9], [0, 0, -0.03])]
    aligned = [0.2, 0.3, 0.4, 0, 0, 0]
    rows = insertion.push_rows(demos, aligned, 0.13, CartesianKinematics(), 11, 3)
    np.testing.assert_allclose(rows[0]['q'], aligned)
    np.testing.assert_allclose(rows[-1]['q'], [0.2, 0.3, 0.38, 0, 0, 0])
    assert all(row['grip'] == 0.13 for row in rows)
    assert rows[-1]['t'] == 3


def test_averaged_orientation_is_a_proper_rotation():
    rotations = [cv2.Rodrigues(np.array([0., 0., angle]))[0] for angle in [-0.2, 0.4]]
    result = insertion.mean_rotation(rotations)
    np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-12)
    assert np.linalg.det(result) == pytest.approx(1)
    np.testing.assert_allclose(cv2.Rodrigues(result)[0].ravel(), [0, 0, 0.1], atol=1e-12)


def test_policy_yields_before_first_command_and_no_grab_or_release_is_added(monkeypatch):
    events = []

    class Arm:
        def image(self):
            return None

        def state(self):
            return np.zeros(6), 0.12

        def command(self, q, grip):
            events.append(('command', grip))

    class Vision:
        calls = 0

        def measure(self, image):
            self.calls += 1
            if self.calls == 1:
                raise ValueError('not visible')
            return None, {'near_opening': self.calls >= 3, 'aligned': True}, None

    monkeypatch.setattr(insertion, 'servo', lambda *args: np.zeros(6))
    monkeypatch.setattr(insertion, 'play', lambda arm, rows, hz: events.append(('play', rows)))
    plan = {'poll_seconds': 0, 'servo': {}, 'hz': 30, 'demonstrations': [demo([0, 0, 0], [0, 0, -0.01])],
            'push_samples': 3, 'push_seconds': 2, 'release_grip': None}
    result = insertion.execute(Arm(), Vision(), plan, CartesianKinematics(),
                               lambda *args: None, lambda: events.append(('policy_stop',)), sleep=lambda t: None)
    assert events[0] == ('policy_stop',)
    assert events[1] == ('command', 0.12)
    assert events[2][0] == 'play'
    assert len(events) == 3
    assert all(row['grip'] == 0.12 for row in events[2][1])
    assert result['push_complete'] and not result['insertion_verified'] and not result['release_commanded']


def test_offline_inspection_never_constructs_arm_endpoint(monkeypatch, tmp_path):
    def forbidden(*args):
        raise AssertionError('Offline mode contacted an arm')

    monkeypatch.setattr(insertion, 'HttpArm', forbidden)
    insertion.main(['run', str(ASSETS / 'plan.json'), '--output', str(tmp_path)])
    insertion.main(['run', str(ASSETS / 'plan.json'), '--check-image', str(ASSETS / 'reference.png'),
                    '--output', str(tmp_path)])


def test_cli_waits_for_async_policy_to_yield_and_emits_one_result(monkeypatch, tmp_path, capsys):
    import io

    requests = []
    replies = iter([{}, {'dream': {'right': {'busy': True}}}, {'dream': {'right': {'busy': False}}}])

    def request(value):
        requests.append(value)
        return io.BytesIO(json.dumps(next(replies)).encode())

    def execute(*args):
        args[-1]()  # Wait for the policy-stop callback before becoming the target producer.
        requests.append('motion')
        return {'push_complete': True, 'insertion_verified': False}

    monkeypatch.setattr(insertion.urllib.request, 'urlopen', request)
    monkeypatch.setattr(insertion, 'execute', execute)
    monkeypatch.setattr(insertion.ToolKinematics, 'from_config', lambda *args: None)
    monkeypatch.setattr(insertion.time, 'sleep', lambda *args: None)
    insertion.main(['run', str(ASSETS / 'plan.json'), '--execute', '--config', 'unused.yaml', '--quiet',
                    '--policy-stop-url', 'http://policy/right/stop',
                    '--policy-status-url', 'http://policy/state', '--policy-busy-path', 'dream.right.busy',
                    '--output', str(tmp_path)])
    assert requests[0].full_url == 'http://policy/right/stop'
    assert requests[0].data == b'{}'
    assert requests[1:3] == ['http://policy/state', 'http://policy/state']
    assert requests[3] == 'motion'
    assert json.loads(capsys.readouterr().out) == {'push_complete': True, 'insertion_verified': False}
