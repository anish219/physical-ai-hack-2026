"""Complete right-only run ordering, pose completion and API ownership handoff."""

import io
import json
import pathlib

import numpy as np
import pytest

from dream_hackathon import usb_run

ROOT = pathlib.Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT / 'configs/usb_insertion_run.json').read_text())
START = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])


class Arm:
    def __init__(self):
        self.q, self.grip = np.zeros(6), 0.12
        self.pending = []

    def state(self):
        if self.pending:
            self.q = self.pending.pop(0)
        return self.q.copy(), self.grip


class Backend:
    def __init__(self, events, arm):
        self.events, self.arm = events, arm

    def state(self):
        return {'arms': {'right': {'zero_joints': START.tolist()}},
                'dream': {'right': {'task': 'current task', 'checkpoint': 'current model'}},
                'api_drafts': {'right': {'task': 'draft task', 'checkpoint': 'draft model'}}}

    def select(self):
        self.events.append('select_right_dream')

    def stop(self):
        self.events.append('stop_api')

    def start(self, task, checkpoint):
        self.events.append(('api_start', task, checkpoint, self.arm.q.copy()))


@pytest.mark.parametrize('align_only', [False, True])
def test_complete_run_waits_for_start_then_starts_api_then_yields_before_script(monkeypatch, align_only):
    events, arm = [], Arm()
    backend = Backend(events, arm)

    def move(arm, q, grip, speed, **kwargs):
        events.append(('move_to_start', q.copy(), grip))
        arm.pending = [q + 0.05, q + 0.02, q.copy()]

    def visual(arm, vision, plan, kinematics, report, *, policy_stop, align_only):
        events.append(('checker_started', align_only))
        policy_stop()
        events.append('scripted_motion')
        return {'script_complete': True, 'descent_complete': not align_only, 'insertion_verified': False}

    monkeypatch.setattr(usb_run, 'move_to', move)
    monkeypatch.setattr(usb_run, 'execute', visual)
    waits = []
    result = usb_run.run(backend, arm, None, {'hz': 30}, None, SETTINGS,
                         lambda *args: None, sleep=waits.append, align_only=align_only)
    assert events[:2] == ['select_right_dream', 'stop_api']
    assert events[2][0] == 'move_to_start' and events[2][2] == 0.12
    np.testing.assert_allclose(events[2][1], START)
    assert events[3][0:3] == ('api_start', 'draft task', 'draft model')
    np.testing.assert_allclose(events[3][3], START)
    assert events[4:] == [('checker_started', align_only), 'stop_api', 'scripted_motion']
    assert len(waits) == 2
    assert result['run_complete'] and result['api_stopped'] and not result['insertion_verified']


def test_explicit_start_grip_task_checkpoint_override_saved_defaults(monkeypatch):
    events, arm = [], Arm()

    def move(arm, q, grip, speed, **kwargs):
        arm.q, arm.grip = q.copy(), grip

    def visual(*args, **kwargs):
        kwargs['policy_stop']()
        return {'script_complete': True}

    monkeypatch.setattr(usb_run, 'move_to', move)
    monkeypatch.setattr(usb_run, 'execute', visual)
    settings = {**SETTINGS, 'start_q': [0.2] * 6, 'start_grip': 0.8, 'task': 'grab', 'checkpoint': 'trained'}
    result = usb_run.run(Backend(events, arm), arm, None, {'hz': 30}, None, settings, lambda *args: None)
    assert events[2][:3] == ('api_start', 'grab', 'trained')
    assert arm.grip == 0.8 and result['start_q'] == [0.2] * 6


def test_operator_interrupt_stops_the_owned_policy(monkeypatch):
    events, arm = [], Arm()
    monkeypatch.setattr(usb_run, 'move_to', lambda arm, q, *args, **kwargs: setattr(arm, 'q', q))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(usb_run, 'execute', interrupted)
    with pytest.raises(KeyboardInterrupt):
        usb_run.run(Backend(events, arm), arm, None, {'hz': 30}, None, SETTINGS, lambda *args: None)
    assert events[-1] == 'stop_api'
    assert events.count('stop_api') == 2


def test_api_client_uses_only_right_endpoints_and_awaits_observation_and_stop(monkeypatch):
    requests = []
    replies = iter([{}, {'dream': {'right': {'observation': None}}},
                    {'dream': {'right': {'observation': {}, 'error': 'old model error'}}}, {},
                    {}, {'dream': {'right': {'busy': True}}}, {'dream': {'right': {'busy': False}}}])

    def request(value):
        requests.append(value)
        return io.BytesIO(json.dumps(next(replies)).encode())

    monkeypatch.setattr(usb_run.urllib.request, 'urlopen', request)
    monkeypatch.setattr(usb_run.time, 'sleep', lambda *args: None)
    api = usb_run.ControllerAPI('http://controller')
    api.start('grab', 'trained')
    api.stop()
    assert [r.full_url for r in requests] == [
        'http://controller/api/dream/right/connect', 'http://controller/api/state',
        'http://controller/api/state', 'http://controller/api/dream/right/run',
        'http://controller/api/dream/right/stop', 'http://controller/api/state', 'http://controller/api/state']
    assert json.loads(requests[3].data) == {'task': 'grab', 'checkpoint': 'trained'}


def test_cli_emits_single_result_and_saves_it_without_direct_network_in_test(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(usb_run, 'InsertionVision', lambda *args: None)
    monkeypatch.setattr(usb_run.ToolKinematics, 'from_config', lambda *args: None)

    def run(*args, **kwargs):
        args[6]('run_complete', {'run_complete': True}, None)
        return {'run_complete': True, 'api_stopped': True}

    monkeypatch.setattr(usb_run, 'run', run)
    usb_run.main(['--profile', str(ROOT / 'configs/usb_insertion_run.json'), '--output', str(tmp_path), '--quiet'])
    result = json.loads(capsys.readouterr().out)
    assert result['run_complete'] and result['api_stopped']
    assert json.loads((tmp_path / 'result.json').read_text()) == result
