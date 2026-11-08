"""Shared sequence definitions and timed execution; fake arm/API workers only."""

import copy
import time
from types import SimpleNamespace

import pytest
from test_armserver import request
from test_dream_caller import Model, fake_arm, listener, wait_for
from test_unified_control import make_node

from dream_hackathon.dream_caller import ArmEndpoint, StationCaller
from dream_hackathon.sequences import SequenceManager, validate_definition
from dream_hackathon.so101_dashboard import Controls
from dream_hackathon.unified_control import UnifiedController, UnifiedPage


def definition(prompt="do the task", *, wait=0.02):
    return {
        "phases": [
            {
                "id": "phase1",
                "title": "First phase",
                "steps": [
                    {"id": "wait1", "label": "Wait", "type": "wait", "durationSeconds": wait},
                    {
                        "id": "dream1",
                        "label": "Dream task",
                        "type": "dream",
                        "durationSeconds": 1,
                        "prompt": prompt,
                        "modelId": "test/checkpoint",
                    },
                ],
            }
        ]
    }


class Caller:
    def __init__(self, side):
        self.sides = ("left", "right") if side == "both" else (side,)
        self.busy = False
        self.calls = []
        self.cancels = 0

    def snapshot(self):
        return dict(busy=self.busy, observation={}, error="", observation_error="", running=self.busy)

    def connect(self):
        pass

    def start(self, values, *, run):
        self.calls.append((dict(values), run))
        self.busy = True

    def cancel_playback(self):
        self.cancels += 1
        self.busy = False


def manager(path=None):
    nodes = {
        side: SimpleNamespace(snapshot=lambda: {"arm_open": False}, cancel_home=lambda: None)
        for side in ("left", "right")
    }
    callers = {side: Caller(side) for side in ("left", "right", "both")}
    events = []
    return SequenceManager(callers, nodes, events.append, path), callers, nodes, events


@pytest.mark.parametrize("selection", ["left", "right", "both"])
def test_phases_wait_then_dream_using_selected_worker_and_saved_run_snapshot(selection):
    sequences, callers, _, _ = manager()
    original = definition("original task", wait=0.08)
    sequences.update(selection, original)
    try:
        sequences.start(selection)
        wait_for(lambda: sequences.runs[selection].snapshot()["step_index"] == 0)
        assert not any(caller.calls for caller in callers.values())
        edited = copy.deepcopy(original)
        edited["phases"][0]["steps"][1]["prompt"] = "edited for the next run"
        sequences.update(selection, edited)
        wait_for(lambda: not sequences.runs[selection].snapshot()["busy"])
        assert sequences.runs[selection].snapshot()["state"] == "complete"
        assert callers[selection].calls == [({"task": "original task", "checkpoint": "test/checkpoint"}, True)]
        assert not callers[selection].busy
        assert all(not caller.calls for side, caller in callers.items() if side != selection)
    finally:
        sequences.close()


def test_wait_steps_send_no_calls_and_stop_prevents_later_dream():
    sequences, callers, _, _ = manager()
    sequences.update("left", definition(wait=1))
    try:
        sequences.start("left")
        wait_for(lambda: sequences.runs["left"].snapshot()["step_index"] == 0)
        sequences.stop_sides(["left"])
        wait_for(lambda: not sequences.runs["left"].snapshot()["busy"])
        assert sequences.runs["left"].snapshot()["state"] == "stopped"
        assert not any(caller.calls for caller in callers.values())
    finally:
        sequences.close()


def test_definitions_are_shared_and_persist_original_phase_format(tmp_path):
    path = tmp_path / "sequences.json"
    sequences, _, _, events = manager(path)
    plan = definition()
    plan["phases"][0]["steps"][0]["state"] = "active"
    sequences.update("both", plan)
    restored, _, _, _ = manager(path)
    expected = validate_definition(plan)
    assert restored.snapshot()["both"]["definition"] == expected
    assert "state" not in expected["phases"][0]["steps"][0]
    assert events[-1]["type"] == "sequences"
    assert events[-1]["sequences"]["both"]["definition"] == expected
    invalid = copy.deepcopy(expected)
    invalid["phases"][0]["steps"][1]["durationSeconds"] = float("nan")
    with pytest.raises(ValueError, match="window"):
        sequences.update("both", invalid)
    assert sequences.snapshot()["both"]["definition"] == expected


def test_independent_sequences_run_concurrently():
    sequences, callers, _, _ = manager()
    try:
        for side in ("left", "right"):
            sequences.update(side, definition(side, wait=0))
            sequences.start(side)
        wait_for(lambda: callers["left"].busy and callers["right"].busy)
        sequences.stop_sides(["left"])
        assert not callers["left"].busy and callers["right"].busy
        wait_for(lambda: not sequences.runs["right"].snapshot()["busy"])
        assert sequences.runs["right"].snapshot()["state"] == "complete"
    finally:
        sequences.close()


def test_sequence_selection_and_http_save_run_stop():
    sequences, callers, nodes, _ = manager()
    leader = SimpleNamespace(side=None, state=lambda: Controls(1, "none").snapshot())
    controller = UnifiedController(nodes, leader, callers)
    page = UnifiedPage(controller, 0)
    base = f"http://127.0.0.1:{page.port}"
    try:
        assert request(base, "/api/sequences/left/save", {"definition": definition()})[0] == 200
        assert request(base, "/api/sequences/left/run", {})[0] == 400
        assert request(base, "/api/arms/left/input", {"source": "sequence"})[0] == 200
        assert request(base, "/api/sequences/left/run", {})[0] == 200
        wait_for(lambda: callers["left"].busy)
        assert request(base, "/api/sequences/left/stop", {})[0] == 200
        wait_for(lambda: not controller.sequences.runs["left"].snapshot()["busy"])
        controller.select_both("sequence")
        assert controller.inputs == {"left": None, "right": None}
        assert controller.combined_input == "sequence"
        controller.select_input("right", "dream")
        assert controller.combined_input is None and controller.inputs["right"] == "dream"
    finally:
        page.close()


def test_handoff_discards_pending_sequence_api_reply_and_never_starts_next_step(tmp_path):
    node, _ = make_node(tmp_path, "left")
    other, _ = make_node(tmp_path, "right")
    nodes = {"left": node, "right": other}
    model, logs = Model(), {"left": [], "right": []}
    leader = SimpleNamespace(side=None, state=lambda: Controls(1, "none").snapshot())
    with listener(fake_arm(logs["left"])) as left, listener(fake_arm(logs["right"])) as right:
        endpoints = {"left": ArmEndpoint(left), "right": ArmEndpoint(right)}
        callers = {
            side: StationCaller(side, endpoints, client_factory=lambda *_: model) for side in ("left", "right", "both")
        }
        controller = UnifiedController(nodes, leader, callers)
        try:
            controller.select_input("left", "sequence")
            controller.sequence_action("left", "save", {"definition": definition(wait=0)})
            controller.sequence_action("left", "run", {})
            assert model.entered.wait(2)
            controller.select_input("left", "dream")
            model.release.set()
            wait_for(lambda: not callers["left"].snapshot()["busy"])
            wait_for(lambda: not controller.sequences.runs["left"].snapshot()["busy"])
            time.sleep(0.05)
            assert not any(path == "/target" for log in logs.values() for path, _ in log)
            assert controller.sequences.runs["left"].snapshot()["state"] == "stopped"
            assert len(model.ticks) >= 1
        finally:
            model.release.set()
            controller.sequences.close()
            for caller in callers.values():
                caller.close()
            for arm in nodes.values():
                arm.close()
