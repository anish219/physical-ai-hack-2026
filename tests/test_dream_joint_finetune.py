"""Checkpoint schema and arm-isolation regressions; no hardware or Dream API calls."""

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dream_hackathon.dream_caller import (
    JointModelClient,
    StationCaller,
    build_observation,
    open_client,
)


def body(side="right"):
    parts = {
        f"{side}_arm": {"kind": "arm", "side": side},
        f"{side}_gripper": {"kind": "gripper", "side": side, "closed": 0, "open": 100},
    }
    return {
        "state": {f"{side}_arm": [0, 6], f"{side}_gripper": [6, 7]},
        "action": {f"{side}_arm": [0, 6], f"{side}_gripper": [6, 7]},
        "video": {"scene": "observation.images.scene", "wrist_r" if side == "right" else "wrist_l": "wrist"},
        "parts": {"state": copy.deepcopy(parts), "action": copy.deepcopy(parts)},
        "actions": "absolute",
    }


class Client:
    def __init__(self, side="right"):
        self.side, self.calls, self.closed = side, 0, False

    def get_models(self):
        return [SimpleNamespace(id="dream-1-ft:test", servable=True, embodiment_config=body(self.side))]

    def policy(self, config):
        self.config = config

    def set_observation(self, frames, state, *, timestamp):
        self.frames, self.state, self.timestamp = frames, state, timestamp

    def get_action(self):
        self.calls += 1
        return {
            "per_key": {
                f"action.{self.side}_arm": [[0.1] * 6, [0.2] * 6, [0.3] * 6],
                f"action.{self.side}_gripper": [[0], [50], [100]],
            }
        }

    def close(self):
        self.closed = True


@pytest.mark.parametrize("side", ["left", "right"])
def test_native_wire_exact_cameras_state_and_mm_actions(side):
    raw = Client(side)
    client = JointModelClient(raw, body(side))
    frames, state = build_observation(side, {"q": [0.1] * 6, "grip": 0.7}, b"scene", b"wrist", b"blank", "unplug")
    client.set_observation(frames, state, timestamp=1.0)
    wrist = "wrist_l" if side == "left" else "wrist_r"
    assert set(raw.frames) == {"video.scene", f"video.{wrist}"}
    assert set(raw.state) == {"prompt", f"state.{side}_arm", f"state.{side}_gripper"}
    assert raw.state[f"state.{side}_gripper"] == [70.0]
    rows = client.get_action()["q_l_chunk" if side == "left" else "q_r_chunk"]
    assert [row[6] for row in rows] == [0.0, 0.5, 1.0]
    assert rows[1][:6] == [0.2] * 6


def test_native_policy_does_not_enable_eef_ik(monkeypatch):
    raw = Client()
    monkeypatch.setitem(sys.modules, "dream", SimpleNamespace(DreamClient=lambda: raw))
    adapter = open_client("dream-1-ft:test", 20)
    assert isinstance(adapter, JointModelClient)
    assert raw.config == {"checkpoint": "dream-1-ft:test", "embodiment": body(), "action_hz": 20}
    assert "ik" not in raw.config


def test_base_policy_keeps_existing_bimanual_eef_configuration(monkeypatch):
    raw = Client()
    monkeypatch.setitem(sys.modules, "dream", SimpleNamespace(DreamClient=lambda: raw))
    assert open_client("dreamlabs/dream-1-beta", 20) is raw
    assert raw.config["ik"] == {"arm": "metal", "head": "eef"}
    assert set(raw.config["embodiment"]["video"]) >= {"scene", "wrist_l", "wrist_r"}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 101])
def test_bad_gripper_reply_is_rejected(value):
    raw = Client()
    raw.get_action = lambda: {"per_key": {"action.right_arm": [[0] * 6], "action.right_gripper": [[value]]}}
    with pytest.raises(ValueError):
        JointModelClient(raw, body()).get_action()


@pytest.mark.parametrize("selection", ["left", "both"])
def test_right_checkpoint_cannot_execute_on_left_or_both(selection):
    class Endpoint:
        url = "fake"

        def request(self, *args):
            pytest.fail("Wrong-arm checkpoint must never send a command")

    raw = Client()
    adapter = JointModelClient(raw, body())
    caller = StationCaller(selection, {"left": Endpoint(), "right": Endpoint()}, client_factory=lambda *_: adapter)
    caller.sample = ({"q": [0] * 6, "grip": 0.5}, b"scene", b"wrist")
    caller.start({"task": "unplug", "checkpoint": "dream-1-ft:test"}, run=True)
    caller.worker.join(timeout=2)
    assert not caller.worker.is_alive()
    assert "controls only right" in caller.snapshot()["error"]
    assert raw.calls == 0
    assert raw.closed


def test_native_right_inspect_never_sends_commands_or_reads_left():
    class Endpoint:
        url = "fake"

        def request(self, *args):
            pytest.fail("Inspect must not write or read hardware")

    raw = Client()
    caller = StationCaller("right", {"right": Endpoint()}, client_factory=lambda *_: JointModelClient(raw, body()))
    caller.sample = ({"q": [0.1] * 6, "grip": 0.7}, b"scene", b"wrist")
    caller.start({"task": "unplug", "checkpoint": "dream-1-ft:test"}, run=False)
    caller.worker.join(timeout=2)
    assert not caller.worker.is_alive()
    assert not caller.snapshot()["error"]
    assert caller.snapshot()["commands"] == 0
    assert raw.calls == 1
    assert np.array(caller.snapshot()["rows"]).shape == (3, 7)
