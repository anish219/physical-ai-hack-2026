"""Behavior parity with the old NUC server, plus independent HTTP/arm lifetimes. No hardware."""

import json
import pathlib
import threading
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, contextmanager
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from dream_hackathon import armserver


class FakeArm:
    def __init__(self, lag=0.0):
        self.q = np.zeros(6)
        self.grip = 0.0
        self.writes = []
        self.lag = lag
        self.closed = False

    def read(self):
        return SimpleNamespace(positions_rad=self.q.copy(), gripper=self.grip)

    def write(self, command):
        self.writes.append((command.positions_rad.copy(), command.gripper))
        self.q = command.positions_rad.copy()
        self.q[1] -= self.lag() if callable(self.lag) else self.lag
        self.grip = command.gripper

    def close(self):
        self.closed = True


def legacy(control, clock):
    """Only copied pure control/HTTP functions are evaluated, not the old module's hardware imports."""
    namespace = {
        "np": np,
        "json": json,
        "BaseHTTPRequestHandler": armserver.BaseHTTPRequestHandler,
        "time": clock,
        "types": SimpleNamespace(ArmCommand=SimpleNamespace),
        "st": control.state,
        "lock": control.lock,
        "frames": control.frames,
        **{name: getattr(armserver, name) for name in ("TICK", "MAX_RATE", "ABORT_RAD", "STALL_RAD")},
    }
    source = pathlib.Path(__file__).with_name("fixtures") / "legacy_armserver.txt"
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


class EndTraceError(Exception):
    pass


def trace(scenario, reference):
    control = armserver.ArmControl()
    tick = 0

    def sleep(_duration):
        nonlocal tick
        tick += 1
        if tick == 1:
            control.post(
                "/move",
                {"q": [0.3, -0.5, 0.1, 0, 0, 0], "grip": 0.8, "rate": 2, "stop_on_stall": scenario == "stall"},
            )
        if scenario == "motion" and tick == 10:
            control.post("/stop", {})
        if scenario == "motion" and tick == 15:
            control.post("/move", {"q": [0.2] * 6, "grip": 0.2, "rate": 0.2})
        if scenario == "motion" and tick == 25:
            control.post("/quit", {})
        if tick == 60:
            raise EndTraceError

    clock = SimpleNamespace(sleep=sleep, time=lambda: tick * armserver.TICK)
    arm = FakeArm(lag=lambda: 0.4 if scenario == "lag" else (0.02 if scenario == "stall" and tick > 12 else 0))
    try:
        if reference:
            legacy(control, clock).control(arm)
        else:
            control.control(arm, SimpleNamespace, sleep=sleep, clock=clock.time)
    except EndTraceError:
        pass
    return arm.writes, control.state


@pytest.mark.parametrize("scenario", ["motion", "lag", "stall"])
def test_motor_commands_and_states_match_original_server(scenario):
    results = [trace(scenario, reference) for reference in (True, False)]
    assert len(results[0][0]) == len(results[1][0])
    for old, new in zip(results[0][0], results[1][0], strict=True):
        np.testing.assert_array_equal(old[0], new[0])
        assert old[1] == new[1]
    assert results[0][1] == results[1][1]
    if scenario == "lag":
        assert results[1][1]["error"].startswith("joint lag")
    if scenario == "stall":
        assert results[1][1]["stalled"]
    if scenario == "motion":
        np.testing.assert_allclose(results[1][0][-1][0], 0, atol=1e-8)


@contextmanager
def serving(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def request(base, path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as response:
        return response.code, response.read(), dict(response.headers)


def test_http_responses_and_image_metadata_match_original():
    results = []
    for reference in (True, False):
        control = armserver.ArmControl()
        control.state.update(q=[0.1] * 6, grip=0.4, goal=[0.1] * 6)
        control.frames["wrist"] = (b"jpeg", [0.2] * 6, 0.6, 123.4567)
        handler = legacy(control, time).Handler if reference else armserver.handler_for(control)
        with serving(handler) as base:
            responses = []
            for path, body in [
                ("/state", None),
                ("/wrist.jpg", None),
                ("/overhead.jpg", None),
                ("/move", {"q": [0.3] * 6, "grip": 2, "rate": 0.4, "stop_on_stall": True}),
                ("/state", None),
                ("/move", {"grip": -1}),
                ("/stop", {}),
                ("/quit", {}),
                ("/unknown", {}),
                ("/unknown", None),
            ]:
                code, data, headers = request(base, path, body)
                responses.append(
                    (
                        code,
                        data,
                        {
                            k: v
                            for k, v in headers.items()
                            if k.startswith("X-") or k in ("Content-Type", "Content-Length")
                        },
                    )
                )
            results.append((responses, control.state))
    assert results[0] == results[1]


def test_camera_hub_shares_capture_and_keeps_each_arms_pose(monkeypatch):
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    hub = armserver.CameraHub()
    left, right = armserver.ArmControl(), armserver.ArmControl()
    left.state.update(q=[1] * 6, grip=0.1)
    right.state.update(q=[2] * 6, grip=0.9)
    hub.subscribe(left, {"wrist": "wrist_l", "overhead": "scene"})
    hub.subscribe(right, {"wrist": "wrist_r", "overhead": "scene"})
    assert len(hub.streams) == 3
    overhead = hub.streams["scene"]
    overhead.publish(True, b"same jpeg")
    assert left.frames["overhead"][:3] == (b"same jpeg", [1] * 6, 0.1)
    assert right.frames["overhead"][:3] == (b"same jpeg", [2] * 6, 0.9)
    assert left.frames["overhead"][3] == right.frames["overhead"][3]
    hub.unsubscribe(left)
    assert "wrist_l" not in hub.streams and not overhead.stop.is_set()
    overhead.publish(True, b"next jpeg")
    assert right.frames["overhead"][0] == b"next jpeg"
    assert left.frames["overhead"][0] == b"same jpeg"
    hub.unsubscribe(right)
    assert overhead.stop.is_set() and not hub.streams


def test_blocked_left_motor_does_not_block_right_http_commands_or_shutdown(tmp_path, monkeypatch):
    # Exercise complete run_arm lifetimes, not just two dictionaries or mocked HTTP dispatch.
    servers, resources, threads, errors = {}, {}, [], []
    left_entered, unblock_left = threading.Event(), threading.Event()
    original_server = armserver.ThreadingHTTPServer

    def make_server(address, handler):
        server = original_server(address, handler)
        servers[threading.current_thread().name] = server
        return server

    monkeypatch.setattr(armserver, "ThreadingHTTPServer", make_server)

    class SlowLeft(FakeArm):
        def write(self, command):
            left_entered.set()
            assert unblock_left.wait(5)
            super().write(command)

    left, right = armserver.ArmControl(), armserver.ArmControl()
    followers = {"left": SlowLeft(), "right": FakeArm()}
    hub = SimpleNamespace(subscribe=lambda *args: None, unsubscribe=lambda *args: None)

    def acquire(binding, share):
        resources[binding.side] = True
        stack = ExitStack()
        stack.callback(resources.__setitem__, binding.side, False)
        return stack, False

    def worker(side, control):
        try:
            armserver.run_arm(
                armserver.ArmBinding(side, None, "wrist", "scene", tmp_path / "arms.lock"),
                control,
                hub,
                0,
                no_overhead=False,
                share=False,
                acquire=acquire,
                open_arm=lambda binding: followers[binding.side],
                command=SimpleNamespace,
                sleep=lambda t: None,
            )
        except Exception as error:
            errors.append(error)

    try:
        for side, control in (("left", left), ("right", right)):
            thread = threading.Thread(target=worker, args=(side, control), name=side)
            threads.append(thread)
            thread.start()
        assert left_entered.wait(2)
        deadline = time.monotonic() + 2
        while "right" not in servers or not followers["right"].writes:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        base = f"http://127.0.0.1:{servers['right'].server_port}"
        assert request(base, "/move", {"q": [0.02] * 6})[0] == 200
        deadline = time.monotonic() + 2
        while right.snapshot()["q"][0] < 0.019:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert request(base, "/stop", {})[0] == 200
        assert request(base, "/quit", {})[0] == 200
        threads[1].join(2)
        assert not threads[1].is_alive()
        assert followers["right"].closed and not resources["right"]
        assert not followers["left"].closed and resources["left"]
        assert not left.state["quit"]
    finally:
        left.post("/quit", {})
        right.post("/quit", {})
        unblock_left.set()
        for thread in threads:
            thread.join(3)
    assert not errors and not any(t.is_alive() for t in threads)


def test_load_bindings_uses_each_saved_calibration_and_wrist(tmp_path, monkeypatch):
    import sys

    left, right = object(), object()
    station = SimpleNamespace(
        body="metal",
        follower_l=left,
        follower_r=right,
        cameras=[
            SimpleNamespace(name=n, device=d)
            for n, d in (("scene", "overhead"), ("wrist_l", "left-wrist"), ("wrist_r", "right-wrist"))
        ],
    )
    monkeypatch.setitem(sys.modules, "robots", SimpleNamespace(config=SimpleNamespace(load=lambda path: station)))
    bindings = armserver.load_bindings(tmp_path / "station.yaml", ("left", "right"))
    assert [b.config for b in bindings] == [left, right]
    assert [b.wrist for b in bindings] == ["left-wrist", "right-wrist"]
    assert [b.overhead for b in bindings] == ["overhead", "overhead"]
    assert all(b.lock_path == pathlib.Path("~/.dreamlabs/arms.lock").expanduser() for b in bindings)


@pytest.mark.skipif(__import__("os").name == "nt", reason="real flock ownership is verified on Linux")
def test_armserver_ownership_coexists_with_opposite_controller_and_rejects_same_side(tmp_path):
    from dream_hackathon import station_lock

    binding = armserver.ArmBinding(
        "left", SimpleNamespace(can_channel="unused"), "wrist", "scene", tmp_path / "arms.lock"
    )
    with station_lock.acquire_sides("SO-101 right", ["right"], path=binding.lock_path):
        ownership, sharing = armserver.acquire_arm(binding, share=False)
        with ownership:
            assert not sharing
            assert {h["holder"] for h in station_lock.holders(binding.lock_path)} == {
                "pah arm server (left arm only)",
                "SO-101 right",
            }
            with pytest.raises(station_lock.StationBusyError):
                armserver.acquire_arm(binding, share=False)
    assert station_lock.holders(binding.lock_path) == []


@pytest.mark.skipif(__import__("os").name == "nt", reason="real flock ownership is verified on Linux")
def test_explicit_legacy_share_keeps_wrist_only_and_never_bypasses_same_side_lock(tmp_path, monkeypatch):
    from dream_hackathon import station_lock

    binding = armserver.ArmBinding(
        "left", SimpleNamespace(can_channel="unused"), "wrist", "scene", tmp_path / "arms.lock"
    )
    monkeypatch.setattr(armserver.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    with station_lock.acquire("legacy right", path=binding.lock_path):
        with pytest.raises(station_lock.StationBusyError):
            armserver.acquire_arm(binding, share=False)
        ownership, sharing = armserver.acquire_arm(binding, share=True)
        with ownership:
            assert sharing
            with pytest.raises(station_lock.StationBusyError):
                armserver.acquire_arm(binding, share=True)
        monkeypatch.setattr(armserver.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="123"))
        with pytest.raises(station_lock.StationBusyError):
            armserver.acquire_arm(binding, share=True)
