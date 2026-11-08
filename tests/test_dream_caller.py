"""Exercise routing and cancellation with HTTP listeners and a blocked model, no hardware."""

import base64
import json
import threading
import time
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from dream_hackathon.dream_caller import ArmCaller, ArmEndpoint, StationCaller, build_observation, handler_for, jpeg


@pytest.mark.parametrize("side", ["left", "right"])
def test_new_armserver_contract(side):
    from dream_hackathon.armserver import ArmControl
    from dream_hackathon.armserver import handler_for as arm_handler

    controls = {name: ArmControl() for name in ("left", "right")}
    frame = jpeg(np.zeros((18, 32, 3), np.uint8))
    for control in controls.values():
        control.state.update(q=[0.1] * 6, grip=0.6, goal=[0.1] * 6)
        for name in ("wrist", "overhead"):
            control.camera_frame(name, frame, time.time())
    with listener(arm_handler(controls["left"])) as left, listener(arm_handler(controls["right"])) as right:
        endpoint = ArmEndpoint({"left": left, "right": right}[side])
        assert endpoint.state()["grip"] == 0.6
        assert endpoint.frame("overhead") == endpoint.frame("wrist") == frame
        other = "right" if side == "left" else "left"
        unchanged = controls[other].snapshot()
        endpoint.move([0.9] * 6 + [0.8], 0.4)
        assert controls[side].snapshot()["goal"] == [0.9] * 6
        assert controls[side].snapshot()["goal_grip"] == 0.8
        endpoint.stop()
        assert controls[side].snapshot()["goal"] == [0.1] * 6
        assert controls[other].snapshot() == unchanged


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out")
        time.sleep(0.005)


@contextmanager
def listener(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def fake_arm(log):
    frame = jpeg(np.full((18, 32, 3), 127, dtype=np.uint8))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            log.append((self.path, None))
            body = json.dumps({"q": [0.1] * 6, "grip": 0.7}).encode() if self.path == "/state" else frame
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            log.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    return Handler


class Model:
    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.ticks = []

    def set_observation(self, frames, observation, *, timestamp):
        self.ticks.append((frames, observation, timestamp))

    def get_action(self):
        self.entered.set()
        assert self.release.wait(3)
        return {"q_l_chunk": [[1.0] * 6 + [0.2]] * 3, "q_r_chunk": [[2.0] * 6 + [0.8]] * 3}

    def close(self):
        pass


@pytest.mark.parametrize("side", ["left", "right"])
def test_dummy_data_and_frames(side):
    blank = jpeg(np.zeros((18, 32, 3), np.uint8))
    live = jpeg(np.full((18, 32, 3), 127, np.uint8))
    frames, state = build_observation(side, {"q": [0.2] * 6, "grip": 0.7}, live, live, blank, "task")
    other = "right" if side == "left" else "left"
    assert state[f"state.{side}_arm"] == [0.2] * 6
    assert state[f"state.{side}_gripper"] == [70]
    assert state[f"state.{other}_arm"] == [0] * 6
    assert state[f"state.{other}_grip"] == state[f"state.{other}_gripper"] == [0]
    key = "video.wrist_r" if side == "left" else "video.wrist_l"
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(frames[key]), np.uint8), cv2.IMREAD_COLOR)
    assert not decoded.any()
    assert frames["video.scene"] == base64.b64encode(live).decode()


@pytest.mark.parametrize("side", ["left", "right"])
def test_http_isolation_streaming_and_stop_during_inference(side):
    logs = {"left": [], "right": []}
    model = Model()
    with listener(fake_arm(logs["left"])) as left, listener(fake_arm(logs["right"])) as right:
        caller = ArmCaller(
            side, ArmEndpoint({"left": left, "right": right}[side]), client_factory=lambda *_: model, observation_hz=50
        )
        try:
            caller.connect()
            wait_for(lambda: caller.sample is not None)
            caller.start({"task": "move"}, run=True)
            assert model.entered.wait(2)
            ticks = len(model.ticks)
            wait_for(lambda: len(model.ticks) > ticks + 2)
            caller.stop()
            model.release.set()
            wait_for(lambda: not caller.snapshot()["busy"])
            assert [p for p, _ in logs[side] if p in ("/move", "/stop", "/quit")] == ["/stop"]
            assert not logs["right" if side == "left" else "left"]
        finally:
            model.release.set()
            caller.close()


@pytest.mark.parametrize("side", ["left", "right"])
def test_only_selected_chunk_is_executed(side):
    log, model = [], Model()
    model.release.set()
    with listener(fake_arm(log)) as url:
        caller = ArmCaller(side, ArmEndpoint(url), client_factory=lambda *_: model, hz=100)
        try:
            caller.connect()
            wait_for(lambda: caller.sample is not None)
            caller.start({"task": "inspect"}, run=False)
            wait_for(lambda: not caller.snapshot()["busy"])
            assert not any(path == "/move" for path, _ in log)
            expected = [1.0] * 6 if side == "left" else [2.0] * 6
            assert caller.snapshot()["rows"][0][:6] == expected
            caller.start({"task": "run"}, run=True)
            wait_for(lambda: any(path == "/move" for path, _ in log))
            caller.stop()
            wait_for(lambda: not caller.snapshot()["busy"])
            moves = [body for path, body in log if path == "/move"]
            assert all(body["q"] == expected for body in moves)
            assert all(body["grip"] == (0.2 if side == "left" else 0.8) for body in moves)
            paths = [path for path, _ in log]
            assert "/move" not in paths[paths.index("/stop") + 1 :]
        finally:
            caller.close()


def test_callers_do_not_block_each_other_and_page_has_no_connect_side_effect():
    logs, models = {"left": [], "right": []}, {"left": Model(), "right": Model()}
    models["right"].release.set()
    with listener(fake_arm(logs["left"])) as left, listener(fake_arm(logs["right"])) as right:
        callers = {
            side: ArmCaller(side, ArmEndpoint(url), client_factory=lambda *_, s=side: models[s])
            for side, url in (("left", left), ("right", right))
        }
        try:
            with listener(handler_for(callers)) as page:
                with urllib.request.urlopen(page) as response:
                    assert b"Dream-1 API caller" in response.read()
                with urllib.request.urlopen(page + "/api/state") as response:
                    assert not json.load(response)["left"]["connected"]
                assert not logs["left"] and not logs["right"]
                for caller in callers.values():
                    caller.connect()
                    wait_for(lambda c=caller: c.sample is not None)
                    caller.start({"task": "inspect"}, run=False)
                assert models["left"].entered.wait(2)
                wait_for(lambda: callers["right"].snapshot()["calls"] == 1)
                assert callers["left"].snapshot()["busy"]
        finally:
            models["left"].release.set()
            for caller in callers.values():
                caller.close()


@pytest.mark.parametrize("selection", ["left", "right", "both"])
def test_main_controller_observations_and_direct_playback(selection):
    logs, model = {"left": [], "right": []}, Model()
    model.release.set()
    with listener(fake_arm(logs["left"])) as left, listener(fake_arm(logs["right"])) as right:
        endpoints = {"left": ArmEndpoint(left), "right": ArmEndpoint(right)}
        caller = StationCaller(selection, endpoints, client_factory=lambda *_: model, hz=100)
        try:
            assert not any(logs.values())
            caller.connect()
            wait_for(lambda: caller.sample is not None)
            caller.start({"task": "inspect"}, run=False)
            wait_for(lambda: not caller.snapshot()["busy"])
            frames, observation, _ = model.ticks[-1]
            assert set(frames) == {"video.scene", "video.wrist_l", "video.wrist_r"}
            selected = ("left", "right") if selection == "both" else (selection,)
            for side, slot in (("left", "l"), ("right", "r")):
                image = cv2.imdecode(np.frombuffer(base64.b64decode(frames[f"video.wrist_{slot}"]), np.uint8), 1)
                if side in selected:
                    assert observation[f"state.{side}_arm"] == [0.1] * 6
                    assert observation[f"state.{side}_grip"] == [0.7]
                    assert image.mean() == pytest.approx(127, abs=1)
                else:
                    assert observation[f"state.{side}_arm"] == [0] * 6
                    assert observation[f"state.{side}_grip"] == [0]
                    assert not image.any()
                    assert logs[side] == []
            assert not any(path == "/target" for log in logs.values() for path, _ in log)
            caller.start({"task": "run"}, run=True)
            wait_for(lambda: all(any(path == "/target" for path, _ in logs[s]) for s in selected))
            caller.stop()
            wait_for(lambda: not caller.snapshot()["busy"])
            for side in selected:
                targets = [body for path, body in logs[side] if path == "/target"]
                assert all(body["q"] == [1.0 if side == "left" else 2.0] * 6 for body in targets)
                paths = [path for path, _ in logs[side]]
                assert "/target" not in paths[paths.index("/stop") + 1 :]
            if selection == "both":
                assert len(caller.snapshot()["rows"][0]) == 14
        finally:
            caller.close()


def test_both_stop_discards_pending_api_reply():
    logs, model = {"left": [], "right": []}, Model()
    with listener(fake_arm(logs["left"])) as left, listener(fake_arm(logs["right"])) as right:
        caller = StationCaller(
            "both", {"left": ArmEndpoint(left), "right": ArmEndpoint(right)}, client_factory=lambda *_: model
        )
        try:
            caller.connect()
            wait_for(lambda: caller.sample is not None)
            caller.start({"task": "run"}, run=True)
            assert model.entered.wait(2)
            caller.stop()
            model.release.set()
            wait_for(lambda: not caller.snapshot()["busy"])
            for log in logs.values():
                assert [p for p, _ in log if p in ("/target", "/stop")] == ["/stop"]
        finally:
            model.release.set()
            caller.close()
