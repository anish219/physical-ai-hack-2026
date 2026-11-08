"""Shared sequence definitions and timed execution; fake arm/API workers only."""

import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        status, body, _ = request(base, "/api/state")
        assert status == 200
        payload = json.loads(body)
        assert payload["monitor_scripts"] == []
        assert payload["sequences"]["left"]["definition"]["phases"][0]["steps"][0]["monitor"]["type"] == "none"
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
        controller.sequences.update("right", {"phases": []})
        assert request(base, "/api/match/start", {})[0] == 200
        wait_for(lambda: controller.sequences.runs["left"].snapshot()["busy"])
        started = json.loads(request(base, "/api/state")[1])
        assert started["match_started_at"] and started["match_stopped_at"] is None
        assert request(base, "/api/match/stop", {})[0] == 200
        wait_for(lambda: not controller.sequences.runs["left"].snapshot()["busy"])
        stopped = json.loads(request(base, "/api/state")[1])
        assert stopped["match_stopped_at"] >= started["match_started_at"]
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


def test_saved_steps_gain_a_default_monitor_and_prompt_text_is_required_to_run():
    plan = definition()
    saved = validate_definition(plan)
    assert saved["phases"][0]["steps"][0]["monitor"] == {"type": "none", "prompt": "", "script": ""}
    saved["phases"][0]["steps"][0]["monitor"] = {"type": "prompt", "prompt": "   ", "script": ""}
    with pytest.raises(ValueError, match="VLM monitor prompt"):
        validate_definition(saved, for_run=True)
    saved["phases"][0]["steps"][0]["monitor"]["prompt"] = "Did the USB leave the port?"
    assert validate_definition(saved, for_run=True)["phases"][0]["steps"][0]["monitor"]["type"] == "prompt"


def test_script_pointer_must_be_registered_to_run_but_can_be_saved_early():
    plan = definition()
    plan["phases"][0]["steps"][0]["monitor"] = {"type": "script", "prompt": "", "script": "usb_unplugged"}
    saved = validate_definition(plan)
    assert saved["phases"][0]["steps"][0]["monitor"]["script"] == "usb_unplugged"
    with pytest.raises(ValueError, match="Unknown monitor script pointer"):
        validate_definition(saved, for_run=True, scripts=())
    assert validate_definition(saved, for_run=True, scripts=("usb_unplugged",))["phases"][0]["steps"][0]["monitor"]["script"] == "usb_unplugged"


def test_registered_monitors_run_after_each_window_without_gating(tmp_path):
    sequences, callers, _, _ = manager(tmp_path / "sequences.json")
    seen = []
    sequences.monitors.set_prompt_evaluator(
        lambda prompt, context: seen.append(("prompt", prompt, context["selection"])) or "yes"
    )
    sequences.monitors.register_script(
        "usb_clear",
        lambda context: seen.append(("script", context["step"]["id"])) or {"result": "no", "detail": "still seated"},
        label="USB clear",
    )
    plan = definition(wait=0)
    plan["phases"][0]["steps"][0]["monitor"] = {"type": "prompt", "prompt": "Was the wait useful?", "script": ""}
    plan["phases"][0]["steps"][1]["monitor"] = {"type": "script", "prompt": "", "script": "usb_clear"}
    sequences.update("left", plan)
    try:
        sequences.start("left")
        wait_for(lambda: not sequences.runs["left"].snapshot()["busy"])
        assert sequences.runs["left"].snapshot()["state"] == "complete"
        assert sequences.runs["left"].snapshot()["monitor_result"] == {"result": "no", "detail": "still seated"}
        assert seen == [("prompt", "Was the wait useful?", "left"), ("script", "dream1")]
        assert callers["left"].calls == [({"task": "do the task", "checkpoint": "test/checkpoint"}, True)]
        assert sequences.monitors.available_scripts() == [{"pointer": "usb_clear", "label": "USB clear"}]
    finally:
        sequences.close()


def test_unwired_monitors_record_unwired_and_continue():
    sequences, callers, _, _ = manager()
    plan = {
        "phases": [
            {
                "id": "phase1",
                "title": "First phase",
                "steps": [
                    {
                        "id": "wait1",
                        "label": "Wait",
                        "type": "wait",
                        "durationSeconds": 0,
                        "monitor": {"type": "prompt", "prompt": "look", "script": ""},
                    }
                ],
            }
        ]
    }
    sequences.update("right", plan)
    try:
        sequences.start("right")
        wait_for(lambda: not sequences.runs["right"].snapshot()["busy"])
        assert sequences.runs["right"].snapshot()["state"] == "complete"
        assert sequences.runs["right"].snapshot()["monitor_result"]["result"] == "unwired"
        assert not any(caller.calls for caller in callers.values())
    finally:
        sequences.close()


def test_http_mode_posts_start_and_stop_without_dream_calls():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            received.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sequences, callers, _, _ = manager()
    url = f"http://127.0.0.1:{server.server_port}/start"
    sequences.update("left", {"mode": "http", "httpUrl": url, "phases": []})
    sequences.update("right", {"phases": []})
    try:
        started = sequences.start_match()
        assert started == ["left"]
        wait_for(lambda: received and received[0]["event"] == "start")
        assert sequences.runs["left"].snapshot()["busy"]
        sequences.stop_sides(["left"])
        wait_for(lambda: not sequences.runs["left"].snapshot()["busy"])
        assert [item["event"] for item in received] == ["start", "stop"]
        assert received[0]["selection"] == "left"
        assert not any(caller.calls for caller in callers.values())
    finally:
        sequences.close()
        server.shutdown()


def test_script_mode_calls_start_and_stop(tmp_path, monkeypatch):
    from dream_hackathon import sequences as sequences_mod

    monkeypatch.setattr(sequences_mod, "PROJECT_ROOT", tmp_path)
    log = tmp_path / "log.txt"
    (tmp_path / "left_go.py").write_text(
        "from pathlib import Path\n"
        f"LOG = Path({str(log)!r})\n"
        "def start(context):\n"
        "    LOG.write_text('start:' + context['selection'])\n"
        "def stop(context):\n"
        "    LOG.write_text(LOG.read_text() + '|stop:' + context['event'])\n"
    )
    sequences, callers, _, _ = manager()
    sequences.update("left", {"mode": "script", "scriptRef": "left_go.py:start", "phases": []})
    sequences.update("right", {"phases": []})
    try:
        started = sequences.start_match()
        assert started == ["left"]
        wait_for(lambda: log.exists() and log.read_text().startswith("start:left"))
        assert sequences.runs["left"].snapshot()["busy"]
        sequences.stop_sides(["left"])
        wait_for(lambda: not sequences.runs["left"].snapshot()["busy"])
        assert log.read_text() == "start:left|stop:stop"
        assert not any(caller.calls for caller in callers.values())
    finally:
        sequences.close()


def test_http_mode_saves_empty_url_but_refuses_run_without_one():
    saved = validate_definition({"mode": "http", "httpUrl": "", "phases": []})
    assert saved["mode"] == "http" and saved["httpUrl"] == ""
    with pytest.raises(ValueError, match="http or https"):
        validate_definition({"mode": "http", "httpUrl": "", "phases": []}, for_run=True)
    with pytest.raises(ValueError, match="http or https"):
        validate_definition({"mode": "http", "httpUrl": "not-a-url", "phases": []})
    script = validate_definition({"mode": "script", "scriptRef": "", "phases": []})
    assert script["mode"] == "script" and script["scriptRef"] == ""
    with pytest.raises(ValueError, match="project file"):
        validate_definition({"mode": "script", "scriptRef": "", "phases": []}, for_run=True)
    with pytest.raises(ValueError, match="this project"):
        validate_definition({"mode": "script", "scriptRef": "../secret.py", "phases": []})
    sequences, _, _, _ = manager()
    try:
        sequences.update("left", {"phases": []})
        sequences.update("right", {"phases": []})
        with pytest.raises(ValueError, match="HTTP start link"):
            sequences.start_match()
    finally:
        sequences.close()


def test_http_input_sets_sequence_mode_and_match_stop_does_not_command_arms():
    posts = []

    def make_node():
        return SimpleNamespace(
            snapshot=lambda: {"arm_open": True, "grip_torque_nm": 1},
            cancel_home=lambda: None,
            control=SimpleNamespace(post=lambda path, body: posts.append(path)),
        )

    nodes = {"left": make_node(), "right": make_node()}
    callers = {side: Caller(side) for side in ("left", "right", "both")}
    leader = SimpleNamespace(side=None, state=lambda: Controls(1, "none").snapshot())
    controller = UnifiedController(nodes, leader, callers)
    try:
        controller.select_input("left", "http")
        assert controller.inputs["left"] == "http"
        assert controller.sequences.snapshot()["left"]["definition"]["mode"] == "http"
        controller.select_input("left", "sequence")
        assert controller.sequences.snapshot()["left"]["definition"]["mode"] == "local"
        controller.sequences.update("left", definition(wait=1))
        controller.sequences.update("right", {"phases": []})
        controller.match_start()
        wait_for(lambda: controller.sequences.runs["left"].snapshot()["busy"])
        posts.clear()
        controller.match_stop()
        wait_for(lambda: not controller.sequences.runs["left"].snapshot()["busy"])
        assert posts == []
        assert controller.match_stopped_at is not None
    finally:
        controller.sequences.close()
