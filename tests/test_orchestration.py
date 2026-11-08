"""Graph scheduling, real subprocess/HTTP adapters, and controller routing with fake arms."""

import copy
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from test_armserver import request
from test_dream_caller import fake_arm, listener, wait_for
from test_sequences import Caller

from dream_hackathon.orchestration import FlowManager, Operations, default_graph, evaluate, expand, validate_graph
from dream_hackathon.so101_dashboard import Controls
from dream_hackathon.unified_control import UnifiedController, UnifiedPage


def block(key, kind, config=None):
    return dict(id=key, type=kind, label=key, x=80, y=80, config=config or {})


def flow(*nodes, edges):
    return {
        "nodes": [block("go", "start"), *nodes, block("end", "end")],
        "edges": [
            dict(id=str(index), source=source, port=port, target=target)
            for index, (source, port, target) in enumerate(edges)
        ],
    }


def chain(*nodes):
    keys = ["go", *(node["id"] for node in nodes), "end"]
    return flow(*nodes, edges=[(a, "next", b) for a, b in zip(keys, keys[1:], strict=False)])


class Executor:
    def __init__(self):
        self.calls = []
        self.stops = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def stop(self, run_id):
        self.stops.append(run_id)

    def execute(self, config, context, cancel, run_id, selection):
        self.calls.append(config)
        if config.get("block"):
            self.entered.set()
            assert self.release.wait(3)
        return {
            "name": config.get("name", "test"),
            "value": config.get("value", True),
            "returncode": 0,
            "verdict": "yes",
        }


def manager(path=None, executor=None):
    events = []
    executor = executor or Executor()
    return FlowManager({}, {}, events.append, path, executor), executor, events


def complete(run):
    wait_for(lambda: not run.snapshot()["busy"])
    assert run.snapshot()["state"] == "complete", run.snapshot()


def test_loops_execute_exact_count_and_no_hidden_cutoff():
    manager_, executor, _ = manager()
    plan = flow(
        block("repeat", "loop", {"count": 4}),
        block("work", "action", {"operation": "emit", "name": "worked"}),
        edges=[
            ("go", "next", "repeat"),
            ("repeat", "body", "work"),
            ("work", "next", "repeat"),
            ("repeat", "done", "end"),
        ],
    )
    manager_.update("both", plan)
    manager_.start("both")
    complete(manager_.runs["both"])
    assert len(executor.calls) == 4
    assert manager_.runs["both"].snapshot()["context"]["results"]["work"]["executions"] == 4


def test_parallel_script_end_trigger_and_join():
    manager_, executor, _ = manager()
    plan = flow(
        block("split", "fork"),
        block(
            "script",
            "action",
            {
                "operation": "script",
                "argv": ["unused"],
                "block": True,
            },
        ),
        block("after", "trigger", {"condition": {"kind": "script_end", "name": "script"}}),
        block("join", "join"),
        edges=[
            ("go", "next", "split"),
            ("split", "next", "script"),
            ("split", "next", "after"),
            ("script", "next", "join"),
            ("after", "next", "join"),
            ("join", "next", "end"),
        ],
    )
    manager_.update("both", plan)
    manager_.start("both")
    assert executor.entered.wait(1)
    assert manager_.runs["both"].snapshot()["state"] == "running"
    executor.release.set()
    complete(manager_.runs["both"])
    assert manager_.runs["both"].snapshot()["nodes"]["join"]["status"] == "complete"


def test_two_parallel_actions_do_not_wait_for_each_other_to_start():
    manager_, executor, _ = manager()
    plan = flow(
        block("split", "fork"),
        *[
            block(
                key,
                "action",
                {
                    "operation": "emit",
                    "name": key,
                    "block": True,
                },
            )
            for key in ("a", "b")
        ],
        block("join", "join"),
        edges=[
            ("go", "next", "split"),
            ("split", "next", "a"),
            ("split", "next", "b"),
            ("a", "next", "join"),
            ("b", "next", "join"),
            ("join", "next", "end"),
        ],
    )
    manager_.update("both", plan)
    manager_.start("both")
    wait_for(lambda: len(executor.calls) == 2)
    executor.release.set()
    complete(manager_.runs["both"])


def test_explicit_stop_discards_late_result_and_next_action():
    manager_, executor, _ = manager()
    manager_.update(
        "left",
        chain(
            block("api", "action", {"operation": "http", "url": "unused", "block": True}),
            block("later", "action", {"operation": "emit", "name": "later"}),
        ),
    )
    manager_.start("left")
    assert executor.entered.wait(1)
    manager_.runs["left"].stop()
    assert manager_.runs["left"].snapshot()["busy"]
    executor.release.set()
    wait_for(lambda: not manager_.runs["left"].snapshot()["busy"])
    state = manager_.runs["left"].snapshot()
    assert state["state"] == "stopped" and not state["context"]["results"]
    assert len(executor.calls) == len(executor.stops) == 1


def test_trigger_waits_for_both_sources_not_just_elapsed_time():
    manager_, executor, _ = manager()
    manager_.update(
        "right",
        chain(
            block(
                "gate",
                "trigger",
                {
                    "condition": {
                        "kind": "all",
                        "conditions": [
                            {"kind": "elapsed", "seconds": 0.02},
                            {"kind": "signal", "name": "measured", "value": True},
                            {"kind": "vlm", "name": "visual", "path": "verdict", "value": "yes"},
                        ],
                    }
                },
            ),
            block("ball", "action", {"operation": "emit", "name": "ball"}),
        ),
    )
    manager_.start("right")
    run = manager_.runs["right"]
    run.signal("measured", True)
    run.signal("visual", {"verdict": "unknown"})
    time.sleep(0.06)
    assert not executor.calls and run.snapshot()["state"] == "running"
    run.signal("visual", {"verdict": "yes"})
    complete(run)
    assert len(executor.calls) == 1
    manager_.start("right")
    wait_for(lambda: run.snapshot()["nodes"].get("gate", {}).get("status") == "active")
    assert not run.snapshot()["context"]["signals"]
    run.stop()


def test_configured_timeout_routes_only_to_explicit_timeout_output():
    manager_, executor, _ = manager()
    plan = flow(
        block("wait", "trigger", {"condition": {"kind": "signal", "name": "never"}, "timeoutSeconds": 0.02}),
        block("expired", "action", {"operation": "emit", "name": "expired"}),
        edges=[
            ("go", "next", "wait"),
            ("wait", "next", "end"),
            ("wait", "timeout", "expired"),
            ("expired", "next", "end"),
        ],
    )
    manager_.update("left", plan)
    manager_.start("left")
    complete(manager_.runs["left"])
    assert executor.calls[0]["name"] == "expired"


def test_conditions_use_activation_time_vs_global_go_and_nested_json():
    context = {"results": {"api": {"body": {"count": 2}}}, "signals": {}, "variables": {}}
    assert evaluate({"kind": "elapsed", "seconds": 10}, context, 12, 1)
    assert not evaluate({"kind": "delay", "seconds": 10}, context, 12, 1)
    assert evaluate({"kind": "result", "name": "api", "path": "body.count", "op": "gte", "value": 2}, context, 0, 0)
    assert not evaluate({"kind": "result", "name": "missing", "op": "ne", "value": 3}, context, 0, 0)


def test_templates_keep_types_and_do_not_expand_secrets_into_saved_graph(monkeypatch):
    monkeypatch.setenv("ORCHESTRATION_TEST_KEY", "test-only")
    context = {"signals": {"count": 3}, "variables": {}, "results": {}}
    assert expand({"value": "{{signals.count}}", "auth": "Bearer {{env.ORCHESTRATION_TEST_KEY}}"}, context) == {
        "value": 3,
        "auth": "Bearer test-only",
    }
    with pytest.raises(ValueError, match="No value"):
        expand("{{signals.unknown}}", context)


def test_save_persist_reload_and_running_snapshot(tmp_path):
    manager_, _, events = manager(tmp_path / "flows.json")
    original = chain(block("wait", "trigger", {"condition": {"kind": "signal", "name": "finish"}}))
    manager_.update("left", original)
    manager_.start("left")
    edited = chain(block("wait", "trigger", {"condition": {"kind": "signal", "name": "different"}}))
    manager_.update("left", edited)
    manager_.runs["left"].signal("finish", True)
    complete(manager_.runs["left"])
    restored, _, _ = manager(tmp_path / "flows.json")
    assert restored.snapshot()["left"]["definition"] == edited
    assert events[-1]["type"] == "flows"
    with pytest.raises(ValueError, match="another browser"):
        manager_.update("left", original, revision=0)


def test_validation_and_explicit_loop_connections():
    assert validate_graph(default_graph(), "both", for_run=True)
    invalid = default_graph()
    invalid["edges"][0]["target"] = "missing"
    with pytest.raises(ValueError, match="missing"):
        validate_graph(invalid, "both")
    with pytest.raises(ValueError, match="Connect"):
        validate_graph({"nodes": default_graph()["nodes"], "edges": []}, "both", for_run=True)
    invalid = chain(
        block("api", "action", {"operation": "dream", "selection": "right", "task": "test", "durationSeconds": 1})
    )
    with pytest.raises(ValueError, match="belong"):
        validate_graph(invalid, "left")


def test_real_script_output_exit_status_and_completion_record():
    manager_, _, _ = manager(executor=Operations({}, {}))
    manager_.update(
        "both",
        chain(
            block(
                "script",
                "action",
                {
                    "operation": "script",
                    "argv": [
                        sys.executable,
                        "-c",
                        "print('script completed')",
                    ],
                },
            )
        ),
    )
    manager_.start("both")
    complete(manager_.runs["both"])
    result = manager_.runs["both"].snapshot()["context"]["results"]["script"]
    assert result["returncode"] == 0 and result["stdout"].strip() == "script completed"
    assert evaluate({"kind": "script_end", "name": "script"}, manager_.runs["both"].snapshot()["context"], 0, 0)


def test_http_operation_calls_real_local_endpoint_and_exposes_body():
    logs = []
    with listener(fake_arm(logs)) as base:
        result = Operations({}, {}).execute(
            {"operation": "http", "url": base + "/state", "method": "GET"}, {}, threading.Event(), "test-run", "left"
        )
    assert result["status"] == 200 and len(result["body"]["q"]) == 6


def test_controller_routes_main_full_auto_and_serves_editor(tmp_path):
    nodes = {
        side: SimpleNamespace(snapshot=lambda: {"arm_open": False}, cancel_home=lambda: None)
        for side in ("left", "right")
    }
    leader = SimpleNamespace(side=None, state=lambda: Controls(1, "none").snapshot())
    callers = {side: Caller(side) for side in ("left", "right", "both")}
    controller = UnifiedController(nodes, leader, callers, flow_path=tmp_path / "flow.json")
    page = UnifiedPage(controller, 0)
    base = f"http://127.0.0.1:{page.port}"
    try:
        assert request(base, "/api/flows/both/go", {})[0] == 400
        assert request(base, "/api/both/input", {"source": "auto"})[0] == 200
        assert request(base, "/api/flows/both/save", {"definition": default_graph()})[0] == 200
        assert request(base, "/api/flows/both/go", {})[0] == 200
        complete(controller.flows.runs["both"])
        assert request(base, "/orchestration.js")[0] == 200
        assert request(base, "/orchestration.css")[0] == 200
        plan = chain(block("gate", "trigger", {"condition": {"kind": "signal", "name": "continue"}}))
        assert request(base, "/api/flows/both/save", {"definition": plan})[0] == 200
        assert request(base, "/api/flows/both/go", {})[0] == 200
        assert request(base, "/api/flows/both/signal", {"name": "continue", "value": True})[0] == 200
        complete(controller.flows.runs["both"])
        controller.flow_action("both", "go", {})
        controller.select_input("left", "dream")
        wait_for(lambda: not controller.flows.runs["both"].snapshot()["busy"])
        assert controller.flows.runs["both"].snapshot()["state"] == "stopped"
    finally:
        page.close()


def test_while_loop_follows_condition_and_false_branch():
    manager_, executor, _ = manager()
    plan = flow(
        block(
            "repeat",
            "loop",
            {
                "mode": "while",
                "condition": {
                    "kind": "variable",
                    "name": "done",
                    "op": "ne",
                    "value": True,
                },
            },
        ),
        block("work", "action", {"operation": "emit", "name": "work"}),
        edges=[
            ("go", "next", "repeat"),
            ("repeat", "body", "work"),
            ("work", "next", "repeat"),
            ("repeat", "done", "end"),
        ],
    )
    # Missing values do not satisfy a comparison, so this loop takes its done output.
    manager_.update("left", copy.deepcopy(plan))
    manager_.start("left")
    complete(manager_.runs["left"])
    assert not executor.calls


def test_while_loop_body_changes_condition_and_exits():
    manager_, _, _ = manager(executor=Operations({}, {}))
    plan = flow(
        block("init", "action", {"operation": "set", "name": "done", "value": False}),
        block(
            "repeat",
            "loop",
            {
                "mode": "while",
                "condition": {
                    "kind": "variable",
                    "name": "done",
                    "value": False,
                },
            },
        ),
        block("work", "action", {"operation": "set", "name": "done", "value": True}),
        edges=[
            ("go", "next", "init"),
            ("init", "next", "repeat"),
            ("repeat", "body", "work"),
            ("work", "next", "repeat"),
            ("repeat", "done", "end"),
        ],
    )
    manager_.update("left", plan)
    manager_.start("left")
    complete(manager_.runs["left"])
    assert manager_.runs["left"].snapshot()["context"]["results"]["work"]["executions"] == 1


def test_branch_false_selects_only_false_output():
    manager_, executor, _ = manager()
    plan = flow(
        block("choose", "condition", {"condition": {"kind": "signal", "name": "pass"}}),
        block("true", "action", {"operation": "emit", "name": "true"}),
        block("false", "action", {"operation": "emit", "name": "false"}),
        edges=[
            ("go", "next", "choose"),
            ("choose", "true", "true"),
            ("choose", "false", "false"),
            ("true", "next", "end"),
            ("false", "next", "end"),
        ],
    )
    manager_.update("left", plan)
    manager_.start("left")
    complete(manager_.runs["left"])
    assert [call["name"] for call in executor.calls] == ["false"]


def test_dream_output_uses_selected_arm_worker_and_configured_window():
    callers = {side: Caller(side) for side in ("left", "right", "both")}
    nodes = {side: SimpleNamespace(snapshot=lambda: {"arm_open": False}) for side in ("left", "right")}
    result = Operations(callers, nodes).execute(
        {"operation": "dream", "selection": "right", "task": "unplug", "durationSeconds": 0.02},
        {},
        threading.Event(),
        "test",
        "both",
    )
    assert result["window_complete"]
    assert callers["right"].calls[0][0]["task"] == "unplug"
    assert not callers["left"].calls and not callers["both"].calls
    assert not callers["right"].busy


def test_vlm_operation_returns_structured_verdict_without_hardware(monkeypatch, tmp_path):
    from dream_hackathon.demo_vlm import Verdict

    image = tmp_path / "scene.jpg"
    image.write_bytes(b"fixture")
    calls = []

    def verify(_client, encoded, question):
        calls.append((encoded, question))
        return Verdict("yes", "USB visibly separated"), {}

    monkeypatch.setattr("dream_hackathon.demo_vlm.LocalVLM.verify", verify)
    result = Operations({}, {}).execute(
        {
            "operation": "vlm",
            "provider": "ollama",
            "model": "test-model",
            "imagePath": str(image),
            "question": "Is it removed?",
        },
        {},
        threading.Event(),
        "test",
        "right",
    )
    assert result["verdict"] == "yes"
    assert calls == [(b"fixture", "Is it removed?")]


def test_script_failure_preserves_exit_record_and_marks_node_error():
    manager_, _, _ = manager(executor=Operations({}, {}))
    manager_.update(
        "left",
        chain(
            block(
                "fail",
                "action",
                {
                    "operation": "script",
                    "argv": [
                        sys.executable,
                        "-c",
                        "raise SystemExit(7)",
                    ],
                },
            )
        ),
    )
    manager_.start("left")
    run = manager_.runs["left"]
    wait_for(lambda: not run.snapshot()["busy"])
    state = run.snapshot()
    assert state["state"] == "error" and state["nodes"]["fail"]["status"] == "error"
    assert state["context"]["results"]["fail"]["returncode"] == 7
    assert evaluate({"kind": "script_end", "name": "fail"}, state["context"], 0, 0)


def test_script_runtime_limit_routes_to_configured_timeout():
    manager_, _, _ = manager(executor=Operations({}, {}))
    plan = flow(
        block(
            "slow",
            "action",
            {
                "operation": "script",
                "timeoutSeconds": 0.02,
                "argv": [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(30)",
                ],
            },
        ),
        edges=[("go", "next", "slow"), ("slow", "next", "end"), ("slow", "timeout", "end")],
    )
    manager_.update("left", plan)
    manager_.start("left")
    complete(manager_.runs["left"])
    assert manager_.runs["left"].snapshot()["nodes"]["slow"]["status"] == "timeout"


def test_completed_elapsed_time_freezes_and_stop_before_execute_starts_nothing():
    manager_, _, _ = manager()
    manager_.start("left")
    run = manager_.runs["left"]
    complete(run)
    elapsed = run.snapshot()["elapsedSeconds"]
    time.sleep(0.02)
    assert run.snapshot()["elapsedSeconds"] == elapsed
    cancel = threading.Event()
    cancel.set()
    assert Operations({}, {}).execute(
        {"operation": "script", "argv": ["does-not-exist"]}, {}, cancel, "cancelled-run", "left"
    ) == {"cancelled": True}


def test_independent_flows_publish_concurrently_without_lock_inversion():
    manager_, _, _ = manager(executor=Operations({}, {}))
    for side in ("left", "right"):
        manager_.update(side, chain(block("event", "action", {"operation": "emit", "name": side})))
        manager_.start(side)
    for run in (manager_.runs["left"], manager_.runs["right"]):
        complete(run)
