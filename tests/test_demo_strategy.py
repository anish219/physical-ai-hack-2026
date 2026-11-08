"""Exercise the actual competition map with explicit simulated scene results."""

import json
import threading
from types import SimpleNamespace

from test_dream_caller import wait_for

from dream_hackathon.demo_strategy import competition_graph
from dream_hackathon.orchestration import FlowManager, Operations, evaluate, validate_graph


def test_strategy_has_no_balls_and_dummy_observers_do_not_claim_success():
    graph = validate_graph(competition_graph(), "both", for_run=True)
    assert not any("ball" in node["id"] for node in graph["nodes"])
    assert graph["variables"]["runSeconds"] == 300
    assert graph["variables"]["maxBlocks"] == 7
    assert graph["bindings"]["verify_usb"]["result"]["known"] is False
    assert graph["bindings"]["verify_tower"]["result"]["known"] is False
    assert all(binding["operation"] == "dummy" for binding in graph["bindings"].values())


def test_cap_and_late_rules_use_editable_values_and_strictly_above_five():
    graph = competition_graph()
    condition = next(node for node in graph["nodes"] if node["id"] == "late_hold")["config"]["condition"]
    context = {"variables": graph["variables"], "results": {"tower_view": {"block_count": 6}}}
    assert not evaluate(condition, context, 239.99, 0)
    assert evaluate(condition, context, 240, 0)
    context["results"]["tower_view"]["block_count"] = 5
    assert not evaluate(condition, context, 240, 0)


class StrategySimulation:
    def __init__(self):
        self.calls = []
        self.counts = {}
        self.stops = []

    def stop(self, run_id):
        self.stops.append(run_id)

    def execute(self, config, context, cancel, run_id, selection):
        if config["operation"] == "finish":
            return {"finish": True}
        key = config["binding"]
        self.counts[key] = self.counts.get(key, 0) + 1
        self.calls.append((key, config["selection"], context))
        if key == "verify_usb":
            if self.counts.get("pull_all", 0) < 2:
                return {"known": True, "unplugged_count": 0, "plugged_count": 4}
            if self.counts.get("unplug_policy", 0) == 0:
                return {"known": True, "unplugged_count": 1, "plugged_count": 3}
            return {
                "known": True,
                "unplugged_count": 4,
                "plugged_count": 0,
                "fully_inserted": self.counts.get("plug_usb", 0) >= 2,
                "gripper_released": True,
                "cable_on_table": self.counts.get("unplug_policy", 0) >= 3,
            }
        if key == "verify_tower":
            n = self.counts[key]
            count = [6, 0, 1, 5, 6][min(n - 1, 4)]
            return {"known": True, "block_count": count, "collapsed": n == 2, "standing_free": True}
        return {}


def test_full_map_retries_unplugs_starts_stack_recovers_and_cycles_until_clock():
    graph = competition_graph()
    graph["variables"].update(runSeconds=1.6, lateAtSeconds=0, pollSeconds=0.005)
    sim = StrategySimulation()
    manager = FlowManager({}, {}, lambda _event: None, executor=sim)
    manager.update("both", graph)
    manager.start("both")
    run = manager.runs["both"]
    wait_for(lambda: not run.snapshot()["busy"])
    assert run.snapshot()["state"] == "complete"
    assert sim.counts["pull_all"] == 2
    assert sim.counts["rebuild"] == 1
    assert sim.counts["stack_next"] >= 1
    assert sim.counts["plug_usb"] >= 3
    assert sim.counts["return_usb_start"] >= 1
    assert sim.counts["unplug_policy"] >= 3
    assert sim.stops
    for key, side, context in sim.calls:
        if key in ("stack_next", "rebuild"):
            assert side == "left" and context["results"]["first_usb_view"]["unplugged_count"] >= 1
        if key == "stack_next":
            assert context["results"]["tower_view"]["block_count"] <= 5


def test_binding_swap_keeps_node_ids_and_wiring():
    graph = competition_graph()
    before = graph["edges"][:]
    binding = graph["bindings"]["pull_all"]
    graph["bindings"]["pull_all"] = binding["replaceWith"]
    validate_graph(graph, "both", for_run=True)
    assert graph["edges"] == before
    assert next(n for n in graph["nodes"] if n["id"] == "pull_all")["config"]["binding"] == "pull_all"


def test_verifier_binding_can_unwrap_a_provider_without_changing_conditions():
    bindings = {
        "inspect": {
            "operation": "dummy",
            "delaySeconds": 0,
            "result": {"body": {"known": True, "unplugged_count": 1}},
            "resultPath": "body",
        }
    }
    result = Operations({}, {}).execute(
        {"operation": "binding", "binding": "inspect", "selection": "right"},
        {"bindings": bindings},
        threading.Event(),
        "test",
        "both",
    )
    assert result == {"known": True, "unplugged_count": 1}


def test_recorded_episode_targets_selected_arm_and_ignores_other_plan_endpoint(tmp_path):
    (tmp_path / "targets.jsonl").write_text(json.dumps({"t": 0, "q": [0] * 6, "grip": 1}) + "\n")
    plan = tmp_path / "episode.json"
    plan.write_text(json.dumps({"targets_file": "targets.jsonl", "target_url": "http://unused-right/target"}))
    calls = []
    nodes = {"left": SimpleNamespace(control=SimpleNamespace(post=lambda path, body: calls.append((path, body))))}
    result = Operations({}, nodes).execute(
        {"operation": "episode", "selection": "left", "planPath": str(plan)}, {}, threading.Event(), "test-run", "both"
    )
    assert result["selection"] == "left" and calls[0][0] == "/target"
