"""The team's editable five-minute strategy. Bindings deliberately start as dummies."""

import copy


def competition_graph():
    graph = {
        "version": 2,
        "name": "Tower height + repeating USB cycle",
        "variables": {"maxBlocks": 7, "lateAtSeconds": 240, "lateHoldAbove": 5, "runSeconds": 300, "pollSeconds": 0.5},
        "notes": [
            "Team reports judges approved repeated insert/release/unplug/table/release scoring.",
            "Written rubric v4 forbids replugging unplugged cables; team ruling takes precedence in this map.",
            "Bindings are explicit dummies until recordings, policies, and verifiers are supplied.",
            "No balls. Stacking starts only after a known first unplug. Collapse always returns to rebuilding.",
        ],
        "bindings": {},
        "groups": [
            {"label": "First cable: recorded attempt + verification", "x": 30, "y": 15, "width": 1810, "height": 230},
            {"label": "Left arm: build height / restack on collapse", "x": 30, "y": 360, "width": 1190, "height": 760},
            {
                "label": "Right arm: clear hub / repeat scoring cycle",
                "x": 1230,
                "y": 360,
                "width": 1500,
                "height": 1210,
            },
            {"label": "Five-minute match clock", "x": 2130, "y": 15, "width": 900, "height": 230},
        ],
        "nodes": [],
        "edges": [],
    }

    def binding(key, intended, note, result=None):
        graph["bindings"][key] = {"operation": "dummy", "note": note, "result": result or {}, "replaceWith": intended}

    binding(
        "pull_all",
        {"operation": "episode", "planPath": "TODO/pull-all.plan.json"},
        "TODO record the pull-all-cables episode",
    )
    binding(
        "plug_usb",
        {"operation": "episode", "planPath": "TODO/plug-usb.plan.json"},
        "TODO record insertion, including releasing the gripper",
    )
    binding(
        "return_usb_start",
        {"operation": "episode", "planPath": "TODO/return-usb-start.plan.json"},
        "TODO record returning to the unplug-policy start pose",
    )
    binding(
        "unplug_policy",
        {
            "operation": "dream",
            "checkpoint": "TODO/unplug-checkpoint",
            "run": True,
            "durationSeconds": 3,
            "task": "Unplug a cable from the hub, put it on the table, and let go.",
        },
        "TODO bind trained unplug policy; one action window per verification attempt",
    )
    binding(
        "stack_next",
        {
            "operation": "dream",
            "checkpoint": "TODO/stack-checkpoint",
            "run": True,
            "durationSeconds": 3,
            "task": "Add one block to the existing tower, release it, and clear the tower.",
        },
        "TODO bind trained stacking policy",
    )
    binding(
        "rebuild",
        {
            "operation": "dream",
            "checkpoint": "TODO/stack-checkpoint",
            "run": True,
            "durationSeconds": 3,
            "task": "The tower fell. Collect a fallen block and rebuild a stable base. Release it.",
        },
        "TODO bind restacking policy; no reset of the global match clock",
    )
    usb_result = {
        "known": False,
        "unplugged_count": 0,
        "plugged_count": 4,
        "gripper_released": False,
        "cable_on_table": False,
        "fully_inserted": False,
    }
    tower_result = {"known": False, "block_count": 0, "collapsed": False, "standing_free": False}
    for key, result, question in [
        (
            "verify_usb",
            usb_result,
            "Inspect all four hub ports. plugged_count is occupied ports. "
            "unplugged_count is 4 minus occupied ports; "
            "do not count unrelated loose cables. Mark known=false when a port is obscured. "
            "Is a cable fully inserted? Is the gripper released? Is the unplugged cable resting on the table?",
        ),
        (
            "verify_tower",
            tower_result,
            "Count blocks in the standing tower. Has the tower collapsed? Is it free standing with the gripper clear?",
        ),
    ]:
        properties = {
            name: {"type": "boolean" if isinstance(value, bool) else "integer"} for name, value in result.items()
        }
        binding(
            key,
            {
                "operation": "vlm",
                "provider": "openai",
                "model": "TODO/vision-model",
                "imageUrl": "http://127.0.0.1:8081/api/camera/overhead.jpg",
                "question": question,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
            "TODO connect automatic scene verification; unknown never passes a gate",
            result,
        )

    def node(key, kind, label, x, y, config=None):
        graph["nodes"].append({"id": key, "type": kind, "label": label, "x": x, "y": y, "config": config or {}})

    def action(key, label, x, y, impl, side):
        node(key, "action", label, x, y, {"operation": "binding", "binding": impl, "selection": side})

    def result(name, path, value=True, op="eq", ref=None):
        return {
            "kind": "result",
            "name": name,
            "path": path,
            "op": op,
            **({"valueRef": ref} if ref else {"value": value}),
        }

    def all_of(*conditions):
        return {"kind": "all", "conditions": list(conditions)}

    def decision(key, label, x, y, condition):
        node(key, "condition", label, x, y, {"condition": condition})

    def edge(source, port, target):
        graph["edges"].append({"id": f"{source}-{port}-{target}", "source": source, "port": port, "target": target})

    node("go", "start", "Go · five-minute match", 60, 60)
    node("match_clock", "fork", "Start match clock + cable work", 360, 60)
    action("pull_all", "Recorded pull-all attempt", 660, 60, "pull_all", "right")
    action("first_usb_view", "Verify any cable unplugged", 960, 60, "verify_usb", "right")
    decision(
        "first_usb_gate",
        "Known removal ≥ 1?",
        1260,
        60,
        all_of(result("first_usb_view", "known"), result("first_usb_view", "unplugged_count", 1, "gte")),
    )
    node("start_stack_and_clear", "fork", "USB gate passed · both jobs", 1560, 60)
    node(
        "deadline",
        "trigger",
        "At 5:00 from Go",
        2160,
        60,
        {"condition": {"kind": "elapsed", "secondsRef": "variables.runSeconds"}},
    )
    node("finish", "action", "End match · finish both arms", 2460, 60, {"operation": "finish", "selection": "both"})
    node("end", "end", "Match complete", 2760, 60)

    action("tower_view", "Inspect tower / watch for collapse", 60, 420, "verify_tower", "left")
    decision("tower_known", "Tower state known?", 360, 420, result("tower_view", "known"))
    decision("tower_fell", "Tower fell?", 660, 420, result("tower_view", "collapsed"))
    action("rebuild", "Recover fallen blocks · rebuild base", 960, 420, "rebuild", "left")
    decision(
        "tower_cap",
        "At editable height cap?",
        660,
        660,
        result("tower_view", "block_count", op="gte", ref="variables.maxBlocks"),
    )
    decision(
        "late_hold",
        "After 4:00 AND more than 5?",
        660,
        900,
        all_of(
            {"kind": "elapsed", "secondsRef": "variables.lateAtSeconds"},
            result("tower_view", "block_count", op="gt", ref="variables.lateHoldAbove"),
        ),
    )
    action("stack_next", "Place next block · release", 960, 900, "stack_next", "left")
    node(
        "watch_tower",
        "trigger",
        "Hold left · keep checking tower",
        360,
        900,
        {"condition": {"kind": "delay", "secondsRef": "variables.pollSeconds"}},
    )

    action("clear_view", "Inspect original hub cables", 1260, 420, "verify_usb", "right")
    decision("clear_known", "Hub state known?", 1560, 420, result("clear_view", "known"))
    decision("all_clear", "All four original USBs out?", 1860, 420, result("clear_view", "unplugged_count", 4, "gte"))
    action("clear_policy", "Trained unplug policy · clear hub", 1560, 660, "unplug_policy", "right")
    node(
        "clear_wait",
        "trigger",
        "Wait for next hub inspection",
        1260,
        660,
        {"condition": {"kind": "delay", "secondsRef": "variables.pollSeconds"}},
    )
    action("plug_attempt", "Attempt USB insertion · let go", 1860, 900, "plug_usb", "right")
    action("plug_view", "Verify insertion + released gripper", 2160, 900, "verify_usb", "right")
    decision(
        "plug_success",
        "Fully plugged AND released?",
        2460,
        900,
        all_of(
            result("plug_view", "known"),
            result("plug_view", "fully_inserted"),
            result("plug_view", "gripper_released"),
        ),
    )
    action("return_start", "Return to unplug start pose", 2460, 1140, "return_usb_start", "right")
    action("cycle_unplug", "Unplug API · set on table · let go", 2160, 1380, "unplug_policy", "right")
    action("unplug_view", "Verify unplugged / on table / released", 1860, 1380, "verify_usb", "right")
    decision(
        "unplug_success",
        "Unplugged + table + release?",
        1560,
        1380,
        all_of(
            result("unplug_view", "known"),
            result("unplug_view", "plugged_count", 0),
            result("unplug_view", "cable_on_table"),
            result("unplug_view", "gripper_released"),
        ),
    )

    for source, port, target in [
        ("go", "next", "match_clock"),
        ("match_clock", "next", "pull_all"),
        ("match_clock", "next", "deadline"),
        ("pull_all", "next", "first_usb_view"),
        ("first_usb_view", "next", "first_usb_gate"),
        ("first_usb_gate", "false", "pull_all"),
        ("first_usb_gate", "true", "start_stack_and_clear"),
        ("start_stack_and_clear", "next", "tower_view"),
        ("start_stack_and_clear", "next", "clear_view"),
        ("deadline", "next", "finish"),
        ("finish", "next", "end"),
        ("tower_view", "next", "tower_known"),
        ("tower_known", "false", "watch_tower"),
        ("tower_known", "true", "tower_fell"),
        ("tower_fell", "true", "rebuild"),
        ("rebuild", "next", "tower_view"),
        ("tower_fell", "false", "tower_cap"),
        ("tower_cap", "true", "watch_tower"),
        ("tower_cap", "false", "late_hold"),
        ("late_hold", "true", "watch_tower"),
        ("late_hold", "false", "stack_next"),
        ("stack_next", "next", "tower_view"),
        ("watch_tower", "next", "tower_view"),
        ("clear_view", "next", "clear_known"),
        ("clear_known", "false", "clear_wait"),
        ("clear_wait", "next", "clear_view"),
        ("clear_known", "true", "all_clear"),
        ("all_clear", "false", "clear_policy"),
        ("clear_policy", "next", "clear_view"),
        ("all_clear", "true", "plug_attempt"),
        ("plug_attempt", "next", "plug_view"),
        ("plug_view", "next", "plug_success"),
        ("plug_success", "false", "plug_attempt"),
        ("plug_success", "true", "return_start"),
        ("return_start", "next", "cycle_unplug"),
        ("cycle_unplug", "next", "unplug_view"),
        ("unplug_view", "next", "unplug_success"),
        ("unplug_success", "false", "cycle_unplug"),
        ("unplug_success", "true", "plug_attempt"),
    ]:
        edge(source, port, target)
    return copy.deepcopy(graph)


def main():
    import argparse
    import json
    import pathlib

    parser = argparse.ArgumentParser(description="Export the editable competition map, without executing it")
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(competition_graph(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
