import threading

import pytest

from dream_hackathon.right_arm_controller import RightArmController, atomic_json, validate_config


class Harness:
    def __init__(self):
        self.now = 1000.0
        self.calls, self.gates, self.stops = [], [], []
        self.scene = dict(
            known=True,
            plugged_count=3,
            unplugged_count=1,
            fully_inserted=False,
            gripper_released=True,
            cable_on_table=True,
        )
        self.worker = RightArmController(
            run_id="match-a",
            go_time=1000.0,
            execute=self.execute,
            stop=lambda: self.stops.append(True),
            publish=self.gates.append,
            clock=lambda: self.now,
            wall_clock=lambda: self.now,
            poll_seconds=0.001,
        )
        self.worker.state = "pull_all"

    def execute(self, binding, request, cancel):
        self.calls.append(binding)
        if binding != "verify_usb":
            return {}
        return dict(
            self.scene,
            run_id=request["run_id"],
            inspection_id=request["inspection_id"],
            observed_at=self.now,
            evidence={"deterministic": "det.json", "vlm": "vlm.json"},
        )


def test_first_removal_retries_and_publishes_once():
    h = Harness()
    for scene in [dict(known=False), dict(known=True, unplugged_count=0, plugged_count=4)]:
        h.scene.update(scene)
        h.worker.step()
        h.worker.step()
        assert h.worker.state == "pull_all"
        assert not h.gates
    h.scene.update(known=True, unplugged_count=1, plugged_count=3)
    h.worker.step()
    h.worker.step()
    assert h.worker.state == "clear_view"
    assert h.gates[0]["run_id"] == "match-a"
    assert h.gates[0]["first_removal"] is True
    h.worker.state = "first_usb_view"
    h.worker.step()
    assert len(h.gates) == 1


def test_clear_unknown_waits_without_moving_then_policy_then_insert():
    h = Harness()
    h.worker.state = "clear_view"
    h.scene["known"] = False
    h.worker.step()
    assert h.calls == ["verify_usb"]
    assert h.worker.state == "clear_view"
    h.scene["known"] = True
    h.worker.step()
    assert h.worker.state == "clear_policy"
    h.worker.step()
    assert h.calls[-1] == "unplug_policy"
    h.scene.update(plugged_count=0, unplugged_count=4)
    h.worker.step()
    assert h.worker.state == "plug_usb"


def test_insertion_return_and_cycle_require_all_scene_fields():
    h = Harness()
    h.worker.state = "plug_usb"
    h.worker.step()
    h.scene.update(fully_inserted=True, gripper_released=False)
    h.worker.step()
    assert h.worker.state == "plug_usb"
    h.worker.step()
    h.scene["gripper_released"] = True
    h.worker.step()
    assert h.worker.state == "return_usb_start"
    h.worker.step()
    h.worker.step()
    assert h.calls[-2:] == ["return_usb_start", "unplug_policy"]
    h.scene.update(plugged_count=0, unplugged_count=4, cable_on_table=False)
    h.worker.step()
    assert h.worker.state == "cycle_unplug"
    h.worker.step()
    h.scene["cable_on_table"] = True
    h.worker.step()
    assert h.worker.state == "plug_usb"
    assert not h.gates


@pytest.mark.parametrize(
    "change",
    [
        {"known": "true"},
        {"plugged_count": True},
        {"unplugged_count": 4},
        {"gripper_released": "yes"},
        {"plugged_count": -1},
    ],
)
def test_invalid_scene_cannot_release_first_gate(change):
    h = Harness()
    h.scene.update(change)
    h.worker.state = "first_usb_view"
    h.worker.step()
    assert h.worker.state == "pull_all"
    assert not h.gates


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "old-match"},
        {"inspection_id": "old-inspection"},
        {"observed_at": 999.0},
        {"observed_at": float("nan")},
        {"observed_at": 1001.0},
        {"evidence": {}},
    ],
)
def test_stale_or_unsupported_evidence_never_releases(change):
    h = Harness()
    original = h.worker.execute
    h.worker.execute = lambda *a: dict(original(*a), **change)
    h.worker.state = "first_usb_view"
    h.worker.step()
    assert not h.gates
    assert h.worker.state == "pull_all"


def test_shared_deadline_survives_retries_and_stops_only_once():
    h = Harness()
    h.now += 299
    h.worker.step()
    assert h.worker.elapsed() == 299
    h.now += 1
    h.worker.step()
    h.worker.stop()
    assert h.calls == ["pull_all"]
    assert len(h.stops) == 1


def test_result_arriving_after_deadline_is_discarded():
    h = Harness()
    original = h.worker.execute

    def execute(*args):
        result = original(*args)
        h.now += 300
        return result

    h.worker.execute = execute
    h.worker.state = "first_usb_view"
    h.worker.step()
    assert not h.gates
    assert h.worker.cancel.is_set()


def test_exception_is_not_a_physical_retry():
    h = Harness()
    h.worker.go_monotonic = h.worker.clock()

    def broken(*args):
        raise RuntimeError("provider failed")

    h.worker.execute = broken
    with pytest.raises(RuntimeError, match="provider failed"):
        h.worker.run()
    assert h.worker.state == "error"
    assert len(h.stops) == 1


def test_expired_run_never_executes():
    h = Harness()
    h.now += 301
    h.worker.run()
    assert not h.calls
    assert h.worker.state == "complete"


def test_atomic_gate(tmp_path):
    import json

    path = tmp_path / "gate.json"
    atomic_json(path, {"run_id": "match-a"})
    assert json.loads(path.read_text()) == {"run_id": "match-a"}
    assert list(tmp_path.iterdir()) == [path]


def test_stop_interrupts_wait_for_future_go():
    h = Harness()
    h.worker.go_monotonic += 100
    thread = threading.Thread(target=h.worker.run)
    thread.start()
    h.worker.stop()
    thread.join(1)
    assert not thread.is_alive()
    assert not h.calls


def test_config_rejects_dummy_and_other_arm():
    config = {"judge_approved_repeated_cycle": True, "bindings": {}}
    with pytest.raises(ValueError, match="pull_all"):
        validate_config(config)
    config["bindings"]["pull_all"] = {"operation": "script", "selection": "left"}
    with pytest.raises(ValueError, match="right only"):
        validate_config(config)


def test_native_adapter_targets_only_right_and_stop_does_not_power_or_home(monkeypatch):
    from dream_hackathon import dream_caller
    from dream_hackathon.right_arm_controller import make_executor

    requests = []

    class Endpoint:
        def __init__(self, url):
            self.url = url

        def state(self):
            return {"q": [0] * 6, "grip": 0.5}

        def request(self, path, body):
            requests.append((self.url, path, body))

    class Caller:
        def __init__(self, side, endpoints):
            assert side == "right" and set(endpoints) == {"right"}
            self.sides = ("right",)
            self.endpoint = endpoints["right"]

        def connect(self):
            pass

        def snapshot(self):
            return {"observation": {}, "busy": False}

        def start(self, values, *, run):
            requests.append(("policy", values, run))

        def stop(self):
            self.endpoint.request("/stop", {})

        def close(self):
            pass

    monkeypatch.setattr(dream_caller, "ArmEndpoint", Endpoint)
    monkeypatch.setattr(dream_caller, "StationCaller", Caller)
    config = {
        "right_url": "http://right-only:8692",
        "bindings": {
            "unplug_policy": {
                "operation": "dream",
                "checkpoint": "native-checkpoint",
                "task": "matching prompt",
                "run": True,
                "durationSeconds": 3,
            }
        },
    }
    execute, stop, close = make_executor(config, "match-a")
    assert execute("unplug_policy", {}, threading.Event()) == {"calls": None, "rows": None}
    stop()
    close()
    assert requests == [
        ("policy", {"task": "matching prompt", "checkpoint": "native-checkpoint"}, True),
        ("http://right-only:8692", "/stop", {}),
    ]


def test_configuration_requires_full_cv_replay_and_resolved_assets(tmp_path):
    import json

    plan = tmp_path / "plan.json"
    targets = tmp_path / "targets.jsonl"
    targets.write_text('{"t": 0, "q": [0,0,0,0,0,0], "grip": 0.5}\n')
    plan.write_text(json.dumps({"targets_file": "targets.jsonl"}))
    config = {
        "judge_approved_repeated_cycle": True,
        "bindings": {
            "pull_all": {
                "operation": "script",
                "argv": ["python", "scripts/replay_usb_bundle.py", "--execute", "--server", "http://127.0.0.1:8692"],
            },
            "verify_usb": {"operation": "http", "url": "http://verifier"},
            "unplug_policy": {"operation": "dream", "checkpoint": "checkpoint", "task": "task", "durationSeconds": 3},
            "plug_usb": {"operation": "episode", "planPath": str(plan)},
            "return_usb_start": {"operation": "episode", "planPath": str(plan)},
        },
    }
    validate_config(config)
    config["bindings"]["pull_all"]["argv"] = ["python", "generic-playback.py"]
    with pytest.raises(ValueError, match="CV alignment"):
        validate_config(config)


def test_pose_return_waits_for_fresh_measured_arrival_and_preserves_grip():
    from dream_hackathon.right_arm_controller import move_to_pose

    class Endpoint:
        def __init__(self):
            self.reads, self.commands = 0, []

        def state(self):
            self.reads += 1
            return {"q": [1.0] * 6, "grip": 0.42, "stamp": 1 if self.reads < 3 else 2, "moving": False}

        def request(self, path, body):
            self.commands.append((path, body))

    endpoint = Endpoint()
    result = move_to_pose(endpoint, {"q": [1.0] * 6}, threading.Event())
    assert result["pose_reached"]
    assert endpoint.reads == 3  # Matching cached q alone cannot complete a return.
    assert endpoint.commands == [("/move", {"q": [1.0] * 6, "grip": 0.42, "rate": 0.3})]


def test_return_cancel_and_listener_error_cannot_become_policy_success():
    from dream_hackathon.right_arm_controller import move_to_pose

    cancel = threading.Event()

    class Endpoint:
        def state(self):
            cancel.set()
            return {"q": [0.0] * 6, "grip": 0.5, "stamp": 1, "moving": True}

        def request(self, *args):
            pytest.fail("Cancelled return must not send a move")

    assert move_to_pose(Endpoint(), {"q": [1.0] * 6}, cancel) == {"cancelled": True}

    class FailedEndpoint:
        def state(self):
            return {"error": "joint lag: stopped"}

    with pytest.raises(RuntimeError, match="joint lag"):
        move_to_pose(FailedEndpoint(), {"q": [1.0] * 6}, threading.Event())


@pytest.mark.parametrize(
    "pose",
    [
        {"q": [0] * 5},
        {"q": [float("nan")] * 6},
        {"q": [0] * 6, "rate_rad_s": 0},
        {"q": [0] * 6, "grip": 27},
        {"q": [0] * 6, "waypoints": [[0] * 5]},
    ],
)
def test_pose_rejects_invalid_joints_rates_and_mm_grip(pose):
    from dream_hackathon.right_arm_controller import validate_pose

    with pytest.raises(ValueError):
        validate_pose(pose)


def test_policy_retries_return_to_pose_before_recalling_exact_models(monkeypatch):
    from dream_hackathon import dream_caller
    from dream_hackathon.right_arm_controller import make_executor

    calls = []

    class Endpoint:
        def __init__(self, url):
            self.url, self.q, self.stamp = url, [0.0] * 6, 0

        def state(self):
            self.stamp += 1
            return {"q": self.q, "grip": 0.4, "stamp": self.stamp, "moving": False}

        def request(self, path, body):
            calls.append((path, body))
            if path == "/move":
                self.q = body["q"]

    class Caller:
        def __init__(self, side, endpoints):
            self.sides = ("right",)

        def connect(self):
            pass

        def snapshot(self):
            return {"observation": {}, "busy": False}

        def start(self, values, *, run):
            calls.append(("policy", values["checkpoint"]))

        def close(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(dream_caller, "ArmEndpoint", Endpoint)
    monkeypatch.setattr(dream_caller, "StationCaller", Caller)
    unplug = "dream-1-ft:f9ed26ed-usb-unplug-right-500-20261107"
    plug = "dream-1-ft:f9ed26ed-usb-manip-gen-cos1000-20261108"
    config = {
        "poses": {"start": {"q": [1.0] * 6}},
        "bindings": {
            "unplug_policy": {
                "operation": "dream",
                "checkpoint": unplug,
                "task": "unplug",
                "durationSeconds": 3,
                "prepare": {"pose": "start", "first": True, "retry_every": 1},
            },
            "plug_usb": {"operation": "dream", "checkpoint": plug, "task": "insert", "durationSeconds": 3},
            "return_usb_start": {"operation": "pose", "pose": "start"},
        },
    }
    execute, stop, close = make_executor(config, "run")
    for name in ("unplug_policy", "unplug_policy", "plug_usb", "return_usb_start"):
        execute(name, {}, threading.Event())
    assert [name for name, _ in calls] == ["/move", "policy", "/move", "policy", "policy", "/move"]
    assert [value for name, value in calls if name == "policy"] == [unplug, unplug, plug]
    assert all(value["grip"] == 0.4 for name, value in calls if name == "/move")
    stop()
    # Even a late request from an adapter cannot move after explicit Stop.
    assert execute("return_usb_start", {}, threading.Event()) == {"cancelled": True}
    assert len(calls) == 6
    close()


def test_retry_counters_reset_after_verified_success_without_resetting_match_clock():
    h = Harness()
    h.worker.state = "plug_usb"
    h.worker.step()
    h.worker.step()  # Failed insertion verification.
    assert h.worker.attempts["plug_usb"] == 1
    h.worker.step()
    assert h.worker.attempts["plug_usb"] == 2
    h.scene.update(fully_inserted=True)
    h.now += 20
    h.worker.step()
    assert "plug_usb" not in h.worker.attempts
    assert h.worker.elapsed() == 20
    h.worker.state = "cycle_unplug"
    h.worker.step()
    h.scene.update(plugged_count=0, unplugged_count=4, cable_on_table=True, gripper_released=True)
    h.worker.step()
    assert "cycle_unplug" not in h.worker.attempts
    assert h.worker.state == "plug_usb"
