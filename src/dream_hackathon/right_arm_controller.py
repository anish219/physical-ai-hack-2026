"""Independent right-arm relay worker; importing this module never touches hardware."""

import argparse
import contextlib
import hashlib
import json
import math
import os
import pathlib
import signal
import tempfile
import threading
import time
import uuid

BINDINGS = ("pull_all", "verify_usb", "unplug_policy", "plug_usb", "return_usb_start")


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RightArmController:
    """Injected operations execute synchronously; stop must cancel active operations.

    go_time is Unix seconds shared by both workers, converted once to monotonic
    time so retries and wall-clock corrections cannot reset the match clock.
    """

    def __init__(
        self,
        *,
        run_id,
        go_time,
        execute,
        stop,
        publish,
        poll_seconds=0.5,
        clock=time.monotonic,
        wall_clock=time.time,
        status=None,
    ):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("A unique shared run_id is required")
        if not math.isfinite(go_time):
            raise ValueError("go_time must be finite Unix seconds")
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive and finite")
        self.run_id, self.go_time = run_id, go_time
        self.clock, self.wall_clock = clock, wall_clock
        self.go_monotonic = clock() + go_time - wall_clock()
        self.execute, self.stop_operation, self.publish = execute, stop, publish
        self.poll_seconds, self.status_callback = poll_seconds, status
        self.cancel = threading.Event()
        self.stop_lock = threading.Lock()
        self.stopped = False
        self.gate_published = False
        self.attempts = {}
        self.state = "await_go"

    def elapsed(self):
        return max(0.0, self.clock() - self.go_monotonic)

    def stop(self):
        self.cancel.set()
        with self.stop_lock:
            if not self.stopped:
                self.stopped = True
                self.stop_operation()

    def active(self):
        if self.elapsed() >= 300:
            self.stop()
        return not self.cancel.is_set()

    def report(self, state, **details):
        self.state = state
        if self.status_callback:
            self.status_callback(
                dict(run_id=self.run_id, side="right", state=state, elapsed_seconds=self.elapsed(), **details)
            )

    def operation(self, binding):
        if not self.active():
            return None
        result = self.execute(
            binding,
            {
                "run_id": self.run_id,
                "go_time": self.go_time,
                "elapsed_seconds": self.elapsed(),
                "phase": self.state,
                "attempt_index": self.attempts.get(self.state, 0),
            },
            self.cancel,
        )
        if not self.active():
            return None
        if not isinstance(result, dict):
            raise ValueError(f"{binding} must return a JSON object")
        if result.get("cancelled"):
            self.stop()
            return None
        if result.get("returncode", 0) != 0:
            raise RuntimeError(f"{binding} failed: exit {result['returncode']}")
        self.attempts[self.state] = self.attempts.get(self.state, 0) + 1
        return result

    def inspect(self):
        inspection_id = uuid.uuid4().hex
        requested = self.wall_clock()
        result = self.execute(
            "verify_usb",
            {
                "run_id": self.run_id,
                "go_time": self.go_time,
                "inspection_id": inspection_id,
                "requested_at": requested,
                "elapsed_seconds": self.elapsed(),
            },
            self.cancel,
        )
        if not self.active():
            return {"known": False}
        if not isinstance(result, dict):
            raise ValueError("verify_usb must return a JSON object")
        if result.get("cancelled"):
            self.stop()
            return {"known": False}
        if result.get("returncode", 0) != 0:
            raise RuntimeError("verify_usb provider failed")
        if result.get("known") is not True:
            return {"known": False}
        # One fresh, coherent scene; never merge fields from prior inspections.
        count = result.get("plugged_count")
        observed = result.get("observed_at")
        valid = (
            result.get("run_id") == self.run_id
            and result.get("inspection_id") == inspection_id
            and type(count) is int
            and 0 <= count <= 4
            and type(result.get("unplugged_count")) is int
            and result["unplugged_count"] == 4 - count
            and type(observed) in (int, float)
            and math.isfinite(observed)
            and requested <= observed <= self.wall_clock()
            and all(type(result.get(key)) is bool for key in ("fully_inserted", "gripper_released", "cable_on_table"))
            and isinstance(result.get("evidence"), dict)
            and all(
                isinstance(result["evidence"].get(key), str) and result["evidence"][key]
                for key in ("deterministic", "vlm")
            )
        )
        return result if valid else {"known": False}

    def step(self):
        """One state transition, useful for hardware-free integration tests."""
        if not self.active():
            return
        state = self.state
        if state in ("pull_all", "clear_policy", "plug_usb", "return_usb_start", "cycle_unplug"):
            binding = "unplug_policy" if state in ("clear_policy", "cycle_unplug") else state
            if self.operation(binding) is None:
                return
            self.report(
                {
                    "pull_all": "first_usb_view",
                    "clear_policy": "clear_view",
                    "plug_usb": "plug_view",
                    "return_usb_start": "cycle_unplug",
                    "cycle_unplug": "unplug_view",
                }[state]
            )
            return
        scene = self.inspect()
        if not self.active():
            return
        known = scene.get("known") is True
        if state == "first_usb_view":
            if known and scene["unplugged_count"] >= 1:
                if not self.gate_published:
                    self.publish(
                        dict(
                            version=1,
                            run_id=self.run_id,
                            go_time=self.go_time,
                            first_removal=True,
                            unplugged_count=scene["unplugged_count"],
                            observed_at=scene["observed_at"],
                            inspection_id=scene["inspection_id"],
                            evidence=scene["evidence"],
                        )
                    )
                    self.gate_published = True
                self.report("clear_view")
            else:
                self.report("pull_all")
        elif state == "clear_view":
            if not known:
                self.cancel.wait(self.poll_seconds)
            else:
                if scene["unplugged_count"] == 4:
                    self.attempts.pop("clear_policy", None)
                self.report("plug_usb" if scene["unplugged_count"] == 4 else "clear_policy")
        elif state == "plug_view":
            if known and scene["fully_inserted"] and scene["gripper_released"]:
                self.attempts.pop("plug_usb", None)
            self.report(
                "return_usb_start" if known and scene["fully_inserted"] and scene["gripper_released"] else "plug_usb"
            )
        elif state == "unplug_view":
            if known and scene["plugged_count"] == 0 and scene["cable_on_table"] and scene["gripper_released"]:
                self.attempts.pop("cycle_unplug", None)
            self.report(
                "plug_usb"
                if known and scene["plugged_count"] == 0 and scene["cable_on_table"] and scene["gripper_released"]
                else "cycle_unplug"
            )
        else:
            raise ValueError(f"Unexpected state {state}")

    def run(self):
        timer = threading.Timer(max(0, self.go_monotonic + 300 - self.clock()), self.stop)
        timer.daemon = True
        timer.start()
        try:
            self.report("await_go")
            self.cancel.wait(max(0, self.go_monotonic - self.clock()))
            if self.active():
                self.report("pull_all")
            while self.active():
                self.step()
            self.report("complete" if self.elapsed() >= 300 else "stopped")
        except Exception as error:
            # Errors terminate the worker; no recovery motion or phase change.
            self.report("error", error=str(error))
            raise
        finally:
            timer.cancel()
            self.stop()


def load_poses(config):
    poses = {}
    if config.get("poses_file"):
        poses.update(json.loads(pathlib.Path(config["poses_file"]).read_text()))
    poses.update(config.get("poses", {}))
    return poses


def validate_pose(pose):
    points = [*pose.get("waypoints", []), pose.get("q")]
    for q in points:
        if (
            not isinstance(q, list)
            or len(q) != 6
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in q)
        ):
            raise ValueError("A pose/waypoint must contain six finite joint radians")
    for key, default in (("rate_rad_s", 0.3), ("tolerance_rad", 0.02)):
        value = pose.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Pose {key} must be positive and finite")
    grip = pose.get("grip", "preserve")
    if grip != "preserve" and (type(grip) not in (int, float) or not math.isfinite(grip) or not 0 <= grip <= 1):
        raise ValueError("Pose grip must be preserve or normalized 0–1")


def move_to_pose(endpoint, pose, cancel):
    """Use the listener's existing rate-limited /move; wait for measured arrival.

    No fabricated replay, new power mode, or fixed sleep used as arrival evidence.
    No extra movement timeout: cancellation and the shared deadline still apply.
    """
    validate_pose(pose)
    points = [*pose.get("waypoints", []), pose["q"]]
    tolerance = pose.get("tolerance_rad", 0.02)
    for q in points:
        if cancel.is_set():
            return {"cancelled": True}
        initial = endpoint.state()
        if initial.get("error") or initial.get("stalled"):
            raise RuntimeError(f"Right listener cannot return: {initial.get('error') or 'stalled'}")
        if (
            not isinstance(initial.get("q"), list)
            or len(initial["q"]) != 6
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in initial["q"])
        ):
            raise ValueError("Right listener has no measured six-joint state")
        if type(initial.get("stamp")) not in (int, float) or not math.isfinite(initial["stamp"]):
            raise ValueError("Right listener has no timestamped joint measurement")
        grip = initial.get("grip") if pose.get("grip", "preserve") == "preserve" else pose["grip"]
        if type(grip) not in (int, float) or not math.isfinite(grip) or not 0 <= grip <= 1:
            raise ValueError("Right listener has no normalized measured gripper value")
        if cancel.is_set():
            return {"cancelled": True}
        endpoint.request("/move", {"q": q, "grip": grip, "rate": pose.get("rate_rad_s", 0.3)})
        while not cancel.is_set():
            measured = endpoint.state()
            if measured.get("error") or measured.get("stalled"):
                raise RuntimeError(f"Right return failed: {measured.get('error') or 'stalled'}")
            actual = measured.get("q")
            if (
                isinstance(actual, list)
                and len(actual) == 6
                and all(type(x) in (int, float) and math.isfinite(x) for x in actual)
                and type(measured.get("stamp")) in (int, float)
                and math.isfinite(measured["stamp"])
                and measured["stamp"] > initial["stamp"]
                and measured.get("moving") is False
                and max(abs(a - b) for a, b in zip(actual, q, strict=True)) <= tolerance
                and abs(measured.get("grip", -1) - grip) <= 0.02
            ):
                break
            cancel.wait(1 / 30)
        if cancel.is_set():
            return {"cancelled": True}
    return {"pose_reached": True, "q": pose["q"], "waypoints": len(points)}


def make_executor(config, run_id):
    """Reuse branch adapters with only the right listener and right Dream client."""
    from dream_hackathon.dream_caller import ArmEndpoint, StationCaller
    from dream_hackathon.orchestration import Operations

    endpoint = ArmEndpoint(config.get("right_url", "http://127.0.0.1:8692"))
    command_lock = threading.RLock()
    commands_disabled = threading.Event()
    transport_request = endpoint.request

    def guarded_request(path, body=None):
        if path in ("/move", "/target", "/stop"):
            with command_lock:
                if commands_disabled.is_set() and path != "/stop":
                    return b"{}"
                return transport_request(path, body)
        return transport_request(path, body)

    endpoint.request = guarded_request

    class Control:
        def post(self, path, body):
            return endpoint.request(path, body)

    class Node:
        control = Control()

        def snapshot(self):
            state = endpoint.state()
            return {**state, "arm_open": state.get("q") is not None}

    caller = StationCaller("right", {"right": endpoint})
    operations = Operations({"right": caller}, {"right": Node()})
    poses = load_poses(config)
    attempts = {}

    def execute(binding, request, cancel):
        if commands_disabled.is_set() or cancel.is_set():
            return {"cancelled": True}
        implementation = config["bindings"][binding]
        if implementation["operation"] == "pose":
            if implementation.get("enabled", True) is False:
                return {"pose_skipped": True}
            return move_to_pose(endpoint, poses[implementation["pose"]], cancel)
        preparation = implementation.get("prepare", {})
        phase = request.get("phase", binding)
        previous = request.get("attempt_index", attempts.get(phase, 0))
        retry_every = preparation.get("retry_every", 0)
        prepare = (previous == 0 and preparation.get("first", False)) or (
            previous > 0 and retry_every > 0 and previous % retry_every == 0
        )
        if preparation.get("pose") and prepare:
            result = move_to_pose(endpoint, poses[preparation["pose"]], cancel)
            if result.get("cancelled"):
                return result
        if cancel.is_set():
            return {"cancelled": True}
        attempts[phase] = previous + 1
        context = dict(bindings=config["bindings"], variables=request, results={}, signals={})
        return operations.execute(
            {"operation": "binding", "binding": binding, "selection": "right"}, context, cancel, run_id, "right"
        )

    def stop():
        commands_disabled.set()
        try:
            operations.stop(run_id)
        finally:
            caller.stop()  # Also stops a script replay's right listener; never homes or powers off.

    return execute, stop, caller.close


def validate_config(config):
    if config.get("judge_approved_repeated_cycle") is not True:
        raise ValueError("Repeated reinsertion requires explicit judge-approved strategy configuration")
    poses = load_poses(config)
    for pose in poses.values():
        validate_pose(pose)
    for name in BINDINGS:
        binding = config.get("bindings", {}).get(name, {})
        if binding.get("operation") not in ("script", "http", "episode", "dream", "vlm", "pose"):
            raise ValueError(f"Supply a real {name} binding; dummies cannot execute")
        if binding.get("selection", "right") != "right":
            raise ValueError(f"{name} must target right only")
        if "TODO" in json.dumps(binding) or "REPLACE" in json.dumps(binding):
            raise ValueError(f"Resolve the missing {name} binding")
        operation = binding["operation"]
        if operation == "pose" and binding.get("pose") not in poses:
            raise ValueError(f"Supply the named pose for {name}")
        if operation == "pose" and type(binding.get("enabled", True)) is not bool:
            raise ValueError(f"{name} enabled must be boolean")
        preparation = binding.get("prepare", {})
        if preparation.get("pose") is not None and preparation["pose"] not in poses:
            raise ValueError(f"Missing preparation pose for {name}")
        if type(preparation.get("retry_every", 0)) is not int or preparation.get("retry_every", 0) < 0:
            raise ValueError(f"{name} prepare.retry_every must be a nonnegative integer")
        if type(preparation.get("first", False)) is not bool:
            raise ValueError(f"{name} prepare.first must be boolean")
        if operation == "episode":
            plan_path = pathlib.Path(binding["planPath"])
            plan = json.loads(plan_path.read_text())
            if not (plan_path.parent / plan["targets_file"]).is_file():
                raise ValueError(f"Missing target recording for {name}")
        if operation == "script" and (not isinstance(binding.get("argv"), list) or not binding["argv"]):
            raise ValueError(f"Supply argv for {name}")
        if operation == "dream":
            duration = binding.get("durationSeconds")
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Supply a positive durationSeconds for {name}")
            if not binding.get("task") or not binding.get("checkpoint"):
                raise ValueError(f"Supply task and checkpoint for {name}")
    if config["bindings"]["pull_all"]["operation"] != "script":
        raise ValueError("pull_all must invoke the full CV-aligned replay script")
    argv = config["bindings"]["pull_all"]["argv"]
    if not any(pathlib.Path(str(arg)).name == "replay_usb_bundle.py" for arg in argv) or "--execute" not in argv:
        raise ValueError("pull_all must execute scripts/replay_usb_bundle.py with CV alignment")
    if "--server" not in argv or argv[argv.index("--server") + 1 :][:1] != [
        config.get("right_url", "http://127.0.0.1:8692")
    ]:
        raise ValueError("pull_all --server must match right_url")
    if config["bindings"]["verify_usb"]["operation"] not in ("http", "script"):
        raise ValueError("verify_usb must combine deterministic and VLM evidence in a provider")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--go-time", type=float, required=True, help="Shared Unix seconds from launcher")
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate bindings without connecting")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.check:
        print("Right-arm bindings validated; no connections made.")
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive per-run ownership prevents overwriting an earlier gate or duplicate launch.
    import fcntl

    identity = hashlib.sha256(config.get("right_url", "http://127.0.0.1:8692").rstrip("/").encode()).hexdigest()
    listener_lock = pathlib.Path(tempfile.gettempdir()) / f"dream-right-worker-{identity}.lock"
    with contextlib.ExitStack() as stack:
        for path in (args.run_dir / "right.lock", listener_lock):
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.run_dir / "first-removal.json").exists():
            raise ValueError("Use a new run directory; a first-removal gate already exists")
        execute, stop, close = make_executor(config, args.run_id)
        controller = RightArmController(
            run_id=args.run_id,
            go_time=args.go_time,
            execute=execute,
            stop=stop,
            publish=lambda gate: atomic_json(args.run_dir / "first-removal.json", gate),
            status=lambda state: atomic_json(args.run_dir / "right-status.json", state),
            poll_seconds=config.get("poll_seconds", 0.5),
        )
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: controller.cancel.set())
        try:
            controller.run()
        finally:
            close()


if __name__ == "__main__":
    main()
