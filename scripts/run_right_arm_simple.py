"""Reset -> unplug until four open ports -> reset -> continuous insertion."""

import argparse
import json
import pathlib
import signal
import threading
import time
import uuid

from dream_hackathon.dream_caller import ArmEndpoint, StationCaller
from dream_hackathon.right_arm_controller import atomic_json, move_to_pose
from dream_hackathon.usb_port_detector import UsbPortDetector, load_cv_detector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/right_arm_controller.json")
    p.add_argument("--run-id", default=None)
    p.add_argument("--go-time", type=float, default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--stall-seconds", type=float, default=12.0)
    p.add_argument("--debug-cv", action="store_true")
    p.add_argument("--cv-calibration", default=None)
    args = p.parse_args()
    config = json.loads(pathlib.Path(args.config).read_text())
    run_id = args.run_id or "right-simple-" + uuid.uuid4().hex
    directory = pathlib.Path(args.run_dir or "/tmp/relay/" + run_id)
    directory.mkdir(parents=True, exist_ok=True)
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    endpoint = ArmEndpoint(config.get("right_url", "http://127.0.0.1:8692"))
    caller = StationCaller("right", {"right": endpoint})
    detector, fetch = load_cv_detector()
    calibration = args.cv_calibration or config.get("usb_cv_calibration")
    if calibration:
        import numpy as np

        with np.load(calibration, allow_pickle=False) as loaded:
            detector.templates = {key: loaded[key] for key in loaded.files}
    ports = UsbPortDetector(run_id, detector=detector)
    poses = json.loads(pathlib.Path(config.get("poses_file", "configs/right_arm_start_poses.json")).read_text())
    reset = dict(poses["unplug_start"], tolerance_rad=0.06)
    go = args.go_time if args.go_time is not None else time.time()
    deadline = time.monotonic() + go - time.time() + 300
    phase = "await_go"
    active_checkpoint = None

    def report(**values):
        value = dict(run_id=run_id, side="right", phase=phase, time=time.time(), **values)
        value.setdefault("checkpoint", active_checkpoint if phase in ("unplug", "insert") else None)
        print(json.dumps(value), flush=True)
        atomic_json(directory / "right-status.json", value)
        with (directory / "events.jsonl").open("a") as output:
            output.write(json.dumps(value) + "\n")

    def stop_policy():
        caller.cancel_playback()
        endpoint.stop()
        while caller.snapshot().get("busy") and not cancel.is_set():
            cancel.wait(0.05)

    def start_policy(binding):
        nonlocal active_checkpoint
        caller.connect()
        while not cancel.is_set():
            s = caller.snapshot()
            if s.get("observation_error"):
                raise RuntimeError(s["observation_error"])
            if s.get("observation") is not None:
                break
            cancel.wait(0.05)
        if cancel.is_set():
            return
        b = config["bindings"][binding]
        active_checkpoint = b["checkpoint"]
        caller.start({"task": b["task"], "checkpoint": b["checkpoint"]}, run=True)
        report(checkpoint=b["checkpoint"])

    timer = threading.Timer(max(0, deadline - time.monotonic()), cancel.set)
    timer.daemon = True
    timer.start()
    try:
        cancel.wait(max(0, go - time.time()))
        if cancel.is_set():
            return
        phase = "reset_for_unplug"
        report()
        if move_to_pose(endpoint, reset, cancel).get("cancelled"):
            return
        phase = "unplug"
        start_policy("unplug_policy")
        anchor = None
        last_motion = time.monotonic()
        anchor_commands = 0
        open_frames = []
        last_log = 0
        debug_frame = 0
        while not cancel.is_set():
            s = caller.snapshot()
            if s.get("error") or s.get("observation_error"):
                raise RuntimeError(s.get("error") or s["observation_error"])
            measured = s.get("measured") or endpoint.state()
            if measured.get("error") or measured.get("stalled"):
                raise RuntimeError(measured.get("error") or "arm server stalled")
            if not 0 <= time.time() - measured.get("stamp", 0) <= 2:
                raise RuntimeError("right arm measurement stale")
            current = [*measured["q"], measured["grip"]]
            commands = s.get("commands", 0)
            if anchor is None or max(abs(a - b) for a, b in zip(current, anchor, strict=True)) > 0.015:
                anchor = current
                last_motion = time.monotonic()
                anchor_commands = commands
            if not s.get("busy"):
                raise RuntimeError("policy stopped before phase completion")
            if commands > anchor_commands and time.monotonic() - last_motion > args.stall_seconds:
                raise RuntimeError("policy motion stalled; stopped for operator review")
            if phase == "unplug":
                images, stamps, _, errors = fetch("http://127.0.0.1:8692/overhead.jpg", None, run_id)
                result = ports.observe(rgb_scene=images.get("scene"), stamps=stamps, now=time.monotonic())
                if args.debug_cv:
                    with (directory / "cv.jsonl").open("a") as output:
                        output.write(json.dumps(dict(result, camera_errors=errors, wall_time=time.time())) + "\n")
                # Three distinct captures spanning .4 s; unknown resets confirmation.
                if not errors and result["occupancy_known"] and result["open_count"] == 4:
                    capture = stamps[0].captured_at
                    if open_frames and capture - open_frames[-1] > 0.5:
                        open_frames = []
                    open_frames.append(capture)
                else:
                    open_frames = []
                if time.monotonic() - last_log >= 1:
                    cameras = (result.get("detection") or {}).get("cameras", {})
                    quality = {
                        c: dict(
                            geometry_valid=v.get("geometry_valid"),
                            reason=v.get("quality", {}).get("reason"),
                            port0=(v.get("evidence") or [None])[0],
                        )
                        for c, v in cameras.items()
                    }
                    report(ports=result["ports"], calls=s.get("calls"), commands=commands, cv_quality=quality)
                    if args.debug_cv and "scene" in images:
                        import cv2

                        cv2.imwrite(
                            str(directory / f"cv-scene-{debug_frame:04d}.jpg"),
                            cv2.cvtColor(images["scene"], cv2.COLOR_RGB2BGR),
                        )
                        debug_frame += 1
                    last_log = time.monotonic()
                if len(open_frames) >= 3 and open_frames[-1] - open_frames[0] >= 0.4:
                    stop_policy()
                    report(all_four_open=True)
                    phase = "reset_for_insert"
                    report()
                    if move_to_pose(endpoint, reset, cancel).get("cancelled"):
                        return
                    phase = "insert"
                    start_policy("plug_usb")
                    anchor = None
                    last_motion = time.monotonic()
                    last_log = 0
            elif time.monotonic() - last_log >= 1:
                report(calls=s.get("calls"), commands=commands, grip=measured["grip"])
                last_log = time.monotonic()
            cancel.wait(0.2)
    except Exception as error:
        report(error=str(error))
        raise
    finally:
        timer.cancel()
        cancel.set()
        try:
            caller.stop()
        finally:
            caller.close()
        phase = "stopped"
        report()


if __name__ == "__main__":
    main()
