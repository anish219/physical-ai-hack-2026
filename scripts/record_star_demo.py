"""Arm a read-only, timestamped three-camera/motion capture on the next Star Track.

Also starts the existing Star example recorder. SIGINT saves both recordings.
No power, tracking, or motion commands are issued by this program.
"""

import argparse
import concurrent.futures
import datetime
import json
import pathlib
import signal
import threading
import time
import urllib.request

import cv2
import numpy as np

BASE = "http://127.0.0.1:8081"
TASK = "pull out all the USBs at once"


def request(path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read()


def state():
    return json.loads(request("/api/state"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK)
    parser.add_argument(
        "--output-root", type=pathlib.Path, default=pathlib.Path.home() / "hackathon_data" / "usb-pull-demos"
    )
    args = parser.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = args.output_root.expanduser() / stamp
    root.mkdir(parents=True)
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    manifest = {
        "task": args.task,
        "status": "armed",
        "directory": str(root),
        "motion_file": "motion.jsonl",
        "motion_hz": 30,
        "video_hz": 20,
        "joint_units": "radians",
        "gripper_units": "normalized 0 closed to 1 open",
        "cameras": ["left", "overhead", "right"],
        "note": "Camera timestamps are HTTP acquisition times; video frame indexes map to the shared monotonic clock.",
    }

    def save_status():
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_status()
    print("ARMED", root, flush=True)
    while not done.is_set():
        first = state()
        if first.get("star", {}).get("tracking"):
            break
        done.wait(0.03)
    if done.is_set():
        manifest["status"] = "cancelled before tracking"
        save_status()
        return
    start = time.monotonic()
    manifest.update(
        status="recording",
        started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        start_monotonic=start,
        star_side=first["star"]["assigned_side"],
        initial_state=first,
    )
    save_status()
    # Use the existing recorder too, for its measured-state / action LeRobot dataset.
    started_example = not first["star"]["recording_requested"]
    if started_example:
        request("/api/star/control", {"recording_requested": True})
    print("RECORDING", root, flush=True)

    def camera(role):
        writer = None
        frames = 0
        try:
            with (root / (role + "-frames.jsonl")).open("w", encoding="utf-8") as timeline:
                while not done.is_set():
                    tick = time.monotonic()
                    encoded = request("/api/camera/" + role + ".jpg")
                    received = time.monotonic()
                    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise RuntimeError(role + " camera returned an invalid JPEG")
                    image = cv2.resize(image, (640, 360))
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(root / (role + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 360)
                        )
                        if not writer.isOpened():
                            raise RuntimeError("Could not open " + role + " video writer")
                    writer.write(image)
                    timeline.write(
                        json.dumps({"frame": frames, "t": received - start, "request_t": tick - start}) + "\n"
                    )
                    timeline.flush()
                    frames += 1
                    done.wait(max(0, 0.05 - (time.monotonic() - tick)))
        finally:
            if writer is not None:
                writer.release()
            manifest[role + "_frames"] = frames

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            videos = {role: pool.submit(camera, role) for role in manifest["cameras"]}
            try:
                with (root / "motion.jsonl").open("w", encoding="utf-8") as motion:
                    count = 0
                    while not done.is_set():
                        tick = time.monotonic()
                        current = state()
                        sample = {
                            "t": time.monotonic() - start,
                            "request_t": tick - start,
                            "arms": {
                                side: {
                                    key: arm.get(key)
                                    for key in ("q", "grip", "goal", "goal_grip", "arm_open", "moving", "homing")
                                }
                                for side, arm in current["arms"].items()
                            },
                            "star": current.get("star"),
                            "inputs": current["inputs"],
                        }
                        motion.write(json.dumps(sample) + "\n")
                        motion.flush()
                        count += 1
                        if count % 30 == 0:
                            manifest.update(motion_samples=count, duration_s=sample["t"])
                            save_status()
                        done.wait(max(0, 1 / 30 - (time.monotonic() - tick)))
            finally:
                done.set()
            for role, future in videos.items():
                try:
                    future.result()
                except Exception as error:
                    manifest[role + "_error"] = str(error)
    finally:
        done.set()
        if started_example:
            current = state()
            if current.get("star", {}).get("recording_requested"):
                request("/api/star/control", {"recording_requested": False})
        manifest.update(
            status="saved",
            duration_s=time.monotonic() - start,
            finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        save_status()
        print("SAVED", root, flush=True)


if __name__ == "__main__":
    main()
