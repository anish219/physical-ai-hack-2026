"""Share measured arm poses between independent controllers without sharing CAN ports.

The last published measurement remains usable as static context after its owner stops.
Joint angles are radians; grip is normalized 0 (closed) to 1 (open).
"""

import json
import os
import pathlib
import time

import numpy as np

POSE_DIR = pathlib.Path("~/.cache/dream_hackathon/arm_context").expanduser()


def publish(side, joints, grip, *, directory=None):
    directory = pathlib.Path(directory or POSE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    pose = {"side": side, "joints_rad": [float(x) for x in joints], "grip": float(grip), "measured_at": time.time()}
    temporary = directory / f".{side}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(pose) + "\n")
    temporary.replace(directory / f"{side}.json")


def read(side, *, directory=None):
    path = pathlib.Path(directory or POSE_DIR) / f"{side}.json"
    try:
        pose = json.loads(path.read_text())
    except FileNotFoundError:
        raise RuntimeError(
            f"No measured {side} context pose at {path}; "
            "run that arm's pose publisher or capture its static pose first"
        ) from None
    joints = np.asarray(pose["joints_rad"], dtype=np.float64)
    grip = float(pose["grip"])
    if pose["side"] != side or joints.shape != (6,) or not np.isfinite(joints).all() or not np.isfinite(grip):
        raise ValueError(f"Invalid measured context pose: {path}")
    return joints, grip


def main():
    """Capture a static pose from a free Metal CAN adapter, without changing torque."""
    import argparse

    from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig

    from dream_hackathon import config, station_lock
    from dream_hackathon.arms.makermods import GRIPPER, MOTORS, _grip

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", default="configs/station.yaml")
    parser.add_argument("--side", required=True, choices=("left", "right"))
    args = parser.parse_args()
    station = config.load(args.config)
    if station.arm != "metal":
        parser.error("static capture currently supports Metal arms")
    with station_lock.acquire_sides("arm-context-capture", [args.side]):
        device = MetalFollower(
            MetalFollowerConfig(
                port=station.arms[args.side].port,
                id=f"dream_hackathon_{args.side}",
                can_interface=station.makermods.can_interface,
            )
        )
        bus = device.bus
        try:
            bus.connect(handshake=False)
            angles = {motor: float(bus.read("Present_Position", motor)) for motor in (*MOTORS, GRIPPER)}
            publish(
                args.side, np.radians([angles[m] for m in MOTORS]), _grip(station.arms[args.side], angles[GRIPPER])
            )
        finally:
            bus.disconnect(disable_torque=False)
    print(POSE_DIR / f"{args.side}.json")


if __name__ == "__main__":
    main()
