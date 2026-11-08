"""Capture existing right-listener measurements; never send a motion command."""

import argparse
import json
import pathlib
import time

from dream_hackathon.right_arm_controller import atomic_json, validate_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8692")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("configs/right_arm_start_poses.json"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--rate", type=float, default=0.3)
    args = parser.parse_args()
    from dream_hackathon.dream_caller import ArmEndpoint

    state = ArmEndpoint(args.server).state()
    if state.get("error") or state.get("stalled") or state.get("moving") is not False:
        raise ValueError("Capture only a stationary right arm without a reported error")
    stamp = state.get("stamp")
    if type(stamp) not in (int, float) or not 0 <= time.time() - stamp <= 1:
        raise ValueError("Right arm measurements must be fresh (within one second)")
    pose = {
        "q": state.get("q"),
        "waypoints": [],
        "rate_rad_s": args.rate,
        "tolerance_rad": 0.02,
        "grip": "preserve",
        "provenance": {
            "method": "Captured measured right arm state before Go",
            "server": args.server,
            "observed_at": stamp,
            "physical_return_path_validated": False,
        },
    }
    validate_pose(pose)
    poses = json.loads(args.output.read_text()) if args.output.exists() else {}
    poses[args.name] = pose
    atomic_json(args.output, poses)
    print(f"Saved measured pose {args.name} to {args.output}; no motion commands sent.")


if __name__ == "__main__":
    main()
