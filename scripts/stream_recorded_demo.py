"""Inspect a prepared replay by default. --execute explicitly streams its targets.

Preparation and inspection make no network requests. Execution only sends the
recorded joint/gripper targets; it does not set power, inputs, hold, home or stop.
"""

import argparse
import json
import pathlib
import time
import urllib.request


def load(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    targets = [json.loads(line) for line in (path.parent / plan["targets_file"]).read_text().splitlines()]
    return plan, targets


def stream(plan, targets, *, speed=1.0):
    start = time.monotonic()
    for row in targets:
        remaining = start + row["t"] / speed - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        request = urllib.request.Request(
            plan["target_url"], data=json.dumps({"q": row["q"], "grip": row["grip"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=pathlib.Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    plan, targets = load(args.plan)
    print(json.dumps({"mode": "execute" if args.execute else "inspect only", "task": plan["task"],
                      "side": plan["side"], "targets": len(targets), "duration_s": targets[-1]["t"],
                      "speed": args.speed}, indent=2), flush=True)
    if args.execute:
        stream(plan, targets, speed=args.speed)


if __name__ == "__main__":
    main()
