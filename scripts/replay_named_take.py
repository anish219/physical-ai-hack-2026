"""Inspect a named edited replay; --execute opts into its capped-start playback."""

import argparse
import json
import pathlib
import time
import urllib.request

import numpy as np

from dream_hackathon.star_replay import Replay, Take


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=pathlib.Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.plan.parent / plan["targets_file"]).read_text().splitlines()]
    print(
        json.dumps(
            {
                "name": plan["name"],
                "duration_s": rows[-1]["t"],
                "targets": len(rows),
                "mode": "execute" if args.execute else "inspect only",
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    with urllib.request.urlopen(plan["target_url"].removesuffix("/target") + "/state", timeout=10) as response:
        current = json.load(response)
    take = Take(
        -1,
        np.array([row["t"] for row in rows]),
        np.array([row["q"] for row in rows]),
        np.array([row["grip"] for row in rows]),
    )
    replay = Replay(
        take, current["q"], current["grip"], plan["initial_approach_max_joint_speed_rad_s"], time.monotonic()
    )
    while True:
        tick = time.monotonic()
        q, grip, finished = replay.step(tick)
        request = urllib.request.Request(
            plan["target_url"],
            data=json.dumps({"q": q.tolist(), "grip": grip}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        if finished:
            return
        time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))


if __name__ == "__main__":
    main()
