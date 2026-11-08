"""One full round against a station sim: the real left controller and the real scripted runner, with a fake gate.

    LEFT_URL=http://127.0.0.1:9141 ~/dream-env/bin/python -m hillclimb.sim.e2e_scripted --gate-after 45 --goal 6

Prints the state every 5 s and the sim's truth tower at the end. Refuses real-arm URLs.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.request

from hillclimb.common.arm import ArmClient
from hillclimb.common.config import CFG
from hillclimb.left.controller import LeftController
from hillclimb.sim.fakes import FakeGate


def truth_height(url: str) -> int:
    try:
        t = json.loads(urllib.request.urlopen(url + "/truth", timeout=5).read())
        return 1 + max((b.get("stack_level") or 0) for b in t.values() if isinstance(b, dict) and "stack_level" in b)
    except Exception:  # noqa: BLE001
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-after", type=float, default=45)
    ap.add_argument("--goal", type=int, default=6)
    ap.add_argument("--round-s", type=float, default=600)
    a = ap.parse_args()
    url = CFG["LEFT_URL"]
    if any(p in url for p in (":8691", ":8692")):
        raise SystemExit("refusing: LEFT_URL is a real arm")
    tmp = tempfile.mkdtemp(prefix="hc_e2e_")
    arm = ArmClient(url)
    ctl = LeftController(gate=FakeGate(open_after_s=a.gate_after), arm=arm, overhead_arm=arm,
                         state_file=os.path.join(tmp, "state.json"), log_dir=tmp)
    ok, why = ctl.arm_round({"backend": "scripted", "mode": "practice", "round_s": a.round_s, "max_goal": a.goal,
                             "crunch_goal": min(5, a.goal), "crunch_at_s": 60, "freeze_s": 20})
    print("arm", ok, why, flush=True)
    ctl.start()
    ctl.go()
    t0 = time.time()
    while ctl.state not in ("END",) and time.time() - t0 < a.round_s + 30:
        s = ctl.snapshot()
        print(f"{time.time() - t0:6.0f}s {s.get('state'):13s} tower={len(ctl.tower)} truth={truth_height(url)} "
              f"last={json.dumps(s.get('last_step', {}))[:140]}", flush=True)
        if ctl.state == "REST" and len(ctl.tower) >= a.goal:
            break
        time.sleep(5)
    print("END state", ctl.state, "tower", len(ctl.tower), "truth", truth_height(url), "logs", tmp, flush=True)
    ctl.stop_round("test done")
    ctl.shutdown()


if __name__ == "__main__":
    main()
