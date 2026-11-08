"""Backend test against calib/station_sim (truth from the sim's /truth; the backend itself never reads it).

    RUNNER_SIM_PORT=9111 TALL_KNOWN_BIG='{"green": false, "orange": false}' \\
        python -m hillclimb.runner.sim_test --levels 8 [--gate-file /tmp/gate] [--fail-test]

Checks: prepare() moves no block; place_next() grows the tower (by truth) until it stops; a failed pick
(--fail-test: the planned block is moved away by the sim before the pick) returns ok False without raising.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request

from hillclimb.runner.backend import RunnerBackend

PORT = int(os.environ["RUNNER_SIM_PORT"])
BASE = f"http://127.0.0.1:{PORT}"


def truth():
    return json.loads(urllib.request.urlopen(BASE + "/truth", timeout=10).read())


def tallest(tr):
    blocks = {k: v for k, v in tr.items() if isinstance(v, dict) and "stack_level" in v}
    return 1 + max((v.get("stack_level") or 0) for v in blocks.values()) if blocks else 0


def moved(a, b):
    return {k: round(1000 * math.hypot(a[k]["x"] - b[k]["x"], a[k]["y"] - b[k]["y"]), 1)
            for k in a if k in b and isinstance(a[k], dict) and "x" in a[k]}


def main():
    levels = int(sys.argv[sys.argv.index("--levels") + 1]) if "--levels" in sys.argv else 8
    out = {"port": PORT}
    be = RunnerBackend()
    tr0 = truth()
    t0 = time.time()
    out["prepare"] = {k: v for k, v in be.prepare().items() if k != "tb"}
    tr1 = truth()
    mv = moved(tr0, tr1)
    out["prepare_s"] = round(time.time() - t0, 1)
    out["prepare_max_block_move_mm"] = max(mv.values()) if mv else 0.0
    print("PREPARE", json.dumps(out["prepare"], default=str)[:600], "max block move mm", out["prepare_max_block_move_mm"], flush=True)
    if "--fail-test" in sys.argv and be.moves:
        nxt = next((m for m in be.moves if m[0] == "stack"), None)
        if nxt:  # the planned block vanishes from where the runner saw it (moved 12 cm by the sim) before its pick
            b = tr1[nxt[1]] if nxt[1] in tr1 else None
            if b:
                body = {"blocks": {nxt[1]: {"x": b["x"] - 0.12, "y": b["y"], "yaw": b.get("yaw", 0.0), "level": 0}}}
                req = urllib.request.Request(BASE + "/reset", json.dumps(body).encode(), {"Content-Type": "application/json"})
                try:
                    urllib.request.urlopen(req, timeout=30).read()
                    out["fail_test_moved"] = nxt[1]
                except Exception as e:  # noqa: BLE001
                    out["fail_test_moved"] = f"reset refused: {e}"
    steps = []
    for i in range(levels):
        t1 = time.time()
        r = be.place_next(deadline_s=240)
        h = tallest(truth())
        steps.append({"i": i, "ok": r.get("ok"), "placed": r.get("placed_id"), "why": (r.get("why") or "")[:200],
                      "runner_tower": r.get("tower"), "truth_tallest": h, "s": round(time.time() - t1, 1)})
        print("STEP", json.dumps(steps[-1]), flush=True)
        if not r.get("ok") and "--fail-test" not in sys.argv:
            break
        if not r.get("ok") and len(steps) >= 3:
            break
    out["steps"] = steps
    out["final_truth_tallest"] = tallest(truth())
    be.retreat()
    be.close(clean=True)
    print("RESULT", json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
