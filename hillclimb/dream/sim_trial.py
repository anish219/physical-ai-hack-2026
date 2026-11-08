"""Sim checks for the Dream backend (station only; the sim must already serve on --sim, default port 9121).

    ~/dream-env/bin/python -m hillclimb.dream.sim_trial level 1|2 [--deadline 75] [--checkpoint ID]
    ~/dream-env/bin/python -m hillclimb.dream.sim_trial missed          # deadline 8 s: the policy cannot finish
    ~/dream-env/bin/python -m hillclimb.dream.sim_trial warm [--idle 120]

Builds a tidy tower at the tower spot through the sim's /reset (commanded start pose), gives the backend a scene
taken from the sim's truth (in the harness the left controller gives it from perception), runs one step and prints
the truth before and after: whether the tower grew by one and stands, and whether anything already standing moved.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
import urllib.request

from . import geometry as G
from .backend import DEFAULT_CHECKPOINT, DreamBackend

SPOT = (0.32, 0.12)  # clear of the start pose's hand
FILL = ["blue#1", "green#1", "orange#1", "blue#2"]
LOOSE = {"green#3": (0.37, -0.06), "orange#3": (0.24, -0.03), "blue#5": (0.41, 0.03)}


def _post(sim, path, body, t=30):
    rq = urllib.request.Request(sim + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(rq, timeout=t).read() or b"{}")


def _truth(sim):
    tr = json.loads(urllib.request.urlopen(sim + "/truth?all=1", timeout=10).read())
    return {k: v for k, v in tr["props"].items() if v["kind"] == "block"}


def _tower(b):
    col = [(k, v) for k, v in b.items() if v["stack_level"] is not None and abs(v["x"] - SPOT[0]) < 0.05
           and abs(v["y"] - SPOT[1]) < 0.05 and not v["held"]]
    return sorted(col, key=lambda kv: kv[1]["stack_level"])


def build(sim, height):
    bl = {k: {"x": SPOT[0], "y": SPOT[1], "yaw": 0.0, "level": i} for i, k in enumerate(FILL[:height])}
    for k, (x, y) in LOOSE.items():
        bl[k] = {"x": x, "y": y, "yaw": 0.25, "level": 0}
    _post(sim, "/reset", {"blocks": bl, "q": list(G.START_Q), "grip": G.START_GRIP})
    time.sleep(4)


def scene_from_truth(b):
    tw = _tower(b)
    top_id, top = tw[-1]
    colour = lambda k: k.split("#")[0]  # noqa: E731
    loose = [{"id": k, "colour": colour(k), "x": v["x"], "y": v["y"], "level": 0, "lying": not v["flat"]}
             for k, v in b.items() if v["on_table"] and v["stack_level"] == 0 and k in LOOSE
             and not any(o["on"] == k for o in b.values())]
    return {"tower": {"x": top["x"], "y": top["y"], "top_z": top["z"] + 0.0175, "height": len(tw),
                      "top": {"id": top_id, "colour": colour(top_id), "x": top["x"], "y": top["y"]}},
            "loose": loose, "tower_count": 1 if len(tw) >= 2 else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["level", "missed", "warm", "all"])
    ap.add_argument("n", nargs="?", type=int, default=1)
    ap.add_argument("--sim", default="http://127.0.0.1:9121")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--deadline", type=float, default=75.0)
    ap.add_argument("--idle", type=float, default=120.0)
    ap.add_argument("--warm-timeout", type=float, default=1200.0)
    a = ap.parse_args()
    out = pathlib.Path.home() / "hspike" / "R" / "out"
    out.mkdir(parents=True, exist_ok=True)
    be = DreamBackend(a.sim, a.checkpoint, log_dir=str(out))
    build(a.sim, 1)
    be.prepare({})
    t0 = time.time()
    ok = be.loop.wait_ready(a.warm_timeout)
    print(json.dumps({"ready": ok, "warm_s": round(time.time() - t0, 1), "status": be.status()}), flush=True)
    if not ok:
        be.loop.close()
        return
    if a.what == "all":
        steps = [("warm", 0)] + [("level", n) for n in (1, 1, 2, 2)] + [("missed", 2)]
    else:
        steps = [(a.what, a.n)]
    for what, n in steps:
        if what == "warm":
            c0, r0 = be.loop.status["calls"], be.loop.status["rows_sent"]
            time.sleep(a.idle)
            st = be.status()
            print("RESULT " + json.dumps({"what": "warm", "idle_s": a.idle, "warm_calls": st["calls"] - c0,
                                          "last_warm_ok": st["last_warm_ok"], "mode": st["mode"],
                                          "rows_sent_while_idle": st["rows_sent"] - r0}), flush=True)
            continue
        build(a.sim, n)
        b0 = _truth(a.sim)
        scene = scene_from_truth(b0)
        r0, f0 = be.loop.status["rows_sent"], be.loop.status["rows_refused"]
        res = be.place_next(scene, deadline_s=8.0 if what == "missed" else a.deadline)
        time.sleep(1.0)
        b1 = _truth(a.sim)
        h0, h1 = len(_tower(b0)), len(_tower(b1))
        stood = {k: round(math.hypot(b1[k]["x"] - b0[k]["x"], b1[k]["y"] - b0[k]["y"]) * 1000) for k, _ in _tower(b0)}
        standing = all(v["tilt_deg"] < 10 for _, v in _tower(b1))
        print("RESULT " + json.dumps({"what": what, "n": n, "result": {k: res[k] for k in ("ok", "placed_id", "prompt", "why")},
                                      "timing": res["timing"], "tower_before": h0, "tower_after": h1,
                                      "grew_by_one_and_stands": h1 == h0 + 1 and standing,
                                      "tower_blocks_moved_mm": stood, "rows_sent": be.loop.status["rows_sent"] - r0,
                                      "rows_refused": be.loop.status["rows_refused"] - f0}), flush=True)
    be.loop.close()


if __name__ == "__main__":
    main()
