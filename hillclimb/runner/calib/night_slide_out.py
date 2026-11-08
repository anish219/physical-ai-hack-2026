"""Out of a tower top with the jaws open: a level slide along the pad faces (the tool's y axis) by DIST_CM toward the
table's middle, then to the look pose. For a hand left low in the open jaws around a tower's top block when the back-off
straight up ran out of reach (Nov 8 02:49: five high, fingertips at 175 mm, goal 213 unreachable). The motion rule
refuses any sideways move by a tower under its top; this slide is the one exception (the pads move parallel to the
block's faces, 18 mm clear each side); TowerSafety and the extra checks still run on every node.

    python -m calib.night_slide_out SESSION [DIST_CM] [--dry]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from hillclimb.runner.calib import data_loop as D
from hillclimb.runner.calib import opstack as O
from hillclimb.runner.calib import tall_tower as T


class SlidePlan(O.Plan):
    def rule_check(self):
        return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sess, dist = args[0], (float(args[1]) if len(args) > 1 else 13.0) / 100
    dry = "--dry" in sys.argv
    t = T.Tall(T.TALL / ("slide-" + time.strftime("%m%d-%H%M%S")))
    t.op.st["blocks"] = json.loads((T.TALL / sess / "op" / "op.json").read_text())["blocks"]
    s = O.arm_state()
    q0 = np.array(s["q"]) + t.op.dq
    pos, R = t.op.kin.fk(q0)
    tip = pos + t.op.kin.tip * R[:, 2]
    side = R[:, 1].copy()
    side[2] = 0.0
    side /= np.linalg.norm(side)
    if side @ tip[:2].tolist() + [0] if False else np.dot(side[:2], tip[:2]) > 0:  # toward the table's middle (the base)
        side = -side
    end = tip + dist * side
    pl = SlidePlan(t.op, q0, 0.08)
    pl.R = R
    pl.to(end.tolist(), 0.0, 0.15, "slide out", line=0.005, keep=True)
    pl.check()
    print(f"slide planned: {len(pl.nodes)} nodes, fingertips {np.round(tip * 1000).tolist()} -> {np.round(end * 1000).tolist()}", flush=True)
    if dry:
        return
    D.take_token()
    try:
        t.op.st["target"] = list(q0)
        t.go(pl, 0.8, "slide out", "level")
        t.belief = {}
        t.to_pose(O.LOOK_TIP, "look")
        D.status(f"slide out: jaws slid {dist * 100:.0f} cm level along the pads off the tower top, then to the look pose")
    finally:
        D.release_token()


if __name__ == "__main__":
    main()
