"""Keep-out cylinders around standing towers (Isaiah, Nov 8 11:55: "a big circular cylinder we know where it is in 3D
space based on where the arm placed the first block, so it always avoids it in paths").

Twice on Nov 8 (11:14 and 11:51) a look move swept a 6-high tower over: the planner's idea of the tower came from the
cameras and the records, and both missed blocks. Here the tower is known from the arm itself: every place onto a block
(not onto the table) records the fingertips' position at the release, from the measured joints, and raises the top.
Every streamed motion in every process (opstack.stream: picks, places, looks, fallbacks, dry-run looks) is checked
against every cylinder with every arm link (the link boxes of the runner's model, after
tilt_helper.link_check), not the fingertips only, at joint steps of at most 1.5 deg.

The cylinder: centre where the base went, radius = half the block's footprint diagonal (35.6 mm for a 50.3 mm cube,
any rotation) + the farthest any later level was set off that centre + LINK_MARGIN (20 mm) for the links (HAND_MARGIN
10 mm for the jaws, pads and palm); height from the table to the top + the same margin. The hand may enter only the
tower it is working on in this block (setting a block on its top, or picking its top block): then the hand's points
are not checked against that tower, every other link still is.

State: runs/station/tall/keepout.json, read by every process at each check. Towers older than world_override.json's
"since" (a person reset the table) are ignored.

    python -m calib.keepout show | clear
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
_RUNS_ROOT = pathlib.Path(__import__("os").environ.get("HILLCLIMB_RUNS", str(pathlib.Path.home() / "hillclimb_runs")))
FILE = _RUNS_ROOT / "station" / "tall" / "keepout.json"  # tall_tower.use_sim points these at the sim's own folder
OVERRIDE = _RUNS_ROOT / "station" / "tall" / "world_override.json"
FACE = 0.0503
R_BLOCK = FACE * math.sqrt(2) / 2  # 35.6 mm: the cube's corner, at any yaw
LINK_MARGIN, HAND_MARGIN = 0.020, 0.010
SAME_TOWER = 0.04  # a place within 4 cm of a tower's centre is on that tower
JAW_ADR = (6, 7)  # the jaw slides in the runner's model (0 to 50 mm each; pads 6.7 mm apart at 0)
STEP = math.radians(1.5)
ESCAPE_SLACK = 0.001  # an escaping path may let a point's clearance drop by at most this per step (m)

# the tower(s) the current block may put the hand into (set by data_loop.Loop.block for the length of one block)
ACTIVE: list[tuple[float, float]] = []
# towers whose MARGIN a pick may enter (the picked block lies in that margin; Isaiah, Nov 8: picking a cube from the
# zone is the exception): every point keeps only the bare cylinder there. Set for the pick leg, cleared after the lift.
PICK_EXEMPT: list[tuple[float, float]] = []


def pick_zone(x, y):
    """The towers whose keep-out margin a block centred at (x, y) reaches into (its corner within r + the link margin)."""
    return [(t["x"], t["y"]) for t in load() if math.hypot(t["x"] - x, t["y"] - y) < t["r"] + LINK_MARGIN + R_BLOCK]


def _since():
    try:
        return json.loads(OVERRIDE.read_text()).get("since", "")
    except Exception:  # noqa: BLE001
        return ""


def load():
    try:
        towers = json.loads(FILE.read_text()).get("towers", [])
    except Exception:  # noqa: BLE001
        return []
    s = _since()
    return [t for t in towers if t.get("t", "") >= s]


def save(towers):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps({"towers": towers, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def _near(towers, x, y):
    c = [t for t in towers if math.hypot(t["x"] - x, t["y"] - y) < SAME_TOWER]
    return min(c, key=lambda t: math.hypot(t["x"] - x, t["y"] - y)) if c else None


def add_place(x, y, bottom, height, table_z, half_diag=R_BLOCK, note=""):
    """A block set at (x, y) (fingertips at the release, from the measured joints) with its bottom at `bottom`: onto a
    block (bottom over the table) it starts or raises a tower. Onto the table: nothing (a lone block is not a tower)."""
    if bottom < table_z + 0.005:
        return None
    towers = load()
    t = _near(towers, x, y)
    if t is None:
        t = {"x": round(x, 4), "y": round(y, 4), "r": round(half_diag, 4), "top": 0.0, "n": 1, "blocks": []}
        towers.append(t)
    t["top"] = round(max(t["top"], bottom + height), 4)
    t["r"] = round(max(t["r"], math.hypot(x - t["x"], y - t["y"]) + half_diag), 4)
    t["n"] = int(round((t["top"] - table_z) / height)) if height > 0 else t["n"] + 1
    t["blocks"].append({"x": round(x, 4), "y": round(y, 4), "bottom": round(bottom, 4), "note": note})
    t["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(towers)
    return t


def remove_pick(x, y, bottom, table_z):
    """A block picked from (x, y) with its bottom at `bottom`: the tower's top drops to it; its base (on the table): gone."""
    towers = load()
    t = _near(towers, x, y)
    if t is None:
        return None
    if bottom < table_z + 0.005:
        towers.remove(t)
    else:
        t["top"] = round(min(t["top"], bottom), 4)
        t["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(towers)
    return t


def link_points(kin, q, opening=0.095):
    """Points on every arm link's oriented box (corners, edge and face middles, centre) of the runner's model:
    (names, points (n, 3)). After tilt_helper.link_points."""
    import mujoco
    m, d = kin.m, kin.d
    d.qpos[kin.qadr] = q
    d.qpos[list(JAW_ADR)] = max(0.0, (opening - 0.0067) / 2)
    mujoco.mj_kinematics(m, d)
    if not hasattr(kin, "_ko_geoms"):
        kin._ko_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0
                         and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith(kin.side)]
        kin._ko_signs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
        kin._ko_names = []
        for g in kin._ko_geoms:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
            kin._ko_names += [nm] * len(kin._ko_signs)
        kin._ko_hand = np.array([("pad" in k or "palm" in k or "jaw" in k or "finger" in k) for k in kin._ko_names])
    pts = []
    for g in kin._ko_geoms:
        R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        a = m.geom_aabb[g]
        pts.append(c + (a[:3] + kin._ko_signs * a[3:]) @ R.T)
    return kin._ko_names, np.vstack(pts)


def check(op, nodes, tag, opening=0.095, active=None, hand=True):
    """Every link along the path kept out of every tower's cylinder (grown by the margins). The hand's points are not
    checked against a tower in `active` (default ACTIVE: the tower this block works on). Raises SystemExit on a hit;
    returns the smallest clearance (mm) or None when there is no tower. hand=False (plan time): the hand is not checked
    at all, only the other links, because a plan's own target tower is not marked active until its block starts."""
    towers = load()
    if not towers or len(nodes) < 1:
        return None
    act = ACTIVE if active is None else active
    hand_on = hand
    worst = None
    seq = [nodes[0]] + list(nodes) if len(nodes) == 1 else list(nodes)
    # escape: a path that STARTS inside a tower's margin (but outside the bare cylinder; after a stop mid-move, say)
    # may move as long as no point's clearance to that tower drops by more than ESCAPE_SLACK per step, until every
    # point is outside the margin again; from then on the normal rule. A path that starts outside is unchanged.
    escaping: dict[int, np.ndarray] = {}
    first = True
    for (q0, _, _), (q1, _, t) in zip(seq[:-1], seq[1:]):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / STEP)) + 1)
        for f in np.linspace(0, 1, n):
            names, P = link_points(op.kin, q0 + (q1 - q0) * f, opening)
            hand = op.kin._ko_hand
            for tw in towers:
                exempt = any(math.hypot(tw["x"] - ax, tw["y"] - ay) < SAME_TOWER for ax, ay in act)
                pick_ex = any(math.hypot(tw["x"] - px, tw["y"] - py) < 1e-3 for px, py in PICK_EXEMPT)
                mg = np.where(hand, HAND_MARGIN, LINK_MARGIN) * (0.0 if pick_ex else 1.0)  # the pick zone: bare cylinder only
                d = np.hypot(P[:, 0] - tw["x"], P[:, 1] - tw["y"]) - tw["r"]
                clear = np.maximum(d - mg, P[:, 2] - (tw["top"] + mg))
                if exempt or not hand_on:
                    clear = np.where(hand, np.inf, clear)
                k = id(tw)
                if first and float(np.min(clear)) < 0:
                    bare = np.maximum(d, P[:, 2] - tw["top"])
                    if exempt or not hand_on:
                        bare = np.where(hand, np.inf, bare)
                    if float(np.min(bare)) >= 0:  # inside the margin only: an escape may start here
                        escaping[k] = clear.copy()
                if k in escaping:
                    prev = escaping[k]
                    bad = (clear < 0) & (clear < prev - ESCAPE_SLACK)
                    if np.any(bad):
                        j = int(np.argmax(bad))
                        raise SystemExit(f"keep-out refused {tag}/{t}: escaping, {names[j]} moves {(prev[j] - clear[j]) * 1000:.1f} mm "
                                         f"deeper into the margin of the tower at ({tw['x'] * 100:.1f}, {tw['y'] * 100:.1f}) cm")
                    if float(np.min(clear)) >= 0:
                        del escaping[k]  # out of the margin: the normal rule from here on
                    else:
                        escaping[k] = clear.copy()
                        continue
                i = int(np.argmin(clear))
                if worst is None or clear[i] < worst:
                    worst = clear[i]
                if clear[i] < 0:
                    raise SystemExit(f"keep-out refused {tag}/{t}: {names[i]} {-clear[i] * 1000:.0f} mm inside the tower cylinder at "
                                     f"({tw['x'] * 100:.1f}, {tw['y'] * 100:.1f}) cm, r {tw['r'] * 1000:.0f} mm, top {tw['top'] * 1000:.0f} mm")
            first = False
    return None if worst is None else round(float(worst) * 1000, 1)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "clear":
        save([])
        print("keep-out cleared")
    else:
        print(json.dumps(load(), indent=1))


if __name__ == "__main__":
    main()
