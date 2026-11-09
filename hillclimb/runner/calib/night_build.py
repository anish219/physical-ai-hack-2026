"""One tower in exactly the order given, bottom first (Isaiah, Nov 8 02:45: big orange, big green, purple, yellow, blue,
small orange). Blocks named by colour and, where two share a colour, size: orange:big, green, purple, yellow, blue,
orange:small. A block sitting on another comes off first (top first, to a free spot). The tower spot: the base where it
lies if every level plans there, else the first spot in the high-reach band where every level plans, strictly in this
order (each pick, place, close look and check photo planned before anything moves). Then each block with the night
loop's own move (re-look, support look, standing check, re-place, knock check), recorded as stack episodes.

    FT_ROOT=night_nov8 python -m calib.night_build orange:big green purple yellow blue orange:small [--dry]
"""
from __future__ import annotations

import collections
import math
import sys
import time

from hillclimb.runner.calib import data_loop as D
from hillclimb.runner.calib import opstack as O
from hillclimb.runner.calib import tall_tower as T


def drop_ghosts(t):
    """One look saw a raised block at table level (its support hidden), the other at its level: the table-level one
    within 3 cm of a raised block of its colour is that block (02:45: purple#1 at 31.4, 6.4 next to purple#2 one up)."""
    for k in [k for k, b in t.belief.items() if T.bot(b) < T.TABLE_Z + 0.005]:
        b = t.belief[k]
        if any(o is not b and T.colour_of(o_k) == T.colour_of(k) and T.bot(o) > T.TABLE_Z + 0.005
               and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.03 for o_k, o in t.belief.items()):
            D.status(f"build: {k} at ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) is the raised block of its colour seen at table level: dropped")
            del t.belief[k]
    t.put_blocks()


def resolve(t, names):
    ids, used = [], set()
    for n in names:
        colour, _, size = n.partition(":")
        cands = [k for k, b in t.belief.items() if T.colour_of(k) == colour and k not in used
                 and (not size or D.size_word(b) == size)]
        if not cands:
            raise SystemExit(f"no {n} block in the look: {sorted(t.belief)}")
        k = max(cands, key=lambda k: t.belief[k].get("iou", 0))
        ids.append(k)
        used.add(k)
    return ids


def plan_strict(t, spot, order):
    """Every level in this order at spot: the number of levels that plan, and the first refusal."""
    saved = {k: dict(v) for k, v in t.belief.items()}
    q = t.q_look if t.q_look is not None else t.start_q()
    n, why = 0, None
    try:
        for lvl, bid in enumerate(order):
            sup = order[lvl - 1] if lvl else None
            at = (t.belief[sup]["x"], t.belief[sup]["y"], t.belief[sup]["yaw"]) if sup else spot
            if lvl == 0 and math.hypot(t.belief[bid]["x"] - spot[0], t.belief[bid]["y"] - spot[1]) < 0.005:
                n = 1
                continue  # the base stays where it lies
            try:
                t.plan_level(bid, lvl, at, sup, q)
            except SystemExit as e:
                why = f"level {lvl} ({bid}): {str(e)[:200]}"
                break
            st = T.top_of(t.belief[sup]) if sup else T.TABLE_Z
            t.belief[bid] = dict(t.belief[bid], x=at[0], y=at[1], yaw=at[2], level=lvl, z0=st)
            t.put_blocks()
            n += 1
    finally:
        t.belief = saved
        t.put_blocks()
        t.op.st["target"] = None
    return n, why


def main():
    dry = "--dry" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    D.take_token()  # the dry run moves too: its two looks (no block is touched)
    t = T.Tall(T.TALL / ("build-" + time.strftime("%m%d-%H%M%S")))
    t.op.st["target"] = None
    t.no_vet = True
    clean = True
    try:
        if not dry:
            D.rec("/start")
        t.multi_look()
        drop_ghosts(t)
        # blocks sitting on others come off first (top first)
        raised = sorted([k for k in t.belief if T.bot(t.belief[k]) > T.TABLE_Z + 0.005], key=lambda k: -T.bot(t.belief[k]))
        if raised and dry:
            D.status(f"build: {raised} sit on other blocks: they come off first (not in a dry run); nothing more to plan")
            return
        for k in raised:
            px, py, pyaw = t.pick_spot(k, [0])
            t.block(k, (px, py, pyaw, 0, None), 1, 0, "unstack")
        if raised:
            t.multi_look()  # the blocks they hid
            drop_ghosts(t)
        order = resolve(t, names)
        D.status(f"build: order {order}: " + ", ".join(f"{k} {D.size_word(t.belief[k])} {'side' if t.belief[k].get('lying') else 'flat'} "
                                                       f"L{t.belief[k]['level']} ({t.belief[k]['x'] * 100:.1f}, {t.belief[k]['y'] * 100:.1f})"
                                                       for k in order))
        # the spot: the base where it lies, else the high-reach band
        base = t.belief[order[0]]
        cands = [(base["x"], base["y"], base["yaw"])]
        area = D.AREA
        D.AREA = ((0.18, 0.30), (-0.10, 0.15))
        t.why = collections.Counter()
        gen = t.spot(order[0], list(range(len(order))), avoid=[], gap=0.075)
        best = None
        t_end = time.time() + 60
        for i in range(80):
            spot = cands[i] if i < len(cands) else next(gen, None)
            if spot is None or time.time() > t_end:
                break
            n, why = plan_strict(t, spot, order)
            if best is None or n > best[1]:
                best = (spot, n, why)
            if n == len(order):
                break
        D.AREA = area
        spot, n, why = best
        D.status(f"build: {n} of {len(order)} levels plan at ({spot[0] * 100:.1f}, {spot[1] * 100:.1f}){'' if n == len(order) else '; ' + str(why)}")
        if dry or n < 2:
            return
        order = order[:n]
        k = 0
        if math.hypot(base["x"] - spot[0], base["y"] - spot[1]) >= 0.005:
            k += 1
            t.block(order[0], (spot[0], spot[1], spot[2], 0, None), 1, k, "stack")
        for lvl in range(1, len(order)):
            sup = t.belief[order[lvl - 1]]
            k += 1
            t.block(order[lvl], (sup["x"], sup["y"], sup["yaw"], lvl, order[lvl - 1]), 1, k, "stack")
            D.status(f"build: {lvl + 1} high ({' / '.join(order[:lvl + 1])}), top at {(T.top_of(t.belief[order[lvl]]) - T.TABLE_Z) * 1000:.0f} mm")
        img, q, grip, stem = t.op.snap("tower")
        D.status(f"build: done, {len(order)} high; photos {t.op.run / (stem + '-wrist.jpg')} and {t.op.run / (stem + '-overhead.jpg')}")
    except BaseException as e:  # noqa: BLE001
        D.status(f"build stopped: {type(e).__name__}: {e}")
        if not dry:
            st = O.arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
    finally:
        if dry:
            D.release_token()
        else:
            T.finish(t, clean)


if __name__ == "__main__":
    main()
