"""The top block of a tower past the fingers-down approach height (Nov 8 03:00: the sixth block on a 204 mm tower, its
approach point 260 mm against the 238 mm straight-down reach), with the hand tilted THETA deg about its level jaw axis,
the block held level all the way (the sim's stack9 helper trick, here straight from the table): a tilted pick on the
table, straight up, level over the tower top with the held block's bottom 2 cm clear, straight down to a set-down,
open, a level slide out along the pad faces (no straight-up exit: the reach ends there), the look pose.

The tilted hand sags more than the model: SAG_MM (measured 9 mm at 245 mm, 31.7 cm out, 20 deg) is added to every
commanded height near the top. Every leg planned before any motion (TowerSafety, opstack's extra checks, the runner's
sweep); the motion rule on every leg but the slide (the night_slide_out exception). Recorded as a stack episode, the
overhead knock check on it, a look after.

    FT_ROOT=night_nov8 python -m calib.night_top BLOCK SUPPORT[@X_CM,Y_CM,TOP_MM] [THETA_DEG] [SAG_MM] [--dry]
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from hillclimb.runner.calib import data_loop as D
from hillclimb.runner.calib import opstack as O
from hillclimb.runner.calib import tall_tower as T
from hillclimb.runner.calib.night_reach import TiltPlan, frame_tilt
from hillclimb.runner.calib.night_slide_out import SlidePlan


def plan_top(t, bid, s_b, top, theta, sag, quiet=False):
    """The tilted top place of `bid` (from where it lies) onto a support at s_b's x, y with its top at `top` (m): the
    first tilt azimuth (radial to the tower first, then +-15 ... +-60 deg) whose pick, carry and slide-out all plan.
    Returns (dpsi, pick, carry, out, gh, z_set, tx, ty); SystemExit when none plans. Shared by main() and the
    stack-order oracle (runs/station/order/oracle.py)."""
    op = t.op
    b = t.belief[bid]
    cx, cy = b["x"] - O.PICK_DOWN[0], b["y"] - O.PICK_DOWN[1]
    z_grip = O.GRIP_TIP - O.PICK_DOWN[2]
    tx, ty = s_b["x"], s_b["y"]
    gh = 0.015  # fingertips over the held block's bottom
    z_set = top + gh + 0.002 + sag
    z_carry = top + 0.02 + gh + 0.01 + sag
    best = None
    az_t = math.atan2(ty, tx)
    for dpsi in sorted(range(-60, 61, 15), key=abs):  # the tilt's azimuth: radial to the tower first
        psi = az_t + math.radians(dpsi)
        R = frame_tilt(theta, psi)
        try:
            pre = O.Plan(op, t.start_q(), 0.08)
            pre.R = O.frame_down(psi + math.pi / 2)
            # short of it by 6 cm or more: the route there crosses at 15-19 cm, where fingers down reach about 38 cm
            back_by = max(0.06, math.hypot(cx, cy) - 0.38)  # (Nov 8 11:47: blue at 49 cm refused at every azimuth)
            pre.to([cx - back_by * math.cos(psi), cy - back_by * math.sin(psi), z_grip + 0.08], math.degrees(psi) + 90, D.V_HIGH, "toward", keep=True)
            pre.check(skip=bid)
            t.sweep(pre, exempt={bid})
            pick = TiltPlan(op, pre.nodes[-1][0], 0.08)
            pick.nodes = list(pre.nodes)
            p_pre = op.tip(pre.nodes[-1][0])[0]
            q_i = pre.nodes[-1][0]
            for i in range(1, 9):
                f = i / 8
                pick.R = frame_tilt(theta * f, psi)
                q_i = pick.solve_from(q_i, [p_pre[0] + (cx - p_pre[0]) * f, p_pre[1] + (cy - p_pre[1]) * f, p_pre[2]], 0.0, True)
                pick.nodes.append((q_i, 0.3, "tilt over it"))
            pick.R = R
            pick.to([cx, cy, z_grip + 0.03], 0.0, D.V_DOWN, "down", line=0.01, keep=True)
            pick.to([cx, cy, z_grip], 0.0, D.V_LAST, "last 3 cm", line=0.005, keep=True)
            pick.check(skip=bid)
            t.sweep(pick, exempt={bid})
            held = (bid, gh)
            carry = TiltPlan(op, pick.nodes[-1][0], 0.044, held=held)
            carry.R = R
            carry.to([cx, cy, z_carry], 0.0, 0.3, "lift", line=0.01, keep=True)
            carry.to([tx, ty, z_carry], 0.0, 0.3, "over the tower", line=0.01, keep=True)
            carry.to([tx, ty, z_set + 0.01], 0.0, 0.15, "down", line=0.003, keep=True)
            carry.to([tx, ty, z_set], 0.0, D.V_SET, "set on the top", line=0.002, keep=True)
            carry.check(skip=bid)
            side = R[:, 1].copy()
            side[2] = 0.0
            side /= np.linalg.norm(side)
            if np.dot(side[:2], [tx, ty]) > 0:
                side = -side
            out = SlidePlan(op, carry.nodes[-1][0], 0.08)
            out.R = R
            out.to([tx + 0.12 * side[0], ty + 0.12 * side[1], z_set], 0.0, 0.15, "slide out", line=0.005, keep=True)
            out.check(skip=bid)
            best = (dpsi, pick, carry, out)
            break
        except SystemExit as e:
            if not quiet:
                print(f"azimuth {dpsi:+d} deg off radial: {str(e)[:160]}", flush=True)
    if best is None:
        raise SystemExit("no azimuth plans")
    dpsi, pick, carry, out = best
    return dpsi, pick, carry, out, gh, z_set, tx, ty


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bid, sup_arg = args[0], args[1]
    sup, _, at = sup_arg.partition("@")  # SUPPORT@X_CM,Y_CM,TOP_MM: where the tower's top is, when the records are behind
    theta = math.radians(float(args[2]) if len(args) > 2 else 20.0)
    sag = (float(args[3]) if len(args) > 3 else 9.0) / 1000
    dry = "--dry" in sys.argv
    D.take_token()
    t = T.Tall(T.TALL / ("top-" + time.strftime("%m%d-%H%M%S")))
    t.op.st["target"] = None
    op = t.op
    clean = True
    try:
        if not dry:
            D.rec("/start")
        if "--multi" in sys.argv:
            # every look (the far blocks are out of the look pose's view) and the standing tower from the records,
            # kept clear of by every move (Nov 8 11:42)
            extra = [tuple(float(v) for v in p_.split(",")) for p_ in os.environ.get("EXTRA_LOOKS", "").split(";") if p_.strip()]
            t.multi_look(list(T.LOOKS) + [T.LOOK_TIP] + extra)  # not LOOKS_OLD: two of them sit over a tower at (27, 0)
            # the loose block's sighting renamed before the records' tower goes in: the same id (yellow#1) in both made
            # the tower's entry overwrite the sighting (Nov 8 11:49)
            if "@" in bid:
                c0, xy0 = bid.split("@")
                p0 = [float(v) / 100 for v in xy0.split(",")]
                near0 = [k for k in t.belief if T.colour_of(k) == c0 and math.hypot(t.belief[k]["x"] - p0[0], t.belief[k]["y"] - p0[1]) < 0.04]
                if near0:
                    k0 = min(near0, key=lambda k: math.hypot(t.belief[k]["x"] - p0[0], t.belief[k]["y"] - p0[1]))
                    t.belief[f"{c0}#77"] = t.belief.pop(k0)
            T.do_scatter_setup(t)
        else:
            t.look_all("look-top", move=False, vet=False)
        # the tower from the build's records (the look sees its top from the side); the loose block from the look
        import json
        if at:
            s_b = {"x": 0.0, "y": 0.0, "yaw": 0.0, "z0": 0.0}
        else:
            prev = sorted(T.TALL.glob("build-1108-*"), key=lambda d: d.stat().st_mtime)[-1]
            last = json.loads((prev / "op" / "op.json").read_text())["blocks"]
            s_b = dict(last[sup])
        top = s_b["z0"] + T.H  # the support's top (a flat small block)
        if at:
            v = [float(u) for u in at.split(",")]
            s_b.update(x=v[0] / 100, y=v[1] / 100, z0=v[2] / 1000 - T.H)
            top = v[2] / 1000
        if "@" in bid:  # COLOUR@X_CM,Y_CM: that colour's block in the look nearest there (ids are per look)
            c, xy = bid.split("@")
            px, py = (float(v) / 100 for v in xy.split(","))
            cands = [k for k in t.belief if T.colour_of(k) == c and math.hypot(t.belief[k]["x"] - px, t.belief[k]["y"] - py) < 0.04]
            if not cands and os.environ.get("SURVEY_YAW") is not None:
                # not seen now (the looks that would see it are refused by the keep-out, or the tower hides it): the
                # first survey's fit, unchanged since (Nov 8 12:02; Isaiah: look once at the start, then trust it)
                t.belief[f"{c}#88"] = {"x": px, "y": py, "yaw": float(os.environ["SURVEY_YAW"]), "level": 0, "z0": T.TABLE_Z,
                                       "lying": False, "iou": 0.9, "edge": False, "t_seen": time.time()}
                t.put_blocks()
                cands = [f"{c}#88"]
                D.status(f"top: {c} not in this look; the survey's fit at ({px * 100:.1f}, {py * 100:.1f}) yaw {os.environ['SURVEY_YAW']} rad")
            if not cands:
                raise SystemExit(f"no {c} block within 4 cm of ({px * 100:.0f}, {py * 100:.0f}) in the look ({sorted(t.belief)})")
            bid = min(cands, key=lambda k: math.hypot(t.belief[k]["x"] - px, t.belief[k]["y"] - py))
            D.status(f"top: {c} near ({px * 100:.0f}, {py * 100:.0f}) is {bid} at ({t.belief[bid]['x'] * 100:.1f}, {t.belief[bid]['y'] * 100:.1f})")
        dpsi, pick, carry, out, gh, z_set, tx, ty = plan_top(t, bid, s_b, top, theta, sag)
        b = t.belief[bid]
        D.status(f"top: {bid} onto {sup} (top {(top - T.TABLE_Z) * 1000:.0f} mm) at {math.degrees(theta):.0f} deg tilt, azimuth {dpsi:+d} deg "
                 f"off radial: pick {len(pick.nodes)}, carry {len(carry.nodes)}, slide {len(out.nodes)} nodes; set at {z_set * 1000:.0f} mm "
                 f"(sag {sag * 1000:.0f} mm added)")
        if dry:
            return
        t.oh_before = t.overhead_now()
        T0 = D.now()
        src = dict(b)
        t.go(pick, 0.8, "top pick", "down")
        t.grip_to(O.arm_state(), D.GRIP_CLOSE, wait_close=True)
        s = O.arm_state()
        if s["grip"] < D.GRIP_HELD:
            t.grip_to(s, 0.8)
            raise D.CycleFail(f"{bid}: the tilted grasp closed empty (grip {s['grip']:.3f})")
        op.held = (bid, gh)
        from hillclimb.runner.calib import keepout as KO
        KO.ACTIVE[:] = [(tx, ty)]  # the hand may enter this tower's cylinder (setting the block on it); every other link not
        t.go(carry, D.GRIP_CLOSE, "top carry", "down")
        t.release(O.arm_state())
        op.held = None
        tip_rel = op.tip(np.array(O.arm_state()["q"]) + op.dq)[0]
        KO.add_place(float(tip_rel[0]), float(tip_rel[1]), top, T.H, T.TABLE_Z, note=f"{bid} tilted top place")
        t.go(out, 0.8, "slide out", "level")
        KO.ACTIVE.clear()
        t.belief = {}
        t.to_pose(O.LOOK_TIP, "look")  # checked against the keep-out cylinders in stream, belief or not
        T2 = D.now()
        img, q, grip, stem = op.snap("tower-top")
        t.belief = {sup: dict(s_b, level=4, iou=0.9, edge=False, lying=False), bid: dict(src)}
        t.episode(T0, T2, skill="task", prompt="", success=True, colour=bid, start_pose=src,
                  target={"x": tx, "y": ty, "yaw": src["yaw"], "level": 5, "support": sup}, end_pose=None,
                  error={}, cycle=0, block=6, recovery=False, phase="top-tilted", grasp={"tilt_deg": round(math.degrees(theta), 1)})
        t.belief_overlay(f"after {bid} on top")
        D.status(f"top: {bid} set on {sup} with a {math.degrees(theta):.0f} deg tilted hand; photos {op.run / (stem + '-wrist.jpg')} and {op.run / (stem + '-overhead.jpg')}")
    except BaseException as e:  # noqa: BLE001
        D.status(f"top {bid} stopped: {type(e).__name__}: {e}")
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
