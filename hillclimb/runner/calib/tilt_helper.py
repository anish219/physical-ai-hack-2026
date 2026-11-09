"""Levels past the fingers-down reach (8 to 10 on the flat 34.5 mm blocks): the sim builder's helper-stack method
(sim/stack9.py PitchStacker) through the runner's own planner. A loose block goes onto a one-block helper base,
fingers down (a level-1 place); then the hand tilts THETA deg about its level jaw axis over the helper's top block,
grips it level, lifts, carries it level over the tower top (held block's bottom 2 cm clear), sets it down, opens wide
and slides out along the pad faces (the night_slide_out exception), then goes to the look pose. Every leg is planned
before it moves: Plan.check (TowerSafety, opstack's extra checks, the motion rule) and the runner's sweep over the
other blocks; the tilted legs in one fixed frame (night_reach.TiltPlan); the sag model in the stream as always.

Sim test only as written: --truth fills the belief from the sim's /truth (no looks), so the motion and planning are
tested apart from perception.

    python -m calib.tilt_helper PORT TOWER_X_CM,TOWER_Y_CM STAGE_X_CM,STAGE_Y_CM [--theta 40] [--up 25] [--sag 0]
                                [--levels 3] [--out FILE]
"""
from __future__ import annotations

import json
import math
import sys
import time as _wall
import urllib.request

import numpy as np

from hillclimb.runner.calib import data_loop as D
from hillclimb.runner.calib import opstack as O
from hillclimb.runner.calib import tall_tower as T
from hillclimb.runner.calib.night_reach import TiltPlan, frame_tilt


class TiltSlide(TiltPlan):
    """The slide out of the tower top in the tilted frame (night_slide_out's exception to the motion rule), solved with
    the tilted frame's flange offset (SlidePlan's solve assumes fingers down: wrong past about 20 deg)."""

    def rule_check(self):
        return None

H, TABLE_Z = D.H, D.TABLE_Z


LINK_MARGIN, PAD_MARGIN = 0.015, 0.005  # arm links / pads clear of a tower column by this much (m)
COLUMN_HW = 0.025 + 0.005  # a small block's half face, plus 5 mm for a tower's lean and offsets (a square column)


OPEN_FULL = 0.100  # the jaws at grip 1.0 (the release ends there, _post_wide): the way out is checked at this opening
JAW_ADR = (6, 7)  # the two jaw slides in the runner's model (0 to 50 mm each; pads 6.7 mm apart at 0)


def link_points(kin, q, opening=0.095):
    """Points on every arm link's oriented box (mesh aabb in the geom frame: corners, edge and face middles, centre),
    from the runner's own kinematic model: (names, points) with points (n, 3)."""
    import itertools
    import mujoco
    m, d = kin.m, kin.d
    d.qpos[kin.qadr] = q
    d.qpos[list(JAW_ADR)] = max(0.0, (opening - 0.0067) / 2)  # the jaws at the plan's opening, not the model's closed default
    mujoco.mj_kinematics(m, d)
    if not hasattr(kin, "_link_geoms"):
        kin._link_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0
                           and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith(kin.side)]
        kin._signs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    names, pts = [], []
    for g in kin._link_geoms:
        R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        a = m.geom_aabb[g]
        P = c + (a[:3] + kin._signs * a[3:]) @ R.T
        pts.append(P)
        names += [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])] * len(P)
    return names, np.vstack(pts)


def link_check(op, nodes, cols, tag, margin=LINK_MARGIN, pad_margin=PAD_MARGIN, opening=0.095):
    """Every arm link along a planned path (joint steps of at most 1 deg) kept out of each tower column (x, y, top):
    a square column (COLUMN_HW, the top block's yaw) from the table to the top, grown by margin (pad_margin for the pads and palm).
    The runner's sweep checks only the fingertips; a forearm swept an 8-high tower over in the sim (Nov 8, r16)."""
    worst = None
    for (q0, _, _), (q1, _, t) in zip(nodes[:-1], nodes[1:]):
        n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / math.radians(1.0))) + 1)
        for f in np.linspace(0, 1, n):
            names, P = link_points(op.kin, q0 + (q1 - q0) * f, opening)
            hand = np.array([("pad" in k or "palm" in k or "jaw" in k) for k in names])
            for cx, cy, top, yaw in cols:
                mg = np.where(hand, 0.0 if pad_margin is None else pad_margin, margin)
                c, s_ = math.cos(yaw), math.sin(yaw)
                u, v = (P[:, 0] - cx) * c + (P[:, 1] - cy) * s_, -(P[:, 0] - cx) * s_ + (P[:, 1] - cy) * c
                d = np.maximum(np.abs(u), np.abs(v)) - COLUMN_HW  # Chebyshev: never more than the true distance
                clear = np.maximum(d - mg, P[:, 2] - (top + mg))  # outside the grown column if either is positive
                if pad_margin is None:  # the hand is setting a block on this top: only the other links are checked
                    clear = np.where(hand, np.inf, clear)
                i = int(np.argmin(clear))
                if worst is None or clear[i] < worst[0]:
                    worst = (clear[i], names[i], t)
                if clear[i] < 0:
                    raise SystemExit(f"link check refused {tag}/{t}: {names[i]} {-clear[i] * 1000:.0f} mm inside the tower column "
                                     f"at ({cx * 100:.0f}, {cy * 100:.0f}) cm, top {top * 1000:.0f} mm")
    return None if worst is None else {"min_clear_mm": round(worst[0] * 1000, 1), "at": f"{worst[1]} / {worst[2]}"}


def arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def truth(port):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/truth", timeout=10).read())


def truth_belief(port):
    out = {}
    for k, v in truth(port).items():
        if "bottom" not in v or not v.get("on_table", True):
            continue
        out[k] = {"x": v["x"], "y": v["y"], "yaw": v["yaw"], "z0": v["bottom"], "level": int(round((v["bottom"] - TABLE_Z) / H)),
                  "lying": not v.get("flat", True), "iou": 0.9, "edge": False, "t_seen": _wall.time()}
    return out


def column(belief, x, y, r=0.03):
    """Blocks within r of (x, y), bottom first."""
    return sorted([k for k, b in belief.items() if math.hypot(b["x"] - x, b["y"] - y) < r], key=lambda k: belief[k]["z0"])


def z_grip(b):
    return O.GRIP_TIP + (D.bot(b) - TABLE_Z) - O.PICK_DOWN[2]  # fingertips 15 mm over the block's bottom


def fd_move(t, bid, x, y, level, cols=()):
    """Fingers down: bid onto (x, y) at `level` (a one-block helper base under it at level 1), every leg checked."""
    op, b = t.op, t.belief[bid]
    cx, cy, zg = b["x"] - O.PICK_DOWN[0], b["y"] - O.PICK_DOWN[1], z_grip(b)
    yaw = math.degrees(b["yaw"])
    pick = O.Plan(op, t.start_q(), 0.08)
    pick.to([cx, cy, zg + 0.08], yaw, D.V_HIGH, "over")
    pick.to([cx, cy, zg + 0.03], yaw, D.V_DOWN, "down", line=0.01, keep=True)
    pick.to([cx, cy, zg], yaw, D.V_LAST, "last 3 cm", line=0.005, keep=True)
    pick.check(skip=bid)
    t.sweep(pick, exempt={bid})
    gh = 0.015
    z_set = TABLE_Z + level * H + gh + 0.002
    carry = O.Plan(op, pick.nodes[-1][0], 0.044, held=(bid, gh))
    carry.R = pick.R
    carry.to([cx, cy, max(0.13, z_set + 0.05)], yaw, 0.4, "lift", line=0.01, keep=True)
    carry.to([x, y, z_set + 0.03], yaw, 0.4, "over the helper")
    carry.to([x, y, z_set], yaw, D.V_SET, "set", line=0.003, keep=True)
    carry.check(skip=bid)
    t.sweep(carry, exempt={bid} | set(column(t.belief, x, y)), held_h=gh)
    up = O.Plan(op, carry.nodes[-1][0], 0.08)
    up.R = carry.R
    up.to([x, y, z_set + 0.06], yaw, 0.4, "up", line=0.01, keep=True)
    up.check(skip=bid)
    if cols:
        for pl, tag, o in ((pick, "helper pick", 0.08), (carry, "helper carry", 0.05), (up, "helper up", 0.08)):
            link_check(op, pl.nodes, cols, tag, opening=o)
    t.go(pick, 0.8, "helper pick", "down")
    t.grip_to(O.arm_state(), D.GRIP_CLOSE, wait_close=True)
    s = O.arm_state()
    if s["grip"] < D.GRIP_HELD:
        t.grip_to(s, 0.8)
        raise D.CycleFail(f"{bid}: helper pick closed empty (grip {s['grip']:.3f})")
    op.held = (bid, gh)
    t.go(carry, D.GRIP_CLOSE, "helper carry", "down")
    t.release(O.arm_state())
    op.held = None
    t.go(up, 0.8, "helper up", "up")


def tilt_place(t, bid, tower_top_z, tx, ty, theta, up, sag, tyaw=0.0):
    """bid (on the helper) onto the tower top at (tx, ty), the hand tilted theta; the legs planned first, then run.
    Targets are for the grip point (the IK site, 32.5 mm back along the fingers from the tips), as sim/stack9.py: the
    site `up` over the block's centre at the pick and over the held block's centre at the set-down. Aiming the
    fingertips at the centre instead (night_top) puts the pads 32.5 sin(theta) mm short of it: 21 mm at 40 deg, the
    block's edge."""
    op, b = t.op, t.belief[bid]
    L = op.kin.tip
    hc = H / 2 + up  # the site over the held block's bottom
    bx, by = b["x"] - O.PICK_DOWN[0], b["y"] - O.PICK_DOWN[1]
    z_site_pick = D.bot(b) + hc
    z_site_set = tower_top_z + 0.002 + hc + sag
    z_site_carry = max(tower_top_z + 0.02 + hc + sag, z_site_pick + 0.03)
    az_t = math.atan2(ty, tx)
    tried = []
    # the tilt's azimuth along one of the held block's face normals (pads flat on two faces, as sim/stack9.py), the
    # one nearest radial to the tower first; the square block lands on the tower top whatever its turn
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    psis = sorted((b["yaw"] + k * math.pi / 2 for k in range(4)), key=lambda a: abs(wrap(a - az_t)))
    for psi in psis:
        dpsi = round(math.degrees(wrap(psi - az_t)))
        R = frame_tilt(theta, psi)
        tipof = lambda x, y, z: [x + L * R[0, 2], y + L * R[1, 2], z + L * R[2, 2]]  # fingertips for a site at (x, y, z)
        try:
            p_pick = tipof(bx, by, z_site_pick)
            p_hi = tipof(bx, by, z_site_pick + 0.05)
            pre = O.Plan(op, t.start_q(), 0.08)
            pre.R = O.frame_down(psi + math.pi / 2)
            # level with the tilted start over the block, so the blend into the tilt never goes down beside a stack
            pre.to([bx - 0.06 * math.cos(psi), by - 0.06 * math.sin(psi), p_hi[2]], math.degrees(psi) + 90, D.V_HIGH, "toward", keep=True)
            pre.check(skip=bid)
            t.sweep(pre, exempt={bid})
            pick = TiltPlan(op, pre.nodes[-1][0], 0.095)
            pick.nodes = list(pre.nodes)
            p_pre = op.tip(pre.nodes[-1][0])[0]
            q_i = pre.nodes[-1][0]
            for i in range(1, 9):  # blend into the tilted frame while moving over the block
                f = i / 8
                pick.R = frame_tilt(theta * f, psi)
                q_i = pick.solve_from(q_i, [p_pre[k] + (p_hi[k] - p_pre[k]) * f for k in range(3)], 0.0, True)
                pick.nodes.append((q_i, 0.3, "tilt over it"))
            pick.R = R
            pick.to(tipof(bx, by, z_site_pick + 0.02), 0.0, D.V_DOWN, "down", line=0.01, keep=True)
            pick.to(p_pick, 0.0, D.V_LAST, "last 2 cm", line=0.004, keep=True)
            pick.check(skip=bid)
            t.sweep(pick, exempt={bid} | set(column(t.belief, b["x"], b["y"])))
            gh = p_pick[2] - D.bot(b)  # fingertips over the held block's bottom
            lift = TiltPlan(op, pick.nodes[-1][0], 0.044, held=(bid, gh))
            lift.R = R
            lift.to(tipof(bx, by, z_site_pick + 0.03), 0.0, 0.2, "lift 3 cm", line=0.005, keep=True)
            lift.check(skip=bid)
            carry = TiltPlan(op, lift.nodes[-1][0], 0.044, held=(bid, gh))
            carry.R = R
            carry.to(tipof(bx, by, z_site_carry), 0.0, 0.3, "lift", line=0.01, keep=True)
            carry.to(tipof(tx, ty, z_site_carry), 0.0, 0.3, "over the tower", line=0.01, keep=True)
            carry.to(tipof(tx, ty, z_site_set + 0.01), 0.0, 0.15, "down", line=0.003, keep=True)
            carry.to(tipof(tx, ty, z_site_set), 0.0, D.V_SET, "set on the top", line=0.002, keep=True)
            carry.check(skip=bid)
            cols = [(tx, ty, tower_top_z, tyaw)]
            lk = {"carry": link_check(op, carry.nodes, cols, "carry", pad_margin=None, opening=0.05)}  # the hand sets the block on the top: links other than the hand only
            # the way out (no block held, jaws wide): 1. back out along the fingers, 2. straight up until every link
            # clears the new top, 3. un-tilt away from the tower, level until 12 cm off its axis, then down to the
            # look pose; every leg by the planner's checks and the link check against the new top
            new_top = tower_top_z + H
            cols = [(tx, ty, new_top, tyaw)]
            ps = op.tip(carry.nodes[-1][0])[0]
            why_out = []
            for bo in (0.06, 0.045, 0.03):  # back out along the fingers as far as the reach allows near the top
                try:
                    out = TiltSlide(op, carry.nodes[-1][0], 0.095)  # backing out along the fingers is up and sideways: no motion rule
                    out.R = R
                    pb = ps - bo * R[:, 2]
                    out.to(list(pb), 0.0, 0.15, "back out", line=0.004, keep=True)
                    out.check(skip=bid)
                    lk["back out"] = link_check(op, out.nodes, cols, "back out", opening=OPEN_FULL)
                    lk["back_out_mm"] = round(bo * 1000)
                    break
                except SystemExit as e:
                    why_out.append(f"{bo * 100:.1f} cm: {str(e)[:90]}")
            else:
                raise SystemExit("back out: " + " | ".join(why_out))
            look = np.array(O.LOOK_TIP[:3])
            to_look = np.array([look[0] - tx, look[1] - ty])
            dirs = [to_look / np.linalg.norm(to_look)]
            for ang in (math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2):
                c_, s2 = math.cos(ang), math.sin(ang)
                dirs.append(np.array([dirs[0][0] * c_ - dirs[0][1] * s2, dirs[0][0] * s2 + dirs[0][1] * c_]))
            exits = []
            ways = [(dz, dist, away) for dz in (0.0, 0.02, 0.04) for dist in (0.12, 0.16, 0.20) for away in dirs]
            for dz, dist, away in [(None, None, None)]:  # (kept for the loop shape below)
                for dz, dist, away in ways:  # the smallest rise and distance whose way out plans and passes the link check
                    try:
                        rise = TiltPlan(op, out.nodes[-1][0], 0.095)
                        rise.R = R
                        if dz:
                            rise.to([pb[0], pb[1], pb[2] + dz], 0.0, 0.2, "rise", line=0.005, keep=True)
                        rise.check(skip=bid)
                        lr = link_check(op, rise.nodes, cols, "rise", opening=OPEN_FULL) if dz else None
                        pr = op.tip(rise.nodes[-1][0])[0]
                        pa = np.array([tx + dist * away[0], ty + dist * away[1], pr[2]])  # dist off the tower's axis
                        back = TiltSlide(op, rise.nodes[-1][0], 0.095)  # the motion rule's tower test is replaced by link_check below
                        q_i = rise.nodes[-1][0]
                        # the arm reaches that high only tilted: level and away with the tilt kept, then down with it
                        # kept to the look height, and only then un-tilt, moving over to the look pose
                        pd = np.array([pa[0], pa[1], look[2]])
                        legs = [(pa, 1.0, 1.0, 8), (pd, 1.0, 1.0, 8), (look, 1.0, 0.0, 10)]
                        p0 = pr.copy()
                        for p1, a0, a1, n in legs:
                            for i in range(1, n + 1):
                                f = i / n
                                back.R = frame_tilt(theta * (a0 + (a1 - a0) * f), psi)
                                q_i = back.solve_from(q_i, list(p0 + (p1 - p0) * f), 0.0, True)
                                back.nodes.append((q_i, 0.3, "away" if p1 is pa else "down" if p1 is pd else "untilt"))
                            p0 = np.array(p1, float)
                        back.check(skip=bid)
                        t.sweep(back, exempt=set())
                        lk["untilt"] = link_check(op, back.nodes, cols, "untilt", opening=OPEN_FULL)
                        lk["rise"], lk["exit"] = lr, {"rise_mm": round(dz * 1000), "away_cm": round(dist * 100), "away_dir": [round(float(v), 2) for v in away]}
                        break
                    except SystemExit as e:
                        exits.append(f"r{dz * 100:.0f} d{dist * 100:.0f} {np.round(away, 2).tolist()}: {str(e)[:70]}")
                else:
                    continue
                break
            else:
                raise SystemExit("no way out: " + " | ".join(exits[:45]))
            break
        except SystemExit as e:
            tried.append(f"{dpsi:+d}: {str(e)[:6000]}")
    else:
        raise SystemExit("no azimuth plans: " + " | ".join(tried[:4]))
    t0 = _wall.time()
    t.go(pick, 0.8, "tilt pick", "down")
    s0 = O.arm_state()  # diagnostics: where the fingertips are against where they were sent, at the grasp
    diag = {"tip_goal_mm": np.round(op.tip(np.array(s0["goal"]) + op.dq)[0] * 1000, 1).tolist(),
            "tip_measured_mm": np.round(op.tip(np.array(s0["q"]) + op.dq)[0] * 1000, 1).tolist(),
            "tip_planned_mm": np.round(op.tip(pick.nodes[-1][0])[0] * 1000, 1).tolist(), "block_belief_mm": [round(b["x"] * 1000, 1), round(b["y"] * 1000, 1)]}
    print("GRASP DIAG", json.dumps(diag), flush=True)
    t.grip_to(O.arm_state(), D.GRIP_CLOSE, wait_close=True)
    s = O.arm_state()
    if s["grip"] < D.GRIP_HELD:
        t.grip_to(s, 0.8)
        raise D.CycleFail(f"{bid}: the tilted grasp closed empty (grip {s['grip']:.3f})")
    op.held = (bid, gh)
    t.go(lift, D.GRIP_CLOSE, "tilt lift", "up")
    s = O.arm_state()
    if s["grip"] < D.GRIP_HELD or not s.get("grip_holding", True):  # slipped out on the first 3 cm: never carry empty
        t.grip_to(s, 0.8)
        raise D.CycleFail(f"{bid}: slipped out of the tilted grasp on the lift (grip {s['grip']:.3f})")
    t.go(carry, D.GRIP_CLOSE, "tilt carry", "down")
    t.release(O.arm_state())
    _post_wide()
    op.held = None
    t.go(out, 1.0, "back out", "up")
    if len(rise.nodes) > 1:
        t.go(rise, 1.0, "rise", "up")
    t.go(back, 1.0, "untilt away", "level")
    return {"grasp_diag": diag, "azimuth_off_radial_deg": dpsi, "nodes": [len(pick.nodes), len(carry.nodes), len(out.nodes), len(rise.nodes), len(back.nodes)], "link_clearance": lk, "z_site_set_mm": round(z_site_set * 1000),
            "run_s": round(_wall.time() - t0, 1), "refused_azimuths": tried}


def _post_wide():
    s = O.arm_state()
    D._post("/move", {"q": s["goal"], "grip": 1.0, "rate": 0.1})
    D.time.sleep(0.3)


def reload_helper(port, bid, base):
    """Sim only: every block where the truth has it, bid on the helper base, the arm where it is."""
    blocks = {k: {"x": v["x"], "y": v["y"], "yaw": v["yaw"], "level": int(round((v["bottom"] - TABLE_Z) / H))}
              for k, v in truth(port).items() if "bottom" in v}
    blocks[bid] = {"x": base["x"], "y": base["y"], "yaw": base["yaw"], "level": 1}
    q_now = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=10).read())["q"]
    body = json.dumps({"blocks": blocks, "q": q_now, "grip": 0.8}).encode()  # the arm stays where it is
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/reset", body, {"Content-Type": "application/json"}), timeout=60).read()
    _wall.sleep(2.0)


def look_belief(t, port, tower_rec, spots=(), used=()):
    """The runner's own perception (look_all from the look pose): loose and helper blocks as seen. The tower above the
    looks' reach comes from the records (its xy, yaw and the confirmed count), as the runner keeps it; anything seen
    within 3 cm of the tower's axis is the tower and is replaced by the record."""
    t.look_all("look-tilt", move=True, vet=False)
    tx, ty, tyaw, n = tower_rec
    bel = {k: b for k, b in t.belief.items() if math.hypot(b["x"] - tx, b["y"] - ty) > 0.03}
    for i in range(n):
        bel[f"tower{i}"] = {"x": tx, "y": ty, "yaw": tyaw, "z0": TABLE_Z + i * H, "level": i, "lying": False, "iou": 1.0,
                            "edge": False, "t_seen": _wall.time()}
    # the staged helpers from the records (the runner set each one up as a base and a block on it): the block seen
    # nearest a staged spot is its top, at level 1 whatever level its fit took (a raised block's top face can fit at
    # table level from the look pose, Nov 8 09:3x), with the seen xy and yaw; its hidden base from the record
    for k, (sx, sy) in enumerate(spots):
        if k in used:
            continue
        near = sorted((math.hypot(b["x"] - sx, b["y"] - sy), kk) for kk, b in bel.items() if not kk.startswith(("tower", "helper")))
        if near and near[0][0] < 0.04:
            top = bel[near[0][1]]
            top.update(level=1, z0=TABLE_Z + H)
            bel[f"helper{k}"] = {"x": top["x"], "y": top["y"], "yaw": top["yaw"], "z0": TABLE_Z, "level": 0, "lying": False,
                                 "iou": 1.0, "edge": False, "t_seen": _wall.time()}
    return bel


def main():
    port = int(sys.argv[1])
    tx, ty = (float(v) / 100 for v in sys.argv[2].split(","))
    spots = [tuple(float(v) / 100 for v in sp.split(",")) for sp in sys.argv[3].split(";")]  # staged helper stacks
    theta, gh, sag = math.radians(arg("--theta", 40.0)), arg("--up", 25.0) / 1000, arg("--sag", 0.0) / 1000
    n_levels = int(arg("--levels", 3))
    looks = "--look" in sys.argv
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"tilt_{port}.json"
    T.use_sim(port, f"runs/sim_station/p{port}")
    D.take_token()
    t = T.Tall(T.TALL / ("tilt-" + T.time.strftime("%m%d-%H%M%S")))
    t.op.st["target"] = None
    rows = []
    tr0 = column(truth_belief(port), tx, ty)
    rec = [tx, ty, 0.0, int(arg("--tower", len(tr0)))]  # the record: built by the runner before this (its count)
    used = set()
    try:
        t.belief = truth_belief(port) if not looks else {f"tower{i}": {"x": tx, "y": ty, "yaw": 0.0, "z0": TABLE_Z + i * H, "level": i,
                                                                      "lying": False} for i in range(rec[3])}
        t.put_blocks()
        t.to_pose(O.LOOK_TIP, "look")  # unfold from the rest under the home rules first, as the runner does
        for _ in range(n_levels):
            t.op.st["target"] = None
            T0 = _wall.time()
            t.belief = look_belief(t, port, rec, spots, used) if looks else truth_belief(port)
            t.put_blocks()
            row = {"tower_before": rec[3] if looks else len(column(t.belief, tx, ty)), "belief": "looks + tower record" if looks else "truth",
                   "look_s": round(_wall.time() - T0, 1)}
            # the next staged helper whose top block is in place (no reload near the tower: the helpers were staged low)
            pick = None
            for k, (sx, sy) in enumerate(spots):
                if k in used:
                    continue
                stack = column(t.belief, sx, sy, r=0.035)
                if len(stack) >= 2 or (looks and stack and t.belief[stack[-1]]["level"] >= 1):
                    pick = (k, stack[-1])
                    break
            if pick is None:
                raise SystemExit(f"no staged helper with a block on top at {spots} (seen: {sorted(t.belief)})")
            k, bid = pick
            row.update(helper=k, block=bid, block_seen={kk: round(v, 4) if isinstance(v, float) else v for kk, v in t.belief[bid].items()
                                                         if kk in ("x", "y", "yaw", "z0", "level", "iou")})
            if looks:
                top_z, txx, tyy, tyw = TABLE_Z + rec[3] * H, rec[0], rec[1], rec[2]
            else:
                tw = column(t.belief, tx, ty)
                top = t.belief[tw[-1]]
                top_z, txx, tyy, tyw = D.top_of(top), top["x"], top["y"], top["yaw"]
            row["tilt"] = tilt_place(t, bid, top_z, txx, tyy, theta, gh, sag, tyw)
            used.add(k)
            t.op.st["target"] = None
            _wall.sleep(1.5)
            tr = truth(port)  # scoring only: the runner never reads it in --look mode
            col = column(truth_belief(port), tx, ty)
            # scoring by truth: the tower grew by one (the runner's names for seen blocks are not the sim's ids)
            ok = bool(col) and len(col) == row["tower_before"] + 1
            row.update(tower_after=len(col), placed_on_top=ok, level_s=round(_wall.time() - T0, 1),
                       placed=tr.get(col[-1], {}) if col else {}, placed_id=col[-1] if col else None, tower_ids=col)
            rec[3] += 1  # the record: the grasp held through the lift check, the carry and the release (the runner's own evidence)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if not ok:
                break
    except BaseException as e:  # noqa: BLE001
        rows.append({"stop": f"{type(e).__name__}: {e}"[:3000]})
        print("STOP", rows[-1], flush=True)
    finally:
        pathlib_out = __import__("pathlib").Path(out)
        pathlib_out.write_text(json.dumps({"theta_deg": math.degrees(theta), "up_mm": gh * 1000, "sag_mm": sag * 1000, "looks": looks,
                                           "rows": rows, "final_tower": column(truth_belief(port), tx, ty)}, indent=1, default=str))
        D.release_token()


if __name__ == "__main__":
    main()
