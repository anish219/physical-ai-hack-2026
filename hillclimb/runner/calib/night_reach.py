"""A block past the fingers-down reach (Nov 8: purple at x 50.8 cm; straight-down IK ends near 48 cm) brought in with a
tilted grasp: the jaws tilted THETA deg out from straight down toward the block (their closing axis level, across the
radial line), held at that tilt through the pick, the carry and the set-down (untilting would tilt the held block),
set down on the table at X_CM along the same line (or at X_CM,Y_CM), opened, back up, to the look pose. Every leg: Plan.check (TowerSafety
and opstack's extra checks, the motion rule) and the runner's sweep over the other blocks; all legs planned before any
motion. Recorded as one move episode.

    FT_ROOT=night_nov8 python -m calib.night_reach COLOUR#N [THETA_DEG] [X_CM | X_CM,Y_CM] [--dry]
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from hillclimb.runner.calib import data_loop as D
from hillclimb.runner.calib import opstack as O
from hillclimb.runner.calib import tall_tower as T


def frame_tilt(theta, psi, s=1):
    """Tool axes (x jaw travel, y across the pads, z approach): z tilted theta off straight down toward azimuth psi."""
    z = np.array([math.sin(theta) * math.cos(psi), math.sin(theta) * math.sin(psi), -math.cos(theta)])
    x = s * np.array([-math.sin(psi), math.cos(psi), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


class TiltPlan(O.Plan):
    """Plan in one fixed tilted frame: the flange sits along the tilted approach axis from the fingertips (Plan's own
    solve puts it straight above them, true only fingers down). The same checks: within 3 mm, TowerSafety's pose rules."""

    def solve_from(self, ref, xyz, yaw_deg, keep):
        R = self.R
        target = np.asarray(xyz, float) - self.op.kin.tip * R[:, 2]
        for seed in (ref, O.LOOK_SEED):
            q, ok = self.op.kin.ik_one(target, R, seed)
            if not ok:
                q, _ = self.op.kin.ik(target, [R], seed)
                if q is None:
                    continue
            if np.linalg.norm(self.op.tip(q)[0] - np.asarray(xyz, float)) > 0.003:
                continue
            if self.op.safe.check_pose(q, self.opening)[0]:
                return q
        raise SystemExit(f"no safe tilted pose for fingertips {np.round(np.asarray(xyz) * 1000).tolist()}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bid = args[0]
    theta = math.radians(float(args[1]) if len(args) > 1 else 10.0)
    to = [float(v) / 100 for v in (args[2] if len(args) > 2 else "40").split(",")]  # X_CM (along its line) or X_CM,Y_CM
    dry = "--dry" in sys.argv
    if not dry:
        D.take_token()
    t = T.Tall(T.TALL / ("reach-" + time.strftime("%m%d-%H%M%S")))
    t.op.st["target"] = None
    op = t.op
    clean = True
    try:
        if not dry:
            D.rec("/start")
        t.look_all("look-reach", move=not dry, vet=False)
        b = t.belief[bid]
        psi = math.atan2(b["y"], b["x"])
        R = frame_tilt(theta, psi)
        cx, cy = b["x"] - O.PICK_DOWN[0], b["y"] - O.PICK_DOWN[1]
        z_grip = O.GRIP_TIP - O.PICK_DOWN[2]
        tx, ty = (to[0], to[1]) if len(to) > 1 else (to[0] * math.cos(psi), to[0] * math.sin(psi))
        q0 = t.start_q()
        # pick: over, down, last 3 cm, all in the tilted frame
        # fingers down by the motion rule's route to 6 cm short of it (inside the straight-down reach), in the frame
        # whose jaw axis is the tilted one's; then a joint blend into the tilted frame over the block (checked every 2 deg)
        pre = O.Plan(op, q0, 0.08)
        pre.R = O.frame_down(psi + math.pi / 2)
        pre.to([cx - 0.06 * math.cos(psi), cy - 0.06 * math.sin(psi), z_grip + 0.08], math.degrees(psi) + 90, D.V_HIGH, "toward", keep=True)
        pre.check(skip=bid)
        t.sweep(pre, exempt={bid})
        pick = TiltPlan(op, pre.nodes[-1][0], 0.08)
        pick.R = R
        pick.nodes = list(pre.nodes)
        # level: the blend may not go down near the block (the motion rule): 8 steps along a level line, the tilt
        # growing 1/8 each step, so the joints' interpolation between steps stays within a fraction of a mm
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
        # carry: straight up 6 cm, level in along the radial line, straight down to the set-down, all still tilted
        held = (bid, 0.015)
        carry = TiltPlan(op, pick.nodes[-1][0], 0.044, held=held)
        carry.R = R
        carry.to([cx, cy, z_grip + 0.10], 0.0, 0.3, "lift", line=0.01, keep=True)
        carry.to([tx, ty, z_grip + 0.10], 0.0, 0.4, "in", line=0.01, keep=True)
        carry.to([tx, ty, z_grip + 0.004], 0.0, D.V_SET, "set on the table", line=0.003, keep=True)
        carry.check(skip=bid)
        t.sweep(carry, exempt={bid}, held_h=0.015)
        back = TiltPlan(op, carry.nodes[-1][0], 0.08)
        back.R = R
        back.to([tx, ty, z_grip + 0.06], 0.0, 0.3, "back off", line=0.01, keep=True)
        back.check(skip=bid)
        print(f"planned: pick {len(pick.nodes)}, carry {len(carry.nodes)}, back {len(back.nodes)} nodes; "
              f"{bid} from ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) to ({tx * 100:.1f}, {ty * 100:.1f}) at {math.degrees(theta):.0f} deg tilt",
              flush=True)
        if dry:
            return
        T0 = D.now()
        src = dict(b)
        t.oh_before = t.overhead_now()  # the knock check's before frame (block() takes it for the loop's moves)
        t.go(pick, 0.8, "reach pick", "down")
        t.grip_to(O.arm_state(), D.GRIP_CLOSE, wait_close=True)
        s = O.arm_state()
        if s["grip"] < D.GRIP_HELD:
            t.grip_to(s, 0.8)
            up = TiltPlan(op, np.array(op.st["target"]), 0.08)
            up.R = R
            up.to([cx, cy, z_grip + 0.06], 0.0, 0.3, "back up", line=0.01, keep=True)
            up.check(skip=bid)
            t.go(up, 0.8, "back up", "up")
            raise D.CycleFail(f"{bid}: the tilted grasp closed empty (grip {s['grip']:.3f})")
        op.held = held
        t.go(carry, D.GRIP_CLOSE, "reach carry", "down")
        t.release(O.arm_state())
        op.held = None
        t.go(back, 0.8, "back off", "up")
        t.to_pose(O.LOOK_TIP, "look")
        T2 = D.now()
        blocks, _, _ = t.look_all("look-after", move=False, vet=False)
        f = blocks.get(bid) or min(blocks.values(), key=lambda o: math.hypot(o["x"] - tx, o["y"] - ty))
        off = math.hypot(f["x"] - tx, f["y"] - ty)
        ok = off < 0.015 and not f.get("lying") and f["level"] == 0
        t.episode(T0, T2, skill="task", prompt="", success=bool(ok), colour=bid, start_pose=src,
                  target={"x": tx, "y": ty, "yaw": src["yaw"], "level": 0, "support": None}, end_pose=f,
                  error={"offset_mm": round(off * 1000, 1)}, cycle=0, block=1, recovery=False, phase="reach",
                  grasp={"tilt_deg": round(math.degrees(theta), 1)})
        D.status(f"reach {bid}: ({src['x'] * 100:.1f}, {src['y'] * 100:.1f}) -> ({tx * 100:.1f}, {ty * 100:.1f}) at "
                 f"{math.degrees(theta):.0f} deg tilt: landed {off * 1000:.0f} mm off, {'flat' if not f.get('lying') else 'on its side'} "
                 f"({'OK' if ok else 'FAIL'})")
    except BaseException as e:  # noqa: BLE001
        D.status(f"reach {bid} stopped: {type(e).__name__}: {e}")
        if not dry:
            st = O.arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
    finally:
        if not dry:
            T.finish(t, clean)


if __name__ == "__main__":
    main()
