"""Arm geometry for the Dream backend: forward and inverse kinematics, the sag model, the tower keep-out column and
the up-first retreat.

The kinematic model, sag model and joint offsets come from the calibrated runner code (calib/tower_plan.py,
calib/arm_left.json, calib/sag_left.json, sim/servo_fit.json). Its root is HILLCLIMB_CALIB_ROOT, default ~/pah-synth
on the station. Conventions: model joints = measured joints + dq; a command for model joints q is sag.command(q) - dq.
Never send measured joints back as a goal: that skips the sag compensation (the 33 mm settle of Nov 8, 02:50).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time

import numpy as np

CALIB_ROOT = pathlib.Path(os.environ.get("HILLCLIMB_CALIB_ROOT", str(pathlib.Path.home() / "pah-synth")))

# The training start pose, COMMANDED joints (median of action row 0 over the scripted stack, move and unstack
# episodes in fine-tune B's data, ~/pah/loop/ft1_left). Jaws 80 mm = 0.80 normalized.
START_Q = (0.0, -1.78, 1.80, -1.456, 0.02, 0.0)
START_GRIP = 0.80
OPEN_GRIP = 0.85

_lock = threading.Lock()
_geo = None


class Geometry:
    def __init__(self) -> None:
        sys.path.insert(0, str(CALIB_ROOT))
        from calib.tower_plan import Kin, SagModel, frame_down, yaw_of  # noqa: PLC0415

        self.kin = Kin()
        self.sag = SagModel(self.kin)
        self.frame_down, self.yaw_of = frame_down, yaw_of
        self.dq = np.array(json.loads((CALIB_ROOT / "calib" / "arm_left.json").read_text())["dq_rad"])
        self.tip = float(self.kin.tip)

    def fk_measured(self, q_measured):
        with _lock:
            return self.kin.fk(np.asarray(q_measured, float) + self.dq)

    def fk_command(self, q_command):
        """Where a commanded row would put the hand once it settles (command + dq, then the sag)."""
        q_model = np.asarray(q_command, float) + self.dq
        with _lock:
            return self.kin.fk(q_model + self.sag(q_model))

    def points(self, p, R):
        """Tool point and both fingertips."""
        return (p, p + R[:, 2] * self.tip, p - R[:, 2] * self.tip)

    def command_for_model(self, q_model):
        return self.sag.command(q_model) - self.dq

    def ik(self, pos, R_like, q_model_ref):
        with _lock:
            qn, _ = self.kin.ik(np.asarray(pos, float), [R_like, self.frame_down(self.yaw_of(R_like))], q_model_ref, margin=0.0)
        return qn


def geometry() -> Geometry | None:
    """The shared Geometry, or None when the calibrated model cannot load (callers then fall back to joint moves)."""
    global _geo
    if _geo is None:
        try:
            _geo = Geometry()
        except Exception:  # noqa: BLE001
            _geo = False
    return _geo or None


class Column:
    """The tower's keep-out column: footprint + margin from the table up to the top + margin (left arm base frame,
    metres)."""

    def __init__(self, x: float, y: float, top_z: float, half: float = 0.025, margin: float = 0.02, top_margin: float = 0.03):
        self.x, self.y, self.top_z = x, y, top_z
        self.half, self.margin, self.top_margin = half, margin, top_margin

    def contains(self, pt) -> bool:
        return (abs(pt[0] - self.x) < self.half + self.margin and abs(pt[1] - self.y) < self.half + self.margin
                and pt[2] < self.top_z + self.top_margin)

    def hand_inside(self, geo: Geometry, p, R) -> bool:
        return any(self.contains(pt) for pt in geo.points(p, R))

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "top_z": self.top_z, "half": self.half}


def _wait_near(arm, geo: Geometry, want_p, tol: float, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            p, _ = geo.fk_measured(arm.state()["q"])
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
            continue
        if np.linalg.norm(p - want_p) < tol:
            return True
        time.sleep(0.08)
    return False


def retreat(arm, tower_top_z: float | None, column: Column | None = None, *, tol: float = 0.045,
            step_timeout: float = 2.0, rate: float = 0.8) -> list[str]:
    """Up first, then across, then into the start pose.

    1. Open the jaws where the hand is, unless the hand is inside the tower column (then keep them as they are).
    2. Rise straight up until the fingertips clear the tower top by 6 cm.
    3. Cross at that height to above the start pose's hand position.
    4. Join the start pose.
    Waypoints accept the hand within `tol` of the computed point: the arm sags 2 to 3 cm under the model, and a
    tight tolerance only waits out every timeout. Never raises; returns a short log."""
    log: list[str] = []
    geo = geometry()
    try:
        s = arm.state()
    except Exception as e:  # noqa: BLE001
        return [f"no state: {e}"]
    q_meas = np.asarray(s["q"], float)
    goal = np.asarray(s.get("goal") or s["q"], float)
    if geo is None:
        arm.move(list(START_Q), START_GRIP, rate=0.5)
        return ["no kinematic model: joint move to the start pose"]
    p0, R0 = geo.fk_measured(q_meas)
    keep = column is not None and column.hand_inside(geo, p0, R0)
    grip = float(s.get("grip", START_GRIP)) if keep else OPEN_GRIP
    if not keep:
        arm.move(list(goal), OPEN_GRIP, rate=0.3)  # open in place on the current goal, never the measured joints
        time.sleep(0.9)
    top = tower_top_z if tower_top_z is not None else p0[2]
    zt = max(p0[2], top + 0.06 + geo.tip)
    ps, _ = geo.fk_command(START_Q)
    path = [np.array([p0[0], p0[1], z]) for z in np.linspace(p0[2], zt, 4)[1:]]
    path += [np.array([x, y, zt]) for x, y in zip(np.linspace(p0[0], ps[0], 4)[1:], np.linspace(p0[1], ps[1], 4)[1:])]
    q_model = q_meas + geo.dq
    slow = 0
    for pt in path:
        qn = geo.ik(pt, R0, q_model)
        if qn is None:
            log.append(f"ik failed at {np.round(pt, 3).tolist()}")
            continue
        q_model = qn
        arm.move(list(geo.command_for_model(q_model)), grip, rate=rate)
        if not _wait_near(arm, geo, pt, tol, step_timeout):
            slow += 1
    arm.move(list(START_Q), START_GRIP, rate=0.6)
    ok = _wait_near(arm, geo, ps, tol, 6.0)
    log.append(f"rose to {zt:.3f} m" + (" jaws kept (hand in the tower column)" if keep else "")
               + (f", {slow} slow waypoints" if slow else "") + ("" if ok else ", start pose not reached in 6 s"))
    return log


def go_start(arm) -> bool:
    """A plain move to the start pose when the hand is already clear (no tower near the path)."""
    arm.move(list(START_Q), START_GRIP, rate=0.6)
    geo = geometry()
    if geo is None:
        time.sleep(3.0)
        return True
    ps, _ = geo.fk_command(START_Q)
    return _wait_near(arm, geo, ps, 0.045, 8.0)
