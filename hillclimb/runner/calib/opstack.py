"""Operator steps for stacking one foam block flat on another with the real left Metal arm, one small move at a time.

Each call does one thing (snapshot, find, one move, a grip change, a look) and appends to the run's step log, so a
person (or an agent) can look at the wrist and overhead frames between steps and decide the next one. Built on
calib/tower_plan.py (Kin, SagModel, WristView, frame_down) and calib/tower_safety.py (TowerSafety); the tower
runner's flip is not used.

Server: the team's left-arm server through the tunnel http://127.0.0.1:8691 (GET /state, /wrist.jpg, /overhead.jpg;
POST /move {q, grip, rate}). Nothing is ever posted to the team page (18081 / 8081); no stop_on_stall.

Before every move:
  - the team page (18081) says the left arm is enabled, not assigned to a leader, not recording; the Dream caller
    (18084) says left is not running. Otherwise /stop and refuse.
  - the path (joint-space line) passes TowerSafety.check_path from both the last target and the measured joints
    (model = measured + calib/arm_left.json dq_rad), plus: fingertips at y >= -0.15 m, the camera box at least 1 cm
    from every known block (the one being gripped excepted, below its top).
  - rate at most 0.3 rad/s (0.1 for descents).
Commands go out as sag.command(q_target) - dq.

    python -m calib.opstack RUN gate
    python -m calib.opstack RUN snap TAG
    python -m calib.opstack RUN find TAG                   # every block in the wrist view (no guess)
    python -m calib.opstack RUN look COLOUR TAG [Z0]       # one block near its stored estimate; Z0 its bottom
    python -m calib.opstack RUN to X Y TIPZ YAWDEG GRIP RATE TAG [home]   # fingertips to x, y, z (m), jaws at yaw
    python -m calib.opstack RUN grip GRIP TAG
    python -m calib.opstack RUN sheet "STEM:colour@z0[~held],...|caption" ...   # overlay sheet of saved steps
    python -m calib.opstack RUN joints RATE TAG q0..q5 [GRIP]  # straight to measured-frame joints (the way home)
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

from hillclimb.runner.calib.tower_plan import FACE, H, Kin, SagModel, Shape, WristView, frame_down
from hillclimb.runner.calib.tower_safety import BOX_HI, BOX_LO, TowerSafety

ROOT = pathlib.Path(__file__).resolve().parent.parent
_RUNS_ROOT = pathlib.Path(os.environ.get("HILLCLIMB_RUNS", str(pathlib.Path.home() / "hillclimb_runs")))
URL = "http://127.0.0.1:8691"
TEAM = "http://127.0.0.1:18081/api/state"
DREAM = "http://127.0.0.1:18084/api/state"
TABLE_Z = -0.008
MAX_RATE = 0.3
TIP_Y_MIN = -0.20  # -0.15 refused orange at y -154 mm (Nov 8 14:24); the strip and right arm sit past y -35 cm
CYAN, AMBER = (255, 224, 63), (40, 170, 245)
COLOURS = ("green", "blue", "purple")


def _get(url, timeout=5):
    return urllib.request.urlopen(url, timeout=timeout)


def _post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def arm_state():
    return json.loads(_get(URL + "/state").read())


def gate():
    """(ok, why): the left arm is on and nobody else drives it."""
    try:
        a = json.loads(_get(TEAM).read())
        b = json.loads(_get(DREAM).read())
    except Exception as e:  # noqa: BLE001
        return False, f"could not read the team page or Dream caller: {e}"
    left = (a.get("arms") or {}).get("left") or {}
    why = (f"enabled={left.get('enabled')!r} assigned_side={a.get('assigned_side')!r} recording={a.get('recording')!r} "
           f"dream left running={(b.get('left') or {}).get('running')!r}")
    ok = left.get("enabled") is True and a.get("assigned_side") != "left" and a.get("recording") is False \
        and (b.get("left") or {}).get("running") is False
    return ok, why


class Op:
    def __init__(self, run: pathlib.Path):
        self.run = run
        run.mkdir(parents=True, exist_ok=True)
        self.f = run / "op.json"
        self.st = json.loads(self.f.read_text()) if self.f.exists() else {"k": 0, "blocks": {}, "target": None, "start": None}
        self.dq = np.array(json.loads((ROOT / "calib" / "arm_left.json").read_text())["dq_rad"])
        self.kin = Kin()
        self.sag = SagModel(self.kin)
        self.view = WristView(self.kin)
        self.safe = TowerSafety()
        if self.st["start"] is None:
            self.st["start"] = arm_state()["q"]
            self.save()

    def save(self):
        self.f.write_text(json.dumps(self.st, indent=1))

    def log(self, tag, saw="", decided="", command=None):
        with open(self.run / "steps.jsonl", "a") as fh:
            fh.write(json.dumps({"t": time.strftime("%H:%M:%S"), "k": self.st["k"], "tag": tag, "saw": saw, "decided": decided,
                                 "command": command}) + "\n")

    # cameras
    def snap(self, tag):
        """Both cameras and the state, saved as NN-tag-*. Returns the wrist frame at 640x360, its model joints, grip."""
        self.st["k"] += 1
        k = self.st["k"]
        stem = f"{k:02d}-{tag}"
        meta = {}
        wrist = None
        for cam in ("wrist", "overhead"):
            r = _get(f"{URL}/{cam}.jpg")
            data = r.read()
            (self.run / f"{stem}-{cam}.jpg").write_bytes(data)
            meta[cam] = {"joints": json.loads(r.headers.get("X-Joints") or "null"), "grip": json.loads(r.headers.get("X-Grip") or "null"),
                         "stamp": r.headers.get("X-Stamp")}
            if cam == "wrist":
                wrist = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        meta["state"] = arm_state()
        (self.run / f"{stem}.json").write_text(json.dumps(meta))
        self.save()
        q = np.array(meta["wrist"]["joints"], float) + self.dq
        grip = float(meta["wrist"]["grip"] or 0.0)
        return cv2.resize(wrist, (640, 360), interpolation=cv2.INTER_AREA), q, grip, stem

    def overlay(self, img, q, opening, fits, label):
        """Jaw CAD and the fitted blocks: thin cyan outlines (silhouette and top face), amber labels in cm."""
        out = img.copy()
        cs, _ = cv2.findContours(self.view.jaws_mask(q, opening), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, CYAN, 1, cv2.LINE_AA)
        cam = self.view.pose(q, opening)
        for name, (fit, shape) in fits.items():
            if fit is None:
                continue
            hull = self.view.silhouette(cam, shape, fit["x"], fit["y"], fit["yaw"])
            if hull is None:
                continue
            cv2.polylines(out, [hull], True, CYAN, 1, cv2.LINE_AA)
            top = shape.corners(fit["x"], fit["y"], fit["yaw"])[[1, 3, 7, 5]]
            uv, ok = self.view.project(cam, top)
            if ok.all():
                cv2.polylines(out, [np.round(uv).astype(np.int32)], True, CYAN, 1, cv2.LINE_AA)
            at = tuple(int(v) for v in hull.reshape(-1, 2).min(0))
            cv2.putText(out, f"{name} {fit['x'] * 100:.1f}, {fit['y'] * 100:.1f} cm {math.degrees(fit['yaw']):+.0f} deg",
                        (max(2, at[0]), max(28, at[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, AMBER, 1, cv2.LINE_AA)
        cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, AMBER, 1, cv2.LINE_AA)
        return out

    def find_all(self, tag):
        img, q, grip, stem = self.snap(tag)
        opening = grip * 0.1
        flat = Shape((FACE, FACE, H), TABLE_Z)
        fits = {}
        for c in COLOURS:
            fit = self.view.find(img, c, q, opening, flat)
            fits[c] = (fit, flat)
            if fit is not None:
                self.st["blocks"][c] = {"x": fit["x"], "y": fit["y"], "yaw": fit["yaw"], "iou": fit["iou"], "edge": fit["edge"], "z0": TABLE_Z}
        cv2.imwrite(str(self.run / f"{stem}-ov.jpg"), self.overlay(img, q, opening, fits, stem))
        self.save()
        tip = self.tip(q)
        return {c: (None if f is None else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}) for c, (f, _) in fits.items()}, tip, stem

    def look(self, colour, tag, z0=TABLE_Z, guess=None, occlude=None):
        """occlude: the colour of a held block; its pixels (grown 4 px) are masked out with the jaws, so a block
        partly hidden behind it is fitted on what shows."""
        img, q, grip, stem = self.snap(tag)
        opening = grip * 0.1
        shape = Shape((FACE, FACE, H), z0)
        b = self.st["blocks"].get(colour)
        g = guess or ((b["x"], b["y"], b["yaw"]) if b else None)
        jaws_mask = self.view.jaws_mask
        if occlude:
            def masked(q_, o_):
                j = jaws_mask(q_, o_)
                held = self.view.colour_mask(img, occlude, j)
                return np.maximum(j, cv2.dilate(held * 255, np.ones((9, 9), np.uint8)))
            self.view.jaws_mask = masked
        try:
            fit = self.view.find(img, colour, q, opening, shape, g)
        finally:
            self.view.jaws_mask = jaws_mask
        cv2.imwrite(str(self.run / f"{stem}-ov.jpg"), self.overlay(img, q, opening, {colour: (fit, shape)}, stem))
        tip = self.tip(q)
        return fit, tip, stem, grip

    # kinematics
    def tip(self, q):
        """Fingertip point (between the pad ends) of model joints q, and the jaw yaw."""
        pos, R = self.kin.fk(q)
        return pos + self.kin.tip * R[:, 2], math.degrees(math.atan2(R[1, 0], R[0, 0]))

    def solve(self, xyz_tip, yaw, opening):
        ref = np.array(self.st["target"] if self.st["target"] else np.array(arm_state()["q"]) + self.dq)
        frames = [frame_down(yaw + k * math.pi / 2) for k in range(4)]
        from hillclimb.runner.calib.tower_safety import DOWN_MIN, outward_ok
        frames = [R for R in frames if outward_ok(R, xyz_tip[:2])]
        cands = []
        for R in frames:
            q, _ = self.kin.ik(np.asarray(xyz_tip, float) + [0, 0, self.kin.tip], [R], ref)
            if q is not None:
                cands.append((float(np.abs(q - ref).sum()), q, R))
        if not cands:
            return None, "no IK"
        why = ""
        for _, q, R in sorted(cands, key=lambda c: c[0]):
            ok, why = self.safe.check_pose(q, opening)
            if ok:
                return q, "ok"
        return None, why

    def extra_checks(self, q0, q1, skip=None):
        """Fingertips y >= TIP_Y_MIN and the camera box 1 cm clear of every known block along the path."""
        n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / math.radians(2))) + 1)
        corners = np.array([[x, y, z] for x in (BOX_LO[0], BOX_HI[0]) for y in (BOX_LO[1], BOX_HI[1]) for z in (BOX_LO[2], BOX_HI[2])])
        for t in np.linspace(0, 1, n):
            q = q0 + (q1 - q0) * t
            (p, _yaw) = self.tip(q)
            if p[1] < TIP_Y_MIN:
                return False, f"at {t * 100:.0f}%: fingertips at y {p[1] * 1000:.0f} mm (min {TIP_Y_MIN * 1000:.0f})"
            Rc, pc = self.safe.camera(q)
            cw = corners @ Rc.T + pc
            for c, b in self.st["blocks"].items():
                if c == skip:
                    continue
                h = b.get("half", FACE / 2)
                top = b.get("z0", TABLE_Z) + H
                dx = np.maximum(np.abs(cw[:, 0] - b["x"]) - h, 0)
                dy = np.maximum(np.abs(cw[:, 1] - b["y"]) - h, 0)
                dz = np.maximum(cw[:, 2] - top, 0)
                if np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).min() < 0.01:
                    return False, f"at {t * 100:.0f}%: camera within 1 cm of the {c} block"
        return True, "ok"

    def move(self, q_target, grip, rate, tag, home=False, skip=None, note="", raw=False):
        """raw: send q_target - dq as is, no sag compensation (the way back to the measured start joints)."""
        ok, why = gate()
        if not ok:
            _post("/stop", {})
            raise SystemExit(f"GATE: {why}: stopped")
        q_target = np.asarray(q_target, float)
        opening = float(grip) * 0.1
        q_meas = np.array(arm_state()["q"]) + self.dq
        q_last = np.array(self.st["target"]) if self.st["target"] else q_meas
        for name, q0 in (("from the last target", q_last), ("from the measured joints", q_meas)):
            ok, why = self.safe.check_path(q0, q_target, opening, home=home)
            if ok:
                ok, why = self.extra_checks(q0, q_target, skip)
            if not ok:
                self.log(tag, decided=f"REFUSED {name}: {why}", command={"q_model": q_target.tolist(), "grip": grip})
                raise SystemExit(f"REFUSED {name}: {why}")
        # the motion rule holds for single operator moves too (a sideways step low past a stack is refused)
        one = Plan(self, q_last, opening)
        one.nodes.append((q_target, rate, tag))
        try:
            one.rule_check()
        except SystemExit as e:
            self.log(tag, decided=f"REFUSED: {e}", command={"q_model": q_target.tolist(), "grip": grip})
            raise
        rate = float(min(rate, MAX_RATE))
        cmd = (q_target if raw else self.sag.command(q_target)) - self.dq
        body = {"q": [float(v) for v in cmd], "grip": float(np.clip(grip, 0, 1)), "rate": rate}
        _abort_check(tag)
        _post("/move", body)
        t0 = time.time()
        time.sleep(0.3)
        while True:
            _abort_check(tag)
            s = arm_state()
            if s.get("error"):
                raise SystemExit(f"arm server error: {s['error']}")
            if not s["moving"]:
                break
            if time.time() - t0 > 60:
                _post("/stop", {})
                raise SystemExit("move did not finish in 60 s: stopped")
            ok, why = gate()
            if not ok:
                _post("/stop", {})
                raise SystemExit(f"GATE during move: {why}: stopped")
            time.sleep(0.15)
        time.sleep(0.8)  # the servos lag 135-215 ms; let the sag settle
        s = arm_state()
        q_now = np.array(s["q"]) + self.dq
        self.st["target"] = q_target.tolist()
        self.save()
        err = q_now - q_target
        tip_t, _ = self.tip(q_target)
        tip_m, yaw_m = self.tip(q_now)
        res = {"sec": round(time.time() - t0, 1), "grip": round(s["grip"], 3), "goal_grip": round(s["goal_grip"], 3),
               "joint_err_deg": np.round(np.degrees(err), 1).tolist(), "tip_target_mm": np.round(tip_t * 1000).tolist(),
               "tip_measured_mm": np.round(tip_m * 1000).tolist(), "yaw_measured_deg": round(yaw_m, 1)}
        self.log(tag, decided=note, command={"q_model": q_target.tolist(), "cmd": body, "result": res})
        return res


def sheet(op: Op, specs, out_name="sheet.jpg"):
    """Overlay sheet from saved steps. specs: "STEM:colour@z0[~occluder],colour@z0,...|caption"; each frame with the jaw
    CAD and the fitted blocks (thin cyan outlines, amber labels), three to a row."""
    tiles = []
    for spec in specs:
        body, caption = (spec.split("|", 1) + [""])[:2]
        stem, rest = body.split(":", 1)
        meta = json.loads((op.run / f"{stem}.json").read_text())
        q = np.array(meta["wrist"]["joints"], float) + op.dq
        opening = float(meta["wrist"]["grip"] or 0.0) * 0.1
        img = cv2.resize(cv2.imread(str(op.run / f"{stem}-wrist.jpg")), (640, 360), interpolation=cv2.INTER_AREA)
        fits = {}
        for item in rest.split(","):
            occ = None
            if "~" in item:
                item, occ = item.split("~")
            colour, z0 = item.split("@")
            shape = Shape((FACE, FACE, H), float(z0))
            b = op.st["blocks"].get(colour)
            jm = op.view.jaws_mask
            if occ:
                op.view.jaws_mask = lambda q_, o_, jm=jm, occ=occ: np.maximum(jm(q_, o_), cv2.dilate(
                    op.view.colour_mask(img, occ, jm(q_, o_)) * 255, np.ones((9, 9), np.uint8)))
            try:
                fit = op.view.find(img, colour, q, opening, shape, (b["x"], b["y"], b["yaw"]) if b else None)
            finally:
                op.view.jaws_mask = jm
            fits[colour] = (fit, shape)
        tiles.append(op.overlay(img, q, opening, fits, f"{stem} {caption}"))
    while len(tiles) % 3:
        tiles.append(np.zeros((360, 640, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    path = op.run / out_name
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


# ---------------------------------------------------------------------------------------------------------------
# One look, then one streamed motion each for the pick and the place (Nov 7, evening).
RECORDS = _RUNS_ROOT / "station" / "opstack" / "calib_records.jsonl"
# measured minus commanded fingertips (m) at the end of a straight-down approach. Commands aim at target - offset.
# x, y: run 182628's slow stepped descents (pick 8 moves, place 5, spread about 1 mm) and run 184543's streamed ones
# agree (-2.9..-4.5, -0.6..-1.2 mm). z: stepped descents end 2.8 (pick) and 3.8 mm (place) high, but the streamed
# ones of 184543 ended +0.8 and -0.4 mm (one each), so the streamed values are used: the set landed about 1 mm into
# the foam when aimed with the stepped offset.
PICK_DOWN = np.array([-0.0031, -0.0006, 0.0008])
PLACE_DOWN = np.array([-0.0040, -0.0012, -0.0004])
GRIP_TIP = 0.007  # measured fingertips this high (arm frame) when gripping a block on the table (15 mm over it)
CARRY_Z = 0.150  # fingertips over everything by more than 8 cm (the tower top is 61 mm)
LOOK_TIP = (0.28, 0.0, 0.19, 90.0)
if os.environ.get("LOOK_TIP"):  # "x,y,z,yaw": a look pose off a tall tower standing under the usual one (Nov 8 11:46)
    LOOK_TIP = tuple(float(v) for v in os.environ["LOOK_TIP"].split(","))
LOOK_SEED = np.array([0.0, -1.84, 1.76, -1.48, 0.0, 0.0])  # joints near the look pose: an IK seed that solves most of the table
TICK = 1 / 30
STREAM_HOOK = None  # tests: called with (tag, model joints, grip, held) for every streamed goal
# set by the harness (backend.hold) to stop a move in progress: every streamed goal and every stepped move checks it
ABORT = threading.Event()


def _abort_check(tag):
    if ABORT.is_set():
        _post("/stop", {})
        raise SystemExit(f"{tag}: stopped by the harness")


def record(kind, **d):
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORDS, "a") as fh:
        fh.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **d}, default=lambda v: np.asarray(v).tolist()) + "\n")


def fetch(cam):
    r = _get(f"{URL}/{cam}.jpg")
    data = r.read()
    return data, json.loads(r.headers.get("X-Joints") or "null"), json.loads(r.headers.get("X-Grip") or "null")


def overhead_blocks(data):
    from hillclimb.runner.calib.fit_overhead import TABLE_TOP_ROW
    from hillclimb.runner.vision.locate import Camera, track
    from hillclimb.runner.vision.track_real import IGNORE
    cam = Camera.from_calib(json.loads((ROOT / "calib" / "cam_mid.json").read_text()))
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    out = {}
    for p in track(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cam, TABLE_Z, ignore=IGNORE, table_row=TABLE_TOP_ROW):
        if p["kind"] == "block":
            out.setdefault(p["colour"], []).append({"x": p["xyz"][0], "y": p["xyz"][1], "yaw": p["yaw"], "px": p["px"]})
    return out


class GateWatch:
    """The team page and Dream caller polled in the background while a motion streams."""

    def __init__(self):
        import threading
        self.ok, self.why, self.stop = True, "", False
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        misses = 0  # the team page's tunnel drops a request now and then; a closed gate stops at once, 1.5 s of no reply stops too
        while not self.stop:
            ok, why = gate()
            misses = misses + 1 if why.startswith("could not read") else 0
            if not ok and (misses == 0 or misses >= 6):
                self.ok, self.why = False, why
            time.sleep(0.25)


# ---------------------------------------------------------------------------------------------------------------
# Isaiah's motion rule (Nov 7, night: a tower knocked over by a low sideways move past it), for every pick, place,
# lift, back-off, look and carry that goes through Plan:
#   1. before any sideways move, rise in z only until the held block's bottom (or the open fingertips, empty) is
#      RULE_CLEAR over the tallest standing stack within RULE_NEAR of the path;
#   2. then x, y and yaw only, at that height, until right over the target and aligned;
#   3. then down in z only (to grip, to set down, or to the next height);
#   4. after a release, up in z only before anything sideways (the back-off is a straight-up line);
#   5. never downward together with x/y within RULE_NO_DIAG of a stack.
# Plan.to builds every free move that way (route); Plan.rule_check samples the fingertip path of every piece and
# refuses one that breaks it, and stream() runs it again before the first goal is posted. The stacks are the
# believed blocks (op.st["blocks"], the held one left out); the held block's bottom is plan.held[1] under the
# fingertips.
RULE_NEAR = 0.12
RULE_CLEAR = 0.02
RULE_NO_DIAG = 0.06
RULE_XY_TOL = 0.003  # a piece whose fingertips stay within this (x, y) and turn under RULE_YAW_TOL is vertical
RULE_YAW_TOL = 3.0
RULE_Z_TOL = 0.002
RULE_STEP = 0.02  # Cartesian node spacing of routed lines
RULE_REACH = 0.238  # highest fingers-down fingertips (data_loop.REACH_Z)


def rule_stacks(op, exempt=()):
    """(x, y, top) of every believed block but the exempt ones (a stack's top block is its tallest point)."""
    out = [(b["x"], b["y"], (b.get("z0") if b.get("z0") is not None else TABLE_Z) + H)
           for c, b in (op.st.get("blocks") or {}).items() if c not in exempt]
    return np.array(out, float).reshape(-1, 3)


def seg_dist(P, a, b):
    """Distance (x, y) of each point of P (n, 2) to the segment a-b."""
    P, a, b = np.asarray(P, float).reshape(-1, 2), np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def rule_floor(stacks, pts_xy, held_h):
    """The lowest legal fingertip height for sideways motion through these points (x, y): the tallest stack within
    RULE_NEAR of any of them (the table if none), plus RULE_CLEAR, plus the held block under the fingertips."""
    pts_xy = np.asarray(pts_xy, float).reshape(-1, 2)
    top = TABLE_Z
    if len(stacks):
        if len(pts_xy) == 2:
            d = seg_dist(stacks[:, :2], pts_xy[0], pts_xy[1])
        else:
            d = np.min(np.linalg.norm(stacks[None, :, :2] - pts_xy[:, None, :], axis=2), axis=0)
        near = stacks[d <= RULE_NEAR]
        if len(near):
            top = max(top, float(near[:, 2].max()))
    return top + RULE_CLEAR + held_h, top


def _yaw_of(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def _turn(a, b):
    return (b - a + 180.0) % 360.0 - 180.0


class Plan:
    """Waypoints solved to joints, every piece checked before anything moves. held: (block id, its bottom under the
    fingertips in m) while a block is in the jaws (op.held by default)."""

    def __init__(self, op: Op, q0, opening, held=None):
        self.op, self.nodes, self.opening = op, [(np.asarray(q0, float), 0.0, "start")], opening
        self.R = None
        self.held = held if held is not None else getattr(op, "held", None)

    def held_h(self):
        return float(self.held[1]) if self.held else 0.0

    def exempt(self):
        return {self.held[0]} if self.held else set()

    def solve_R(self, ref, xyz, R, max_jump=0.6):
        """Fingertips at xyz with the tool frame R exactly, seeded at ref; None when no safe pose is that close."""
        xyz = np.asarray(xyz, float)
        pos = xyz - self.op.kin.tip * R[:, 2]
        q, ok = self.op.kin.ik_one(pos, R, ref)
        if not ok:
            q, _ = self.op.kin.ik(pos, [R], ref)
            if q is None:
                return None
        if np.linalg.norm(self.op.tip(q)[0] - xyz) > 0.001 or float(np.abs(q - ref).max()) > max_jump:
            return None
        if not self.op.safe.check_pose(q, self.opening)[0]:
            return None
        return q

    def _line_nodes(self, q, pa, pb, Ra, Rb, step=RULE_STEP):
        """Nodes along the straight line pa-pb, the frame turned from Ra to Rb about the vertical as it goes."""
        d = float(np.linalg.norm(np.asarray(pb) - np.asarray(pa)))
        turn = math.radians(_turn(_yaw_of(Ra), _yaw_of(Rb)))
        n = max(1, int(math.ceil(d / step)), int(math.ceil(abs(turn) / math.radians(10))))
        out = []
        for i in range(1, n + 1):
            s = i / n
            a = turn * s
            Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
            R = Rb if i == n else Rz @ Ra
            q = self.solve_R(q, np.asarray(pa) + (np.asarray(pb) - np.asarray(pa)) * s, R)
            if q is None:
                return None
            out.append(q)
        return out

    def route(self, xyz, yaw_deg, v, tag, keep=False):
        """The motion rule's way to xyz: straight up (or down) to the traverse height, across level while turning to
        the end frame, straight down to xyz. The traverse height is the rule's floor for this path, or xyz's own
        height if higher; higher ones are tried when that one does not solve."""
        xyz = np.asarray(xyz, float)
        q0 = self.nodes[-1][0]
        p0 = self.op.tip(q0)[0]
        R0 = self.op.kin.fk(q0)[1]
        q1 = self.solve_from(q0, xyz, yaw_deg, keep)
        R1 = self.R
        held_h = self.held_h()
        stacks = rule_stacks(self.op, self.exempt())
        floor, _ = rule_floor(stacks, [p0[:2], xyz[:2]], held_h)
        if R0[:, 2] @ np.array([0, 0, -1.0]) < 0.9:
            # not fingers down (the folded rest, its fingertips about 17 cm out and 19 cm up): first up, fingers
            # turned down, over where the fingertips already are (a joint piece; nearly straight up), then the rule's
            # route from there like any other move; rule_check judges the unfold's fingertip path too
            z_u0 = max(floor, p0[2], xyz[2])
            for z_u in (z_u0, z_u0 + 0.02, z_u0 + 0.04, RULE_REACH):
                if z_u > RULE_REACH + 1e-6:
                    continue
                u = np.array([p0[0], p0[1], z_u])
                for k in range(4):
                    Ru = frame_down(math.radians(yaw_deg) + k * math.pi / 2)
                    from hillclimb.runner.calib.tower_safety import outward_ok
                    if not outward_ok(Ru, u[:2]):
                        continue
                    q_u = self.solve_R(q0, u, Ru, max_jump=3.0)
                    if q_u is None:
                        continue
                    n0 = len(self.nodes)
                    self.nodes.append((q_u, v, tag))
                    R_keep = self.R
                    self.R = Ru
                    try:
                        self.route(xyz, yaw_deg, v, tag, keep)
                        return
                    except SystemExit:
                        del self.nodes[n0:]
                        self.R = R_keep
            raise SystemExit(f"no unfold to fingertips {np.round(xyz * 1000).tolist()} ({tag})")
        if np.linalg.norm(xyz[:2] - p0[:2]) < RULE_XY_TOL and abs(_turn(_yaw_of(R0), _yaw_of(R1))) < RULE_YAW_TOL:
            qs = self._line_nodes(q0, p0, xyz, R0, R1)
            if qs is None:
                raise SystemExit(f"no straight line to fingertips {np.round(xyz * 1000).tolist()} ({tag})")
            self.nodes += [(q, v, tag) for q in qs]
            self.R = R1
            return
        # the traverse height: the end's own height (or the floor if higher) first; then the start's; then any
        # from the floor up (far out the reach is low: across lower, then straight up at the end)
        z_t0 = max(floor, xyz[2])
        heights = [z_t0, max(floor, p0[2])] + [floor + k * 0.03 for k in range(9)] + [z_t0 + 0.03, z_t0 + 0.06]
        order = []
        for h in heights:
            h = round(h, 4)
            if floor - 1e-6 <= h <= RULE_REACH + 1e-6 and h not in order:
                order.append(h)
        whys = []
        why = f"floor {floor * 1000:.0f} mm"
        for z_t in order:
            a, b = np.array([p0[0], p0[1], z_t]), np.array([xyz[0], xyz[1], z_t])
            nodes = []
            q = q0
            ok = True
            for pa, pb, Ra, Rb in ((p0, a, R0, R0), (a, b, R0, R1), (b, xyz, R1, R1)):
                if np.linalg.norm(np.asarray(pb) - np.asarray(pa)) < 1e-4 and Ra is Rb:
                    continue
                qs = self._line_nodes(q, pa, pb, Ra, Rb)
                if qs is None:
                    ok = False
                    whys.append(f"{'rise' if pa is p0 else 'traverse' if pa is a else 'descent'} at {z_t * 1000:.0f}")
                    why = f"floor {floor * 1000:.0f} mm; no pose along the " + ", ".join(whys)
                    break
                nodes += qs
                q = qs[-1]
            if ok:
                self.nodes += [(qq, v, tag) for qq in nodes]
                self.R = R1
                return
        # an escape waypoint (Nov 8 14:24: blocks 36-45 cm out refused because the move left from over a 4-high tower,
        # so the tower's floor held over the whole straight path, too high that far out): rise to the floor, go along
        # the line until clear of every tall stack's RULE_NEAR, then the rest under the lower floor out there.
        # rule_check still judges the whole path.
        if not getattr(self, "_escaping", False) and len(stacks):
            d = xyz[:2] - p0[:2]
            L = float(np.linalg.norm(d))
            tall = stacks[stacks[:, 2] > TABLE_Z + H + 0.01]  # 2 or more high: a lone block's floor is low anyway
            if L > 0.05 and len(tall):
                for s_ in np.linspace(0.1, 0.8, 15):
                    w = p0[:2] + d * s_
                    if float(np.min(np.linalg.norm(tall[:, :2] - w, axis=1))) <= RULE_NEAR + 0.01:
                        continue
                    n0, R_keep = len(self.nodes), self.R
                    self._escaping = True
                    try:
                        self.route([float(w[0]), float(w[1]), float(min(max(floor, p0[2]), RULE_REACH))], yaw_deg, v, tag + " escape", keep)
                        self.route(xyz, yaw_deg, v, tag, keep)
                        return
                    except SystemExit as e2:
                        del self.nodes[n0:]
                        self.R = R_keep
                        why += f"; escape at {s_:.2f}: {str(e2)[:80]}"
                        break
                    finally:
                        self._escaping = False
        raise SystemExit(f"no rule route to fingertips {np.round(xyz * 1000).tolist()} yaw {yaw_deg:.0f} ({tag}): {why}")

    def rule_check(self):
        """The motion rule on the sampled fingertip path of every piece (1 deg of the largest joint change apart)."""
        stacks = rule_stacks(self.op, self.exempt())
        held_h = self.held_h()
        what = "the held block's bottom" if self.held else "the fingertips"
        for (qa, _, _), (qb, _, tag) in zip(self.nodes[:-1], self.nodes[1:]):
            n = max(3, int(math.ceil(np.max(np.abs(qb - qa)) / math.radians(1))) + 1)
            P, Y = [], []
            for t in np.linspace(0, 1, n):
                pos, R = self.op.kin.fk(qa + (qb - qa) * t)
                P.append(pos + self.op.kin.tip * R[:, 2])
                Y.append(_yaw_of(R))
            P = np.array(P)
            xy_dev = float(np.max(np.linalg.norm(P[:, :2] - P[0, :2], axis=1)))
            turn = float(np.max(np.abs([_turn(Y[0], y) for y in Y])))
            if xy_dev <= RULE_XY_TOL and turn <= RULE_YAW_TOL:
                continue  # vertical: always allowed
            floor, top = rule_floor(stacks, P[:, :2], held_h)
            low = float(P[:, 2].min())
            if low < floor - RULE_Z_TOL:
                c = (low - held_h - top) * 1000
                raise SystemExit(f"motion rule refused {tag}: sideways ({xy_dev * 1000:.0f} mm, {turn:.0f} deg) with {what} "
                                 f"{abs(c):.0f} mm {'over' if c >= 0 else 'under'} the top of the tallest stack within "
                                 f"{RULE_NEAR * 100:.0f} cm (top {top * 1000:.0f} mm; {RULE_CLEAR * 1000:.0f} mm over needed)")
            if len(stacks):
                d = np.min(np.linalg.norm(stacks[None, :, :2] - P[:, None, :2], axis=2), axis=1)
                dz = np.diff(P[:, 2])
                near = (d[1:] < RULE_NO_DIAG) | (d[:-1] < RULE_NO_DIAG)
                down = float(-dz[near & (dz < 0)].sum())
                if down > RULE_Z_TOL:
                    raise SystemExit(f"motion rule refused {tag}: down {down * 1000:.0f} mm together with x/y within "
                                     f"{RULE_NO_DIAG * 100:.0f} cm of a stack")
        return True

    def solve_from(self, ref, xyz, yaw_deg, keep):
        from hillclimb.runner.calib.tower_safety import outward_ok
        all4 = [R for R in (frame_down(math.radians(yaw_deg) + k * math.pi / 2) for k in range(4)) if outward_ok(R, xyz[:2])]
        frames = [self.R] if keep and self.R is not None else all4
        best = None
        # a kept frame or a far seed can fail where the block's other symmetric grasps and the look pose's seed solve
        # (Nov 7 21:20: far blocks at x 36-42 cm refused though yaw +90 grasps there pass IK and every check)
        for frames_, seeds in ((frames, [ref]), (all4, [ref, LOOK_SEED])):
            for R in frames_:
                for seed in seeds:
                    q, ok = self.op.kin.ik_one(np.asarray(xyz, float) + [0, 0, self.op.kin.tip], R, seed)
                    if not ok:
                        q, _ = self.op.kin.ik(np.asarray(xyz, float) + [0, 0, self.op.kin.tip], [R], seed)
                        if q is None:
                            continue
                    # the fallback ik() can return a pose that never reached the target (Nov 7 21:25: a place landed
                    # 116 mm off): accept only a solution whose fingertips are within 3 mm of where they were asked
                    if np.linalg.norm(self.op.tip(q)[0] - np.asarray(xyz, float)) > 0.003:
                        continue
                    if not self.op.safe.check_pose(q, self.opening)[0]:
                        continue
                    c = float(np.abs(q - ref).sum())
                    if best is None or c < best[0]:
                        best = (c, q, R)
            if best is not None:
                break
        if best is None:
            raise SystemExit(f"no safe pose for fingertips {np.round(np.asarray(xyz) * 1000).tolist()} yaw {yaw_deg:.0f}")
        self.R = best[2]
        return best[1]

    def to(self, xyz, yaw_deg, v, tag, line=None, keep=False):
        """Fingertips to xyz. line: Cartesian steps (m) along the straight line from the last node (fingertips),
        else the motion rule's route (up, across level, down)."""
        q_prev = self.nodes[-1][0]
        if line:
            p0 = self.op.tip(q_prev)[0]
            n = max(1, int(math.ceil(np.linalg.norm(np.asarray(xyz) - p0) / line)))
            for i in range(1, n + 1):
                q_prev = self.solve_from(q_prev, p0 + (np.asarray(xyz) - p0) * i / n, yaw_deg, keep=True)
                self.nodes.append((q_prev, v, tag))
        else:
            self.route(xyz, yaw_deg, v, tag, keep)

    def check(self, skip=None, floor=None):
        for (q0, _, _), (q1, v, tag) in zip(self.nodes[:-1], self.nodes[1:]):
            ok, why = self.op.safe.check_path(q0, q1, self.opening)
            if ok:
                ok, why = self.op.extra_checks(q0, q1, skip)
            if ok and floor is not None and tag in floor:
                for t in np.linspace(0, 1, 9):
                    z = self.op.tip(q0 + (q1 - q0) * t)[0][2]
                    if z < floor[tag]:
                        ok, why = False, f"fingertips {z * 1000:.0f} mm, under the {floor[tag] * 1000:.0f} mm floor"
            if not ok:
                raise SystemExit(f"plan refused at {tag}: {why}")
        self.rule_check()
        # the keep-out cylinders at plan time too, so planners (and dry runs) choose a way around a tower instead of
        # being stopped at stream time (Nov 8 12:04: a dry run passed a reach that the stream then refused)
        from hillclimb.runner.calib import keepout as KO
        KO.check(self.op, self.nodes, "plan", opening=float(self.opening or 0.095), hand=False)  # links only at plan time


# the arm server walks its command toward the goal at most 0.6 rad/s (station/armserver.py MAX_RATE, the sim the
# same): goals posted faster run ahead and the command cuts across in joint space (a straight-down line came out
# 7 mm sideways in sim); every piece streams at most this
WALK_MAX = 0.58
LOW_V = 0.3
FAST_SLOW_TAGS = ("last 3 cm",)
CORNER_SETTLE = (0.07, 0.6, 0.004)  # at a corner: at least, at most (s), and the joint change (rad) that counts as still


def corners_of(op: Op, nodes):
    """Nodes where the fingertips turn between straight up or down and sideways (the motion rule's corners)."""
    P = [op.tip(q)[0] for q, _, _ in nodes]
    out, prev = [], None
    for i, (a, b) in enumerate(zip(P[:-1], P[1:])):
        d = b - a
        if np.linalg.norm(d) < 2e-4:
            continue
        k = "v" if np.linalg.norm(d[:2]) <= 0.25 * abs(d[2]) + 5e-4 else "s"
        if prev is not None and k != prev:
            out.append(i)
        prev = k
    return out


def settle(op: Op, watch: GateWatch, tag):
    """Wait until the arm stops at the goal just posted (servo lag rounds a corner otherwise)."""
    t0 = time.time()
    q_prev, still = None, 0
    while time.time() - t0 < CORNER_SETTLE[1]:
        if not watch.ok:
            _post("/stop", {})
            raise SystemExit(f"GATE during {tag}: {watch.why}: stopped")
        s = arm_state()
        q = np.array(s["q"])
        still = still + 1 if q_prev is not None and not s["moving"] and np.abs(q - q_prev).max() < CORNER_SETTLE[2] else 0
        q_prev = q
        if still >= 2 and time.time() - t0 >= CORNER_SETTLE[0]:
            break
        time.sleep(TICK)


def stream(op: Op, plan: Plan, grip, watch: GateWatch, tag):
    """Post joint goals along the plan in real time (30 Hz target), constant joint speed per piece, ramped in and
    out over 0.3 s; then wait for the arm to stop. A corner of the motion rule's route (up to sideways, sideways to
    down) is posted exactly and the arm settles there before the next leg: goals 33 ms apart would otherwise cut it.
    Returns the timing and the end reading."""
    plan.rule_check()  # the motion rule, once more, before the first goal goes out
    # every link against every standing tower's keep-out cylinder, for every motion of every process (calib/keepout.py)
    from hillclimb.runner.calib import keepout as KO
    KO.check(op, plan.nodes, tag, opening=float(getattr(plan, "opening", 0.095) or 0.095))
    nodes = plan.nodes
    # plan.fast (Isaiah, Nov 8 11:10: picks 2x, places as they are): its pieces stream at fast x WALK_MAX through the
    # server's /target (each goal straight to the motors; /move walks at most 0.6 rad/s), except the pieces tagged in
    # FAST_SLOW_TAGS (the last 3 cm onto the block), which stay at WALK_MAX on /move so the grasp lands where it did
    fast = float(getattr(plan, "fast", 1.0) or 1.0)
    quick = [fast > 1.0 and tg not in FAST_SLOW_TAGS for (_, _, tg) in nodes[1:]]
    dur = [max(float(np.abs(q1 - q0).max()) / min(v * (fast if qk else 1.0), WALK_MAX * (fast if qk else 1.0)), 1e-3)
           for (q0, _, _), (q1, v, _), qk in zip(nodes[:-1], nodes[1:], quick)]
    T, ta = sum(dur), 0.3
    cum = np.concatenate([[0], np.cumsum(dur)])
    corners = [float(cum[j]) for j in corners_of(op, nodes)]
    # pieces under the motion rule's floor (straight down onto or up off a block next to a stack) go at LOW_V: at
    # speed the joints' different lags swing the fingertips 5-9 mm off the line (sim, Nov 7)
    stacks = rule_stacks(op, plan.exempt())
    for i, ((q0, _, _), (q1, v, _)) in enumerate(zip(nodes[:-1], nodes[1:])):
        a, b = op.tip(q0)[0], op.tip(q1)[0]
        fl, _ = rule_floor(stacks, [a[:2], b[:2]], plan.held_h())
        if max(a[2], b[2]) < fl and v > LOW_V:
            dur[i] = max(float(np.abs(q1 - q0).max()) / LOW_V, 1e-3)
    T = sum(dur)
    cum = np.concatenate([[0], np.cumsum(dur)])
    corners = [float(cum[j]) for j in corners_of(op, nodes)]
    t0 = time.time()
    tau, last = 0.0, t0
    posts = 0
    dwell, t_leg = 0.0, t0 - ta
    while True:
        if not watch.ok:
            _post("/stop", {})
            raise SystemExit(f"GATE during {tag}: {watch.why}: stopped")
        now = time.time()
        t = now - t0 - dwell
        # ramp in and out; slow posting can leave tau short of T when the ramp-out ends: then finish at 30 %
        f = max(0.05, min(1.0, t / ta)) if t < T + ta - ta else max(0.3, min(1.0, (T + ta - t) / ta))
        f = min(f, max(0.05, (now - t_leg) / ta))  # ramped in again after a corner
        tau_new = min(T, tau + f * (now - last))
        corner = next((c for c in corners if tau + 1e-9 < c <= tau_new), None)
        tau = corner if corner is not None else tau_new
        last = now
        i = min(int(np.searchsorted(cum, tau, side="right")) - 1, len(dur) - 1)
        a = (tau - cum[i]) / dur[i]
        q = nodes[i][0] + (nodes[i + 1][0] - nodes[i][0]) * a
        v = nodes[i + 1][1]
        cmd = op.sag.command(q) - op.dq
        if STREAM_HOOK is not None:
            STREAM_HOOK(nodes[i + 1][2], q, float(grip), getattr(plan, "held", None))
        _abort_check(tag)
        if quick[i]:
            _post("/target", {"q": [float(x) for x in cmd], "grip": float(grip)})
        else:
            _post("/move", {"q": [float(x) for x in cmd], "grip": float(grip), "rate": float(min(1.6, 1.3 * min(v, WALK_MAX)))})
        posts += 1
        if corner is not None and tau < T:
            t_c = time.time()
            settle(op, watch, tag)
            dwell += time.time() - t_c
            last = t_leg = time.time()
            continue
        if tau >= T:
            break
        time.sleep(max(0.0, TICK - (time.time() - now)))
    t_stream = time.time() - t0
    while arm_state()["moving"]:
        _abort_check(tag)
        if not watch.ok:
            _post("/stop", {})
            raise SystemExit(f"GATE during {tag}: {watch.why}: stopped")
        if time.time() - t0 > T + 10:
            _post("/stop", {})
            raise SystemExit(f"{tag}: the arm did not reach the end in time: stopped")
        time.sleep(0.03)
    op.st["target"] = nodes[-1][0].tolist()
    return {"planned_s": round(T + ta, 2), "streamed_s": round(t_stream, 2), "done_s": round(time.time() - t0, 2), "posts": posts}


def end_record(op: Op, plan: Plan, tag, direction, run, settle=0.25):
    time.sleep(settle)
    s = arm_state()
    q_t = plan.nodes[-1][0]
    q_m = np.array(s["q"]) + op.dq
    tip_t, tip_m = op.tip(q_t)[0], op.tip(q_m)[0]
    rec = {"run": run, "tag": tag, "direction": direction, "q_target_model": q_t, "q_cmd_sent": op.sag.command(q_t) - op.dq,
           "q_goal_server": s["goal"], "q_measured": s["q"], "joint_err_deg": np.degrees(q_m - q_t).round(2),
           "tip_target_mm": (tip_t * 1000).round(1), "tip_measured_mm": (tip_m * 1000).round(1),
           "tip_err_mm": ((tip_m - tip_t) * 1000).round(1), "grip": s["grip"]}
    record("move_end", **rec)
    return tip_m, s


def fast_look(op: Op, tag="F-look"):
    """From the arm's present pose: overhead blocks (arm folded: nothing hidden), then to the high look pose, one
    wrist frame: blue on the table, purple on green (its bottom at the green's top)."""
    watch = GateWatch()
    oh, _, _ = fetch("overhead")
    (op.run / f"{tag}-rest-overhead.jpg").write_bytes(oh)
    ohb = overhead_blocks(oh)
    q0 = np.array(arm_state()["q"]) + op.dq
    pl = Plan(op, q0, 0.08)
    pl.to(LOOK_TIP[:3], LOOK_TIP[3], 0.4, "look")
    for (a, _, _), (b, _, _) in zip(pl.nodes[:-1], pl.nodes[1:]):
        ok, why = op.safe.check_path(a, b, 0.08, home=True)
        if not ok:
            raise SystemExit(f"look refused: {why}")
    t = stream(op, pl, 0.8, watch, "look")
    end_record(op, pl, "look", "unfold", op.run.name, settle=0.4)
    watch.stop = True
    img, q, grip, stem = op.snap(tag)
    flat = Shape((FACE, FACE, H), TABLE_Z)
    top = Shape((FACE, FACE, H), TABLE_Z + H)
    fits = {}
    for c, shape in (("blue", flat), ("purple", top), ("green", flat)):
        b = op.st["blocks"].get(c)
        g = (b["x"], b["y"], b["yaw"]) if b else None
        fits[c] = (op.view.find(img, c, q, grip * 0.1, shape, g), shape)
    cv2.imwrite(str(op.run / f"{stem}-ov.jpg"), op.overlay(img, q, grip * 0.1, fits, stem))
    out = {c: f for c, (f, _) in fits.items()}
    record("look", run=op.run.name, stem=stem, wrist={c: f for c, f in out.items()}, overhead_at_rest=ohb,
           q_measured=(q - op.dq), tip_mm=(op.tip(q)[0] * 1000).round(1))
    return out, ohb, t


def fast_go(op: Op, held, pick, tower, tower_top, grasp_ref=None, pick_levels=0):
    """pick: (x, y, yaw) of the block on the table (wrist look); tower: (x, y, yaw) of the tower's top block;
    tower_top: its top face height. One streamed motion to grip, close, lift; one to the tower, set, open, back off."""
    run = op.run.name
    watch = GateWatch()
    T0 = time.time()
    times = {}
    bx, by, byaw = pick
    jy = math.degrees(byaw)
    cmd_xy = np.array([bx, by]) - PICK_DOWN[:2]
    # pick_levels: blocks under the one picked (0: on the table). Off a stack the real fingertips ended 10 mm lower on
    # the block than planned (4.7 mm over its bottom, Nov 7, from=2), so aim 10 mm higher there: then the block is held
    # as a table pick holds it and a set-down on the table keeps the jaws clear of the table rule
    z_grip = GRIP_TIP - PICK_DOWN[2] + pick_levels * H + (0.010 if pick_levels else 0.0)
    q0 = np.array(op.st["target"]) if op.st["target"] else np.array(arm_state()["q"]) + op.dq
    pl = Plan(op, q0, 0.08)
    pl.to([cmd_xy[0], cmd_xy[1], TABLE_Z + (pick_levels + 1) * H + 0.08], jy, 0.5, "over")
    pl.to([cmd_xy[0], cmd_xy[1], z_grip + 0.03], jy, 0.3, "down", line=0.01)
    pl.to([cmd_xy[0], cmd_xy[1], z_grip], jy, 0.2, "last 3 cm", line=0.005)
    pl.check(skip=held)
    t_plan = time.time() - T0
    times["pick_stream"] = stream(op, pl, 0.8, watch, "pick")
    tip_m, s = end_record(op, pl, "grip height", "down", run, settle=0.2)
    # close, wait for the jaws to stop
    _post("/move", {"q": s["goal"], "grip": 0.44, "rate": 0.1})
    t_c = time.time()
    g_prev, still = None, 0
    while time.time() - t_c < 1.5:
        g = arm_state()["grip"]
        still = still + 1 if g_prev is not None and abs(g - g_prev) < 0.002 else 0
        g_prev = g
        if still >= 3 and time.time() - t_c > 0.3:
            break
        time.sleep(0.05)
    s = arm_state()
    q_m = np.array(s["q"]) + op.dq
    tip_grasp, yaw_grasp = op.tip(q_m)
    times["close_s"] = round(time.time() - t_c, 2)
    grasp = {"run": run, "block": held, "grip": s["grip"], "tip_fk_mm": (tip_grasp * 1000).round(1), "jaw_yaw_deg": round(yaw_grasp, 1),
             "wrist_look": {"x": bx, "y": by, "yaw": byaw}, "q_measured": s["q"]}
    if s["grip"] < 0.46:
        record("grasp_failed", **grasp)
        _post("/move", {"q": s["goal"], "grip": 0.8, "rate": 0.1})
        time.sleep(0.6)
        raise SystemExit(f"grip closed to {s['grip']:.3f} (no block between the jaws): opened, stopped at grip height")
    # the held block in the wrist frame, fetched while the lift streams
    import threading
    held_frame = {}
    threading.Thread(target=lambda: held_frame.update(data=fetch("wrist")), daemon=True).start()
    # fingertips over the picked block's own bottom (it sat on pick_levels blocks): Nov 7, measuring this from the
    # table let a block picked off a stack go 3.5-7 cm above its place
    grasp_h = tip_grasp[2] - (TABLE_Z + pick_levels * H)
    # place: over the tower, straight down until the block presses the surface (2 mm into the foam), then open:
    # the block is set down, never dropped
    tx, ty, tyaw = tower
    set_tip = tower_top - 0.002 + grasp_h
    pxy = np.array([tx, ty]) - PLACE_DOWN[:2]
    z_set = set_tip - PLACE_DOWN[2]
    pl2 = Plan(op, np.array(op.st["target"]), 0.044)
    pl2.R = pl.R
    tip_c = op.tip(pl.nodes[-1][0])[0]  # the commanded grip point: lift straight up from it
    pl2.to(tip_c + [0, 0, 0.03], jy, 0.25, "lift", line=0.01, keep=True)
    pl2.to([tip_c[0], tip_c[1], CARRY_Z], jy, 0.5, "up", line=0.03, keep=True)
    pl2.to([pxy[0], pxy[1], CARRY_Z], math.degrees(tyaw), 0.5, "carry")
    pl2.to([pxy[0], pxy[1], z_set + 0.03], math.degrees(tyaw), 0.3, "lower", line=0.01, keep=True)
    pl2.to([pxy[0], pxy[1], z_set + 0.01], math.degrees(tyaw), 0.15, "near", line=0.005, keep=True)
    pl2.to([pxy[0], pxy[1], z_set], math.degrees(tyaw), 0.06, "set on the surface", line=0.002, keep=True)
    pl2.check(skip=held, floor={"carry": CARRY_Z - 0.015})
    times["place_stream"] = stream(op, pl2, 0.44, watch, "place")
    tip_set, s = end_record(op, pl2, "set height", "down", run, settle=0.2)
    _post("/move", {"q": s["goal"], "grip": 0.8, "rate": 0.1})
    time.sleep(0.35)
    q_rel = np.array(arm_state()["q"]) + op.dq
    tip_rel = op.tip(q_rel)[0]
    pl3 = Plan(op, np.array(op.st["target"]), 0.08)
    pl3.R = pl2.R
    pl3.to(op.tip(pl2.nodes[-1][0])[0] + [0, 0, 0.05], math.degrees(tyaw), 0.3, "back off", line=0.01, keep=True)
    pl3.check(skip=held)
    times["back_stream"] = stream(op, pl3, 0.8, watch, "back off")
    watch.stop = True
    T1 = time.time()
    times["block_s"] = round(T1 - T0, 2)
    times["plan_s"] = round(t_plan, 2)
    # the held block as the wrist saw it after the close: the truth across the pads
    held_fit = None
    if "data" in held_frame:
        data, jq, jg = held_frame["data"]
        (op.run / "F-held-wrist.jpg").write_bytes(data)
        (op.run / "F-held.json").write_text(json.dumps({"wrist": {"joints": jq, "grip": jg}}))
        img = cv2.resize(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), (640, 360), interpolation=cv2.INTER_AREA)
        qh = np.array(jq) + op.dq
        tip_h = op.tip(qh)[0]
        held_fit = op.view.find(img, held, qh, jg * 0.1, Shape((FACE, FACE, H), tip_h[2] - grasp_h), None)
        if held_fit:
            held_fit["minus_tip_mm"] = [round((held_fit["x"] - tip_h[0]) * 1000, 1), round((held_fit["y"] - tip_h[1]) * 1000, 1)]
    grasp["held_in_wrist"] = held_fit
    truth = tip_grasp[:2] + (np.array(held_fit["minus_tip_mm"]) / 1000 if held_fit else 0)
    grasp["truth_mm"] = (truth * 1000).round(1)
    grasp["wrist_err_mm"] = ((np.array([bx, by]) - truth) * 1000).round(1)
    if grasp_ref:
        grasp["overhead"] = grasp_ref
        grasp["overhead_err_mm"] = ((np.array([grasp_ref["x"], grasp_ref["y"]]) - truth) * 1000).round(1)
    grasp["grasp_h_mm"] = round(grasp_h * 1000, 1)
    record("grasp", **grasp)
    return times, grasp, tip_set, tip_rel


def fast_check(op: Op, held, tower, tag="F-check"):
    """One photo from over the tower, 7 cm back (the top fully in view): the placed block's offset from the tower."""
    watch = GateWatch()
    pl = Plan(op, np.array(op.st["target"]), 0.08)
    pl.to([tower[0] - 0.065, tower[1], 0.13], math.degrees(tower[2]), 0.4, "check")
    pl.check()
    stream(op, pl, 0.8, watch, "check")
    watch.stop = True
    time.sleep(0.3)
    img, q, grip, stem = op.snap(tag)
    shape = Shape((FACE, FACE, H), TABLE_Z + 2 * H)
    fit = op.view.find(img, held, q, grip * 0.1, shape, (tower[0], tower[1], tower[2]))
    cv2.imwrite(str(op.run / f"{stem}-ov.jpg"), op.overlay(img, q, grip * 0.1, {held: (fit, shape)}, stem))
    return fit, stem


def report():
    rows = [json.loads(x) for x in open(RECORDS)]
    moves = [r for r in rows if r["kind"] == "move_end"]
    by = {}
    for r in moves:
        by.setdefault(r["direction"], []).append(r["tip_err_mm"])
    print("command vs actual, fingertips (measured - target, mm):")
    for k, v in by.items():
        v = np.array(v)
        print(f"  {k}: {len(v)} moves, mean {v.mean(0).round(1).tolist()}, sd {v.std(0).round(1).tolist()}")
    for r in rows:
        if r["kind"] == "grasp":
            print(f"  grasp {r['block']}: truth {r.get('truth_mm')} mm, wrist err {r.get('wrist_err_mm')}, overhead err {r.get('overhead_err_mm')}")
        if r["kind"] == "place":
            print(f"  place {r['block']}: offset {r.get('offset_mm')} mm, yaw {r.get('yaw_off_deg')} deg")


def main():
    run = pathlib.Path(sys.argv[1])
    cmd, a = sys.argv[2], sys.argv[3:]
    if cmd == "gate":
        print(gate())
        return
    op = Op(run)
    if cmd == "snap":
        _, q, grip, stem = op.snap(a[0])
        print(stem, "tip", np.round(op.tip(q)[0] * 1000).tolist(), "grip", round(grip, 3))
    elif cmd == "find":
        fits, tip, stem = op.find_all(a[0])
        print(stem, "tip", np.round(tip[0] * 1000).tolist(), "yaw", round(tip[1], 1))
        for c, f in fits.items():
            print(" ", c, f)
        op.log(a[0], saw=json.dumps(fits))
    elif cmd == "look":
        z0 = float(a[2]) if len(a) > 2 and "=" not in a[2] else TABLE_Z
        occ = next((v.split("=")[1] for v in a[2:] if v.startswith("occlude=")), None)
        fit, tip, stem, grip = op.look(a[0], a[1], z0, occlude=occ)
        print(stem, "tip", np.round(tip[0] * 1000).tolist(), "yaw", round(tip[1], 1), "grip", round(grip, 3))
        print(" ", a[0], fit)
        if fit is not None:
            dx, dy = fit["x"] - tip[0][0], fit["y"] - tip[0][1]
            dyaw = (math.degrees(fit["yaw"]) - tip[1] + 45) % 90 - 45
            print(f"  block minus fingertips: dx {dx * 1000:+.1f} mm, dy {dy * 1000:+.1f} mm, dyaw {dyaw:+.1f} deg")
        op.log(a[1], saw=json.dumps({a[0]: fit, "tip_mm": np.round(tip[0] * 1000).tolist(), "grip": grip}))
    elif cmd == "to":
        x, y, z, yawd, grip, rate, tag = float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]), a[6]
        home = "home" in a[7:]
        skip = next((v.split("=")[1] for v in a[7:] if v.startswith("skip=")), None)
        q, why = op.solve([x, y, z], math.radians(yawd), grip * 0.1)
        if q is None:
            op.log(tag, decided=f"no pose: {why}")
            raise SystemExit(f"no pose: {why}")
        print("q_model", np.round(q, 3).tolist(), "tip", np.round(op.tip(q)[0] * 1000).tolist(), "yaw", round(op.tip(q)[1], 1))
        if "dry" in a[7:]:
            q_meas = np.array(arm_state()["q"]) + op.dq
            q_last = np.array(op.st["target"]) if op.st["target"] else q_meas
            for q0 in (q_last, q_meas):
                print("  path", op.safe.check_path(q0, q, grip * 0.1, home=home), op.extra_checks(q0, q, skip),
                      "max joint change", round(float(np.abs(q - q0).max()), 2))
            return
        res = op.move(q, grip, rate, tag, home=home, skip=skip, note=f"fingertips to {x:.3f} {y:.3f} {z:.3f} yaw {yawd:.0f}")
        print(json.dumps(res))
    elif cmd == "grip":
        q = np.array(op.st["target"] if op.st["target"] else np.array(arm_state()["q"]) + op.dq)
        skip = next((v.split("=")[1] for v in a[2:] if v.startswith("skip=")), None)
        res = op.move(q, float(a[0]), 0.1, a[1], skip=skip, note=f"grip {a[0]}")
        print(json.dumps(res))
    elif cmd == "streamtest":
        dx, v = float(a[0]), float(a[1])
        watch = GateWatch()
        q0 = np.array(op.st["target"])
        p0, yaw = op.tip(q0)
        pl = Plan(op, q0, 0.08)
        pl.to(p0 + [0, dx, 0], yaw, v, "side", line=0.01)
        pl.to(p0, yaw, v, "back", line=0.01, keep=True)
        pl.check()
        t = stream(op, pl, 0.8, watch, "test")
        tip_m, s = end_record(op, pl, "streamtest", "level", op.run.name, settle=0.3)
        watch.stop = True
        print(t, "tip target", np.round(p0 * 1000, 1).tolist(), "measured", np.round(tip_m * 1000, 1).tolist())
    elif cmd == "report":
        report()
    elif cmd == "fastlook":
        out, ohb, t = fast_look(op)
        print(json.dumps({"wrist": out, "overhead_at_rest": ohb, "time": t}, default=float))
        op.save()
    elif cmd == "fastgo":
        held = a[0]
        pick = tuple(float(v) for v in a[1:4])
        tower = tuple(float(v) for v in a[4:7])
        kw = dict(v.split("=", 1) for v in a[7:] if "=" in v and not v.startswith("{"))
        ref = next((json.loads(v) for v in a[7:] if v.startswith("{")), None)
        # top=<blocks under the place spot> (0: on the table), from=<blocks under the picked block>
        top = TABLE_Z + int(kw.get("top", 2)) * H
        times, grasp, tip_set, tip_rel = fast_go(op, held, pick, tower, top, ref, pick_levels=int(kw.get("from", 0)))
        op.save()
        print(json.dumps({"times": times, "grasp": grasp, "tip_set_mm": tip_set * 1000, "tip_release_mm": tip_rel * 1000}, default=lambda v: np.asarray(v).tolist()))
    elif cmd == "fastcheck":
        held = a[0]
        tower = tuple(float(v) for v in a[1:4])
        fit, stem = fast_check(op, held, tower)
        op.save()
        off = None if fit is None else [round((fit["x"] - tower[0]) * 1000, 1), round((fit["y"] - tower[1]) * 1000, 1)]
        yo = None if fit is None else round((math.degrees(fit["yaw"] - tower[2]) + 45) % 90 - 45, 1)
        record("place", run=op.run.name, block=held, tower=tower, fit=fit, offset_mm=off, yaw_off_deg=yo, stem=stem)
        print(stem, fit, "offset mm", off, "yaw off deg", yo)
    elif cmd == "sheet":
        print(sheet(op, a))
    elif cmd == "joints":
        rate, tag = float(a[0]), a[1]
        q = np.array([float(v) for v in a[2:8]]) + op.dq
        home = True
        grip = float(a[8]) if len(a) > 8 else float(arm_state()["goal_grip"])
        res = op.move(q, grip, rate, tag, home=home, note="straight to joints", raw=True)
        print(json.dumps(res))


if __name__ == "__main__":
    main()
