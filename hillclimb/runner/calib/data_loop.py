"""Pick and place data collection on the real left Metal arm: scatter the three foam blocks, restack them, again.

Built on calib/opstack.py's look-once, move-once routine (Op, Plan, stream, GateWatch, end_record, the gate, the sag
compensation, the joint offsets) and its pick and place numbers (PICK_DOWN, PLACE_DOWN, GRIP_TIP, CARRY_Z, LOOK_TIP).
The pick and the place are split into their own streamed motions so each is its own episode.

One cycle (the arm shared through runs/station/ARM_OWNER, taken with mkdir):
  1. one wrist look from the high pose: every green, blue and purple blob fitted as a flat block at table level, one
     and two blocks up; the levels chosen so each block above the table sits on one below it (2 cm) and no two share
     a level within 4.5 cm; other coloured blobs (sticks, balls) and tag cubes are obstacles.
  2. scatter: each block, top first, to a random free spot (x 25-42 cm, y -10..25 cm, any heading, 11 cm between
     block centres, 8 cm from every obstacle, in view of the look pose).
  3. restack: three flat at a random free spot, random colour order.
  4. a check photo after each place (from 6.5 cm back): more than 1 cm off or tipped gets one recovery.
Every path: TowerSafety and opstack's extra checks (Plan.check), plus fingertips (and a held block's bottom) 12 mm
over every block they pass within 8.5 cm of.

Recording: calib/loop_rec.py on the station keeps 30 Hz ticks (both cameras, joints, goals); each pick and each
place is saved as one LeRobot episode (dataset/ or failures/) under ~/pah/loop/<session>/ on the station and synced
to runs/station/loop/<session>/; labels.jsonl beside each; calibration records in calib_records.jsonl.

    python -m calib.data_loop run [CYCLES]
    python -m calib.data_loop report SESSION
"""
from __future__ import annotations

import http.client
import itertools
import json
import math
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

from hillclimb.runner.calib import opstack as O
from hillclimb.runner.calib.opstack import (CARRY_Z, GRIP_TIP, LOOK_TIP, PICK_DOWN, PLACE_DOWN, TABLE_Z, GateWatch, Op, Plan, _post,
                           arm_state, end_record, fetch, gate, overhead_blocks, stream)
from hillclimb.runner.calib.tower_plan import FACE, H, WORKSPACE, Shape, frame_down

ROOT = O.ROOT
_RUNS_ROOT = pathlib.Path(__import__("os").environ.get("HILLCLIMB_RUNS", str(pathlib.Path.home() / "hillclimb_runs")))
STATION = _RUNS_ROOT / "station"
TOKEN = STATION / "ARM_OWNER"
LOOP = STATION / "loop"
STATUS = LOOP / "STATUS.txt"
STOPFILE = LOOP / "STOP"
REC = "http://127.0.0.1:8697"
AREA = ((0.22, 0.45), (-0.14, 0.30))  # as wide as the safety checks and IK allow (y over -15 cm)
GAP = 0.08  # block centres apart (blocks are 5 cm), with the jaws checked against every neighbour (finger_clear)
CLEAR = 0.06  # from sticks, balls and other non-block objects
FINGER = 0.045  # an open finger's distance from the block centre, along either face axis
CANDIDATES = 60
OFF_TOL = 0.010  # table set-downs: spot error allowed
NEAT_MM, NEAT_DEG = 4.0, 4.0  # stack set-downs, relative to the measured block below: worse gets one lift and re-place
# Isaiah, Nov 7 night: a stack place more than 8 mm or 10 deg off the block below is re-placed before anything goes on
# top (up to REPLACE_TRIES re-places; then it comes off the stack)
REPLACE_MM, REPLACE_DEG, REPLACE_TRIES = 8.0, 10.0, 2
# the standing check (after every place on a stack, before every place onto one): the top block fitted at its believed
# level and at table level (flat and on its side), in the wrist frame and the overhead frame; standing only when the
# believed level clearly wins
STAND_W_IOU = 0.6  # wrist: the believed-level fit at least this good ...
STAND_MARGIN = 0.05  # ... and this much better than the best table-level fit ...
STAND_XY = 0.015  # ... and within this of where the block is believed to be
STAND_XY_PLACED = 0.025  # after a place: it stands anywhere on the block under (the re-place rule judges the offset)
STAND_OH_IOU = 0.5  # overhead: the believed-level fit at least this good, within STAND_OH_XY
STAND_OH_XY = 0.02
STAND_OH_OVERLAP = 0.25  # a table-level explanation overlapping the block below by more than this is not possible
CARRY = (0.13, 0.17)  # carry height, randomised per block
IOU_MIN = 0.70
GRIP_CLOSE = 0.30  # close command: foam squeezes fine; empty jaws read about 0.30, a 34.5 mm grip about 0.36, 50 mm 0.52
GRIP_HELD = 0.32
COLOURS = ("green", "blue", "purple")
GOOD = ((0.3667, -0.0592), (0.268, -0.006), (0.387, -0.073), (0.330, 0.140))  # spots that worked on Nov 7
STACK_PHRASES = ("place the {a} block on the {b} block", "stack the {a} block on top of the {b} block")


if "Fatal" not in globals():  # kept across a hot reload
    class Fatal(Exception):
        pass


if "CycleFail" not in globals():  # kept across a hot reload
    class CycleFail(Exception):
        pass


if "NoSpot" not in globals():  # kept across a hot reload
    class NoSpot(CycleFail):
        pass


if "NoPlan" not in globals():  # kept across a hot reload
    class NoPlan(CycleFail):
        pass


if "TowerFell" not in globals():  # kept across a hot reload
    class TowerFell(CycleFail):
        """The standing check says a stack's top block is not at its believed level."""
        def __init__(self, msg, bid=None, table_fit=None):
            super().__init__(msg)
            self.bid, self.table_fit = bid, table_fit


def status(line):
    msg = f"{time.strftime('%H:%M:%S')} {line}"
    with open(STATUS, "a") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)


def rec(path, **q):
    url = f"{REC}{path}" + (("?" + urllib.parse.urlencode(q)) if q else "")
    return json.loads(urllib.request.urlopen(url, timeout=60).read())


def now():
    return rec("/now")["t"]


def jdump(v):
    return json.loads(json.dumps(v, default=lambda x: np.asarray(x).tolist()))


def take_token():
    while True:
        try:
            TOKEN.mkdir()
            break
        except FileExistsError:
            time.sleep(5)
    (TOKEN / "who.txt").write_text(f"data-loop {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def release_token():
    shutil.rmtree(TOKEN, ignore_errors=True)


def arm_state_retry(tries=20):
    for i in range(tries):
        try:
            return arm_state()
        except (OSError, http.client.HTTPException) as e:
            if i == tries - 1:
                raise Fatal(f"arm server unreadable: {e!r}")
            time.sleep(1.0)


# streamed joint speeds (rad/s): high up, lowering, the last 3 cm and the set-down (Nov 7 20:40: 1.5x the 18:45
# values; the server's walk caps every goal at 0.6 rad/s, opstack.stream)
V_HIGH, V_DOWN, V_LAST, V_SET = 0.6, 0.45, 0.3, 0.09
REACH_Z = 0.238  # highest legal fingertip height, fingers down (Kin + TowerSafety sweep, x 16-38 cm)
LOOK_MAX_Z = 0.22  # close looks from 6.5 cm back stop here (the camera rule tightens near the reach limit)


def bot(b):
    """A block's bottom: its measured z0, else its level of flat blocks."""
    return b["z0"] if b.get("z0") is not None else TABLE_Z + b["level"] * H


# the big foam bricks (Nov 8: one green, one orange): 83 mm long, as wide and as tall as the small blocks (wrist fits
# 81-83 x 46-49 mm; heights read 31-42 against 28-37 for the 34.5 mm small ones, so the same 34.5 is kept)
BIG_L = 0.083


def hgt(b):
    if b.get("upright"):
        return BIG_L  # a brick standing on its end: 83 mm tall
    return FACE if b.get("lying") else H  # on its side: 50.3 mm tall, hole across


def half_of(b):
    """A block's footprint half extents (along its yaw, across it), m."""
    if b.get("big") and b.get("upright"):
        return (FACE / 2, H / 2)
    if b.get("big"):
        return (BIG_L / 2, (H if b.get("lying") else FACE) / 2)
    return (H / 2, FACE / 2) if b.get("lying") else (FACE / 2, FACE / 2)


def ext(b):
    """How far a block reaches past a small flat one's half face (m): 0 for the small blocks."""
    return max(half_of(b)) - FACE / 2


def footprint_gap(a, b):
    """Separating-axis gap (m) between two footprints (x, y, yaw, half along, half across): over 0 apart, under 0 overlap."""
    def axes(f):
        c, s_ = math.cos(f[2]), math.sin(f[2])
        return np.array([[c, s_], [-s_, c]])
    A, B = axes(a), axes(b)
    d = np.array([b[0] - a[0], b[1] - a[1]])
    ha, hb = np.array(a[3:5]), np.array(b[3:5])
    return max(abs(d @ n) - np.abs(A @ n) @ ha - np.abs(B @ n) @ hb for n in (*A, *B))


def size_word(b):
    return "big" if b.get("big") else "small"


def top_of(b):
    return bot(b) + hgt(b)


def shape_of(b, z0=None):
    z = bot(b) if z0 is None else z0
    if b.get("big") and b.get("upright"):  # a brick on its end: 50.3 x 34.5 footprint, 83 tall
        return Shape((FACE, H, BIG_L), z, period=math.pi)
    if b.get("big"):  # a brick: only a half turn is the same shape
        return Shape((BIG_L, H, FACE), z, period=math.pi) if b.get("lying") else Shape((BIG_L, FACE, H), z, period=math.pi)
    return Shape((H, FACE, FACE), z, period=math.pi) if b.get("lying") else Shape((FACE, FACE, H), z)


def look_z(level):
    """Fingertip height for a close look at a block at this level from 6.5 cm back: 13 cm, or 6 cm over its top."""
    return max(0.13, TABLE_Z + (level + 1) * H + 0.06)


TUNNELS = ()  # on the station the team page and Dream caller are local: no tunnels
STATION_HOST = __import__("os").environ.get("STATION_HOST", "pah@nuck-1-pah")  # Tailscale (pah@10.103.0.132 on the event LAN)


def reopen_tunnels():
    """The team page and Dream caller tunnels to the station, re-opened (they die now and then; the arm's own 8691
    tunnel is left alone)."""
    import subprocess
    for t in TUNNELS:
        subprocess.run(["pkill", "-f", f"ssh .*-L {t}"], check=False)
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-f", "-N", "-L", t, STATION_HOST],
                       check=False, timeout=20)
    status("gate unreadable for 10 s: re-opened the 18081 and 18084 tunnels")


def wait_gate():
    t0 = time.time()
    t_unread = None
    while True:
        ok, why = gate()
        if ok:
            return
        if why.startswith("could not read"):
            t_unread = t_unread or time.time()
            if time.time() - t_unread > 10:
                reopen_tunnels()
                t_unread = None
        else:
            t_unread = None
        if time.time() - t0 > 600:
            raise Fatal(f"gate closed for 10 minutes: {why}")
        print("gate closed, waiting:", why, flush=True)
        time.sleep(2)


def health():
    s = arm_state()
    if s.get("error"):
        raise Fatal(f"arm server error: {s['error']}")
    hot = []

    def walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{path}.{k}")
        elif isinstance(d, (int, float)) and "temp" in path.lower() and d > 60:
            hot.append((path, d))
    walk(s)
    if hot:
        raise Fatal(f"motor temperature over 60 C: {hot}")
    return s


def full_colour_mask(bgr, m, jaws=None):
    """m (a block's saturated colour mask) grown by the pale, lit part of the same block: pixels of its own hue (the
    blob's median, within 16: a lit blue face reads cyan), saturation from 18, value from 60, in components touching m."""
    if not m.any():
        return m
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h0 = float(np.median(hsv[..., 0][m > 0]))
    dh = np.abs(((hsv[..., 0].astype(float) - h0) + 90) % 180 - 90)
    # the table reads hue 105-115 at saturation 3-16 (Nov 8): a blue block's growth would flood it (3.5k px to 28k), so
    # blue grows only from saturation 35
    p = (dh <= 16) & (hsv[..., 1] >= (35 if 95 <= h0 <= 125 else 18)) & (hsv[..., 2] >= 60)
    if jaws is not None:
        p &= jaws == 0
    n, lab = cv2.connectedComponents(p.astype(np.uint8))
    touch = set(np.unique(lab[(cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & p]).tolist()) - {0}
    out = (np.isin(lab, list(touch)) | (m > 0)).astype(np.uint8)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


_USB_GATE = []


def usb_gate_wait():
    """Hold the first pick or push until GATE_FILE exists (no env: no wait)."""
    import os as _os, time as _wall
    f = _os.environ.get("GATE_FILE")
    if not f or _USB_GATE:
        return
    t0 = _wall.time()
    while not _os.path.exists(f):
        _wall.sleep(0.1)
    _USB_GATE.append(1)
    status(f"GATE open after waiting {_wall.time() - t0:.1f} s at the first pick or push")


class Loop:
    relook_iou = 0.75  # a re-look must fit this well
    def __init__(self, session: pathlib.Path):
        self.dir = session
        session.mkdir(parents=True, exist_ok=True)
        O.RECORDS = session / "calib_records.jsonl"
        self.op = Op(session / "op")
        self.rng = np.random.default_rng(int(time.time()))
        self.belief = {}  # colour -> dict(x, y, yaw, level, iou)
        self.obstacles = np.zeros((0, 2))
        self.q_look = None
        self.jaws_look = None
        self.saves = queue.Queue()
        self.nstack = 0
        self.counts = {"pick": [0, 0], "place": [0, 0]}  # [success, failure]
        self.dr = {"carry_z": CARRY_Z, "over_dz": 0.0}
        self.cyc = {}
        self.bias = self.load_bias()
        self.block_times = []
        threading.Thread(target=self.saver, daemon=True).start()

    # ---- place bias, learned online per level (arm frame): the mean of the last 5 measured (fit - aimed) offsets
    def load_bias(self):
        """Raw place errors (measured minus aimed, mm, arm frame) with where they happened, per level, from every loop
        session's calib_records."""
        hist = {0: [], 1: [], 2: []}
        import glob as _g, os as _os
        extra = [pathlib.Path(p) for pat in _os.environ.get("BIAS_FILES", "").split(":") if pat for p in sorted(_g.glob(pat))]
        for f in sorted(LOOP.glob("*/calib_records.jsonl")) + extra:
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get("kind") == "place" and r.get("offset_mm") and r.get("target") and r.get("fit"):
                    corr = r.get("correction_mm") or [0.0, 0.0]
                    lvl = int(r["target"][3])
                    if r["fit"].get("level", lvl) == lvl:
                        hist.setdefault(lvl, []).append([r["target"][0], r["target"][1], r["offset_mm"][0] + corr[0], r["offset_mm"][1] + corr[1]])
        return hist

    def correction(self, level, x=0.33, y=0.07):
        """Fewer than 3 samples at this level: none. 3 to 5: half their mean. 6 or more: offset = a + B [x, y], least
        squares regularised toward 0 (ridge, lambda 2 on the scaled features). Clipped to 8 mm."""
        h = np.array(self.bias.get(level) or [], float).reshape(-1, 4)
        if len(h) < 3:
            return [0.0, 0.0]
        if len(h) < 6:
            return [float(np.clip(v, -8.0, 8.0)) for v in 0.5 * h[:, 2:].mean(0)]
        h = h[-20:]
        F = np.stack([np.ones(len(h)), (h[:, 0] - 0.33) / 0.1, (h[:, 1] - 0.07) / 0.1], 1)
        W = np.linalg.solve(F.T @ F + 2.0 * np.eye(3), F.T @ h[:, 2:])
        out = np.array([1.0, (x - 0.33) / 0.1, (y - 0.07) / 0.1]) @ W
        return [float(np.clip(v, -8.0, 8.0)) for v in out]

    # ---- recording
    def saver(self):
        while True:
            job = self.saves.get()
            try:
                root = job.get("root") or f"{self.dir.name}/" + ("failures" if not job["label"]["success"] else "dataset")
                r = rec("/save", t0=job["t0"], t1=job["t1"], root=root, task=job["label"]["prompt"])
                lab = dict(job["label"], dataset=root, **{k: r.get(k) for k in ("episode_index", "frames", "seconds", "max_gap_s", "error")})
                d = LOOP / root
                d.mkdir(parents=True, exist_ok=True)
                with open(d / "labels.jsonl", "a") as fh:
                    fh.write(json.dumps(jdump(lab)) + "\n")
                if r.get("error"):
                    print("save error:", r, flush=True)
            except Exception as e:  # noqa: BLE001
                print("save failed:", e, flush=True)
            finally:
                self.saves.task_done()

    def episode(self, t0, t1, root=None, **label):
        self.counts.setdefault(label["skill"], [0, 0])[0 if label["success"] else 1] += 1
        self.saves.put({"t0": t0 - 0.3, "t1": t1 + 0.25, "root": root,
                        "label": dict(label, session=self.dir.name, randomisation=dict(self.dr), cycle_setup=dict(self.cyc))})

    def sync(self):
        """Laptop copy-back removed: the runner runs on the station."""
        self.saves.join()

    # ---- belief helpers
    def put_blocks(self):
        self.op.st["blocks"] = {c: {"x": b["x"], "y": b["y"], "yaw": b["yaw"], "z0": top_of(b) - H, "half": max(half_of(b))}
                               for c, b in self.belief.items()}

    def below(self, colour):
        """The colour and every block under it in its stack."""
        b = self.belief[colour]
        return {colour} | {c for c, o in self.belief.items() if bot(o) < bot(b) - 0.005 and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.03}

    def sweep(self, plan, exempt, held_h=0.0):
        for (q0, _, _), (q1, _, tag) in zip(plan.nodes[:-1], plan.nodes[1:]):
            n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / math.radians(2))) + 1)
            for t in np.linspace(0, 1, n):
                p, jyaw = self.op.tip(q0 + (q1 - q0) * t)
                held_big = held_h > 0 and getattr(self, "held_ext", 0.0) > 0
                for c, b in self.belief.items():
                    if c in exempt:
                        continue
                    d = math.hypot(p[0] - b["x"], p[1] - b["y"])
                    if d < (0.075 if held_h > 0 else 0.06) + ext(b) + (0.0165 if held_big else 0.0):
                        top = top_of(b)
                        if p[2] - held_h < top + 0.012:
                            # a held brick (long side across the jaws): its own footprint against the block's, 15 mm
                            # apart is clear (the round radius alone refused a stack 9.3 cm from a neighbour, Nov 8 01:23)
                            if held_big and footprint_gap((p[0], p[1], math.radians(jyaw) + math.pi / 2, BIG_L / 2, FACE / 2),
                                                          (b["x"], b["y"], b.get("yaw", 0.0), *half_of(b))) >= 0.015:
                                continue
                            raise SystemExit(f"sweep refused at {tag}: {(p[2] - held_h - top) * 1000:.0f} mm over the {c} block")

    # ---- looking
    def fits_all(self, img, q, grip):
        """Every candidate blob of each block colour fitted flat at levels 0, 1, 2."""
        view = self.op.view
        opening = grip * 0.1
        cam = view.pose(q, opening)
        jaws = view.jaws_mask(q, opening)
        cands = {}
        leftover = []
        for c in COLOURS:
            m = view.colour_mask(img, c, jaws)
            n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
            comps = sorted([i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 150], key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:3]
            cands[c] = []
            for i in comps:
                for lvl in range(3):
                    shape = Shape((FACE, FACE, H), TABLE_Z + lvl * H)
                    f = view._fit(cam, jaws, lab, stats, i, shape, None)
                    if f is None or f["iou"] < 0.55:
                        continue
                    if not (WORKSPACE[0][0] < f["x"] < WORKSPACE[0][1] and WORKSPACE[1][0] < f["y"] < WORKSPACE[1][1]):
                        continue
                    cands[c].append(dict(f, level=lvl, blob=int(i)))
                ys, xs = np.nonzero(lab == i)
                sel = np.stack([xs, ys], 1)[:: max(1, len(xs) // 200)]
                leftover.append((c, int(i), view.ray_on_plane(cam, sel, TABLE_Z + 0.015)[:, :2]))
        # a colour not seen at all may be hidden under a block seen one level up: that block's pose, one level down
        for c in COLOURS:
            if not cands[c]:
                cands[c] = [dict(A, level=A["level"] - 1, iou=0.0, blob=-1, hidden=True) for o in COLOURS if o != c
                            for A in cands[o] if A["level"] >= 1]
        # other colours (red, orange sticks and balls): all obstacles
        from vision.locate import HUES, SAT_MIN, VAL_MIN
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        obst = []
        for c in ("red", "orange"):
            bins = HUES.get(c, [(10, 24)])
            mm = np.zeros(hsv.shape[:2], bool)
            for lo, hi in bins:
                mm |= (hsv[..., 0] >= lo) & (hsv[..., 0] < hi)
            mm = (mm & (hsv[..., 1] >= SAT_MIN) & (hsv[..., 2] >= VAL_MIN) & (jaws == 0)).astype(np.uint8)
            mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mm)
            for i in range(1, n):
                if stats[i, 4] >= 120:
                    ys, xs = np.nonzero(lab == i)
                    sel = np.stack([xs, ys], 1)[:: max(1, len(xs) // 200)]
                    obst.append(view.ray_on_plane(cam, sel, TABLE_Z + 0.011)[:, :2])
        return cands, leftover, obst, jaws

    def choose(self, cands, prior):
        best = None
        for combo in itertools.product(*[cands[c] for c in COLOURS]):
            byc = dict(zip(COLOURS, combo))
            ok = True
            for a, b in itertools.combinations(COLOURS, 2):
                A, B = byc[a], byc[b]
                d = math.hypot(A["x"] - B["x"], A["y"] - B["y"])
                if A["level"] == B["level"] and d < 0.045:
                    ok = False
            for c, A in byc.items():
                if A["level"] > 0 and not any(B["level"] == A["level"] - 1 and math.hypot(A["x"] - B["x"], A["y"] - B["y"]) < 0.02
                                              for k, B in byc.items() if k != c):
                    ok = False
            if not ok:
                continue
            score = sum(A["iou"] - (0.1 if A["edge"] else 0) for A in byc.values())
            for c, A in byc.items():
                p = (prior or {}).get(c)
                if p and p["level"] == A["level"] and math.hypot(p["x"] - A["x"], p["y"] - A["y"]) < 0.03:
                    score += 0.05
            if best is None or score > best[0]:
                best = (score, byc)
        return None if best is None else best[1]

    def jitter_look(self, dxy=0.03, dyaw=10.0):
        return (LOOK_TIP[0] + self.rng.uniform(-dxy, dxy), LOOK_TIP[1] + self.rng.uniform(-dxy, dxy), LOOK_TIP[2],
                LOOK_TIP[3] + self.rng.uniform(-dyaw, dyaw))

    def look(self, tag, tip=LOOK_TIP):
        self.to_pose(tip, "look")
        time.sleep(0.25)
        img, q, grip, stem = self.op.snap(tag)
        oh = (self.op.run / f"{stem}-overhead.jpg").read_bytes()
        try:
            ohb = overhead_blocks(oh)
        except Exception as e:  # noqa: BLE001
            ohb = {"error": repr(e)}
        cands, leftover, obst, jaws = self.fits_all(img, q, grip)
        byc = self.choose(cands, self.belief)
        if byc is None:
            return None, stem
        used = {(c, A["blob"]) for c, A in byc.items()}
        pts = [p for c, i, p in leftover if (c, i) not in used] + obst
        self.obstacles = np.concatenate(pts) if pts else np.zeros((0, 2))
        self.obstacles = self.obstacles[(self.obstacles[:, 0] > 0.15) & (self.obstacles[:, 0] < 0.55) & (self.obstacles[:, 1] > -0.2)
                                        & (self.obstacles[:, 1] < 0.35)]
        # the blocks' own side faces (a stack's lower blocks show as strips around it) are not obstacles
        for A in byc.values():
            if len(self.obstacles):
                self.obstacles = self.obstacles[np.hypot(self.obstacles[:, 0] - A["x"], self.obstacles[:, 1] - A["y"]) > 0.06]
        self.belief = {c: {k: A[k] for k in ("x", "y", "yaw", "level", "iou", "edge")} for c, A in byc.items()}
        self.put_blocks()
        self.q_look, self.jaws_look = q, jaws
        self.look_overhead = ohb
        fits = {c: (dict(A), Shape((FACE, FACE, H), TABLE_Z + A["level"] * H)) for c, A in byc.items()}
        out = self.op.overlay(img, q, grip * 0.1, fits, stem)
        for p in self.obstacles[::5]:
            uv, ok = self.op.view.project(self.op.view.pose(q, grip * 0.1), [[p[0], p[1], TABLE_Z + 0.011]])
            if ok.all():
                cv2.circle(out, tuple(int(v) for v in uv[0]), 2, (0, 0, 255), -1)
        cv2.imwrite(str(self.op.run / f"{stem}-ov.jpg"), out)
        O.record("look", run=self.dir.name, stem=stem, wrist=self.belief, overhead=ohb, q_measured=(q - self.op.dq),
                 obstacles=len(self.obstacles))
        return self.belief, stem

    # ---- motion
    def start_q(self):
        """The last target, else the arm's reading; when the sagging reading itself breaks a rule but the server's
        command (within 0.2 rad of it) does not, the command: paths are checked from the command (calib/safety.py)."""
        if self.op.st["target"]:
            return np.array(self.op.st["target"])
        s = arm_state()
        q_m, q_g = np.array(s["q"]) + self.op.dq, np.array(s["goal"]) + self.op.dq
        if not self.op.safe.check_pose(q_m, 0.08)[0] and self.op.safe.check_pose(q_g, 0.08)[0] and np.abs(q_g - q_m).max() < 0.2:
            return q_g
        return q_m

    def to_pose(self, tip, tag, v=None):
        """Fingertips to a high pose (look or end), from wherever the arm is."""
        v = V_HIGH if v is None else v
        q0 = self.start_q()
        if np.linalg.norm(self.op.tip(q0)[0] - np.array(tip[:3])) < 0.01:
            return
        pl = Plan(self.op, q0, 0.08)
        pl.to(tip[:3], tip[3], v, tag)
        try:
            pl.check()
        except SystemExit:
            s = arm_state()
            near = [np.abs(np.array(s["q"]) - np.array(s[k])).max() for k in ("start_joints", "zero_joints") if s.get(k)]
            if not near or min(near) > 0.15:  # only the folded rest (start or saved zero) may unfold under the home rules
                raise
            for (a, _, _), (b, _, _) in zip(pl.nodes[:-1], pl.nodes[1:]):
                ok, why = self.op.safe.check_path(a, b, 0.08, home=True)
                if not ok:
                    raise SystemExit(f"{tag} refused: {why}")
        try:
            self.sweep(pl, exempt=set())
        except SystemExit as e:
            # low over a block (a run stopped there): straight up first (only ever away from what is under it), then on
            p0, yaw0 = self.op.tip(q0)
            done = None
            # straight up if the reach allows, else sideways 10 cm at this height (clear of a tall tower), then on
            for way in ([p0[0], p0[1], min(max(p0[2] + 0.04, 0.17), REACH_Z)], [p0[0], p0[1] + 0.10, p0[2]], [p0[0], p0[1] - 0.10, p0[2]],
                        [p0[0] - 0.04, p0[1] + 0.10, p0[2]], [p0[0] - 0.04, p0[1] - 0.10, p0[2]]):
                try:
                    up = Plan(self.op, q0, 0.08)
                    up.to(way, yaw0, 0.3, "clear", line=0.01)
                    up.check()
                    self.sweep(up, exempt=set())
                    pl = Plan(self.op, up.nodes[-1][0], 0.08)
                    pl.to(tip[:3], tip[3], v, tag)
                    pl.check()
                    self.sweep(pl, exempt=set())
                    pl.nodes = up.nodes + pl.nodes[1:]
                    done = way
                    break
                except SystemExit:
                    continue
            if done is None:
                raise
            status(f"{tag}: {e}; clearing by {np.round(np.array(done) * 1000).tolist()} first")
        wait_gate()
        watch = GateWatch()
        try:
            stream(self.op, pl, 0.8, watch, tag)
        finally:
            watch.stop = True
        end_record(self.op, pl, tag, "level", self.dir.name, settle=0.3)

    def plan_pick(self, colour, q0, regrip=0.0):
        b = self.belief[colour]
        jy = math.degrees(b["yaw"])
        cxy = np.array([b["x"], b["y"]]) - PICK_DOWN[:2]
        # off a stack the real fingertips end about 10 mm lower on the block than planned (Nov 7): aim 10 mm higher
        z_grip = GRIP_TIP + (bot(b) - TABLE_Z) - PICK_DOWN[2] + (0.010 if bot(b) > TABLE_Z + 0.005 or b.get("lying") else 0.0) + regrip
        z_over = min(top_of(b) + 0.08 + self.dr["over_dz"], REACH_Z)
        ex = self.below(colour)
        last = None
        for route in ("direct", "high"):
            pl = Plan(self.op, q0, 0.08)
            try:
                if route == "high":
                    p0 = self.op.tip(q0)[0]
                    pl.to([p0[0], p0[1], max(self.dr["carry_z"], p0[2])], self.op.tip(q0)[1], 0.4, "up", line=0.02)
                    pl.to([cxy[0], cxy[1], max(self.dr["carry_z"], z_over)], jy, 0.5, "over high")
                    pl.to([cxy[0], cxy[1], z_over], jy, 0.4, "over", line=0.02, keep=True)
                else:
                    pl.to([cxy[0], cxy[1], z_over], jy, V_HIGH, "over")
                pl.to([cxy[0], cxy[1], z_grip + 0.03], jy, V_DOWN, "down", line=0.01, keep=True)
                pl.to([cxy[0], cxy[1], z_grip], jy, V_LAST, "last 3 cm", line=0.005, keep=True)
                pl.check(skip=colour)
                self.sweep(pl, ex)
                return pl
            except SystemExit as e:
                last = e
        raise SystemExit(f"pick {colour}: {last}")

    def plan_place(self, colour, q0, R, target, grasp_h):
        """target: (x, y, yaw, support_top, support colour or None)."""
        tx, ty, tyaw, top, sup = target
        pxy = np.array([tx, ty]) - PLACE_DOWN[:2]
        z_set = top - 0.002 + grasp_h - PLACE_DOWN[2]  # pressed 2 mm into the surface, then opened: set down, never dropped
        pl = Plan(self.op, q0, 0.044, held=(colour, grasp_h))
        pl.R = R
        # arrive above the target already aligned (x, y, yaw), the held block's bottom clear of the support's top, then
        # only straight down: no sideways motion while the block can touch the tower (Isaiah, Nov 7: sideways contact
        # while aligning is what pushes a tower over). The carry is the motion rule's route (opstack.Plan.route):
        # straight up from the lift until the held block clears every stack near the path by 2 cm, level across, down
        cz = max(self.dr["carry_z"], z_set + 0.04)
        if cz > REACH_Z:
            if z_set + 0.02 > REACH_Z:
                raise SystemExit(f"no straight-down place: aligned point {cz * 1000:.0f} mm is over the reach ({REACH_Z * 1000:.0f} mm)")
            cz = REACH_Z
        # a brick on a brick: their long sides lined up (Isaiah, Nov 8 03:03): the held brick's long side lies across the
        # jaws, so the end frame's jaws are across the support's long side; else any quarter turn that plans
        if (self.belief.get(colour) or {}).get("big") and sup and (self.belief.get(sup) or {}).get("big"):
            last = None
            for a in (self.belief[sup]["yaw"] + math.pi / 2, self.belief[sup]["yaw"] - math.pi / 2):
                try:
                    pa = Plan(self.op, q0, 0.044, held=(colour, grasp_h))
                    pa.R = frame_down(a)
                    pa.to([pxy[0], pxy[1], cz], math.degrees(a), V_HIGH, "carry", keep=True)
                    pl = pa
                    break
                except SystemExit as e:
                    last = e
            else:
                raise SystemExit(f"no carry that lines {colour} up with {sup}: {last}")
        else:
            pl.to([pxy[0], pxy[1], cz], math.degrees(tyaw), V_HIGH, "carry")
        pl.to([pxy[0], pxy[1], z_set + (0.03 if z_set + 0.03 <= REACH_Z else 0.015)], math.degrees(tyaw), V_DOWN, "lower", line=0.01, keep=True)
        pl.to([pxy[0], pxy[1], z_set + 0.01], math.degrees(tyaw), 0.22, "near", line=0.005, keep=True)
        pl.to([pxy[0], pxy[1], z_set], math.degrees(tyaw), V_SET, "set on the surface", line=0.002, keep=True)
        # the carry starts at the lift's top: a block lifted where the reach is low starts the carry lower
        pl.check(skip=colour, floor={"carry": min(self.dr["carry_z"], self.op.tip(q0)[0][2]) - 0.015})
        self.sweep(pl, {colour} | (self.below(sup) if sup else set()), held_h=grasp_h)
        return pl

    def held_for(self, colour):
        """(id, bottom under the fingertips) of the block in the jaws: the measured one while held, else nominal."""
        h = getattr(self.op, "held", None)
        if h and h[0] == colour:
            return h
        if colour is None:
            return None
        return (colour, 0.025 if (self.belief.get(colour) or {}).get("lying") else 0.015)

    def plan_lift(self, q0, R, jy, colour=None):
        pl = Plan(self.op, q0, 0.044, held=self.held_for(colour))
        pl.R = R
        tip_c = self.op.tip(q0)[0]
        pl.to(tip_c + [0, 0, 0.03], jy, 0.35, "lift", line=0.01, keep=True)
        try:
            pl.to([tip_c[0], tip_c[1], self.dr["carry_z"]], jy, V_HIGH, "up", line=0.03, keep=True)
        except SystemExit:
            # far out (x 38 cm) the reach is low: lift what it allows; the carry rises on its way in
            for z in (0.12, 0.10, 0.08):
                try:
                    pl.nodes = pl.nodes[:2]
                    pl.to([tip_c[0], tip_c[1], max(z, tip_c[2] + 0.03)], jy, V_HIGH, "up", line=0.03, keep=True)
                    break
                except SystemExit:
                    continue
        pl.check(skip=colour)
        return pl

    def go(self, pl, grip, tag, direction):
        if tag in ("pick", "push"):
            usb_gate_wait()
        wait_gate()
        health()
        watch = GateWatch()
        try:
            t = stream(self.op, pl, grip, watch, tag)
        finally:
            watch.stop = True
        tip_m, s = end_record(self.op, pl, tag, direction, self.dir.name, settle=0.1)
        return t, tip_m, s

    def release(self, s):
        """Open slowly, partly first, to cut the twist on release."""
        _post("/move", {"q": s["goal"], "grip": 0.6, "rate": 0.1})
        time.sleep(0.12)
        _post("/move", {"q": s["goal"], "grip": 0.8, "rate": 0.1})
        time.sleep(0.12)

    def top_look(self, x, y, yaw, top_z, tag, skip=0):
        """The wrist camera nearly straight over a stack's top face (centre x, y at height top_z): the camera sits
        7.6 cm off the tool point and looks 20 deg back toward the fingers, so the fingertips go beside the stack and
        the camera over it. Every candidate is planned, checked and swept like any move; the one whose camera looks
        closest to straight down at the top face, with the face in view and clear of the jaws, is taken. skip: the
        next-best instead (a second, independent view). True when the arm went there."""
        view = self.op.view
        q0 = np.array(self.op.st["target"]) if self.op.st.get("target") is not None else self.start_q()
        P = np.array([x, y, top_z])
        cands = []
        for yd in (math.degrees(yaw), math.degrees(yaw) + 90.0):
            for ang in range(0, 360, 30):
                for r in (0.07, 0.085):
                    for dz in (0.05, 0.07):
                        tz = min(top_z + dz, LOOK_MAX_Z)
                        if tz < top_z + 0.04:
                            continue
                        tp = [x + r * math.cos(math.radians(ang)), y + r * math.sin(math.radians(ang)), tz]
                        try:
                            pl = Plan(self.op, q0, 0.08)
                            pl.to(tp, yd, 0.4, tag)
                        except SystemExit:
                            continue
                        Rc, pc = view.pose(pl.nodes[-1][0])
                        v = P - pc
                        if v[2] > -0.04:
                            continue
                        ray = v / np.linalg.norm(v)
                        c = ray @ Rc
                        if c[2] <= 0.3 or abs(c[0] / c[2]) > 0.8 or not -0.55 < c[1] / c[2] < 0.35:
                            continue
                        cands.append((math.degrees(math.acos(min(1.0, -ray[2]))), pl, tp, yd))
        cands.sort(key=lambda t: t[0])
        n_ok = 0
        for off, pl, tp, yd in cands[:12]:
            try:
                pl.check()
                self.sweep(pl, exempt=set())
            except SystemExit:
                continue
            if n_ok < skip:
                n_ok += 1
                continue
            self.go(pl, 0.8, tag, "level")
            status(f"top look ({tag}): camera {off:.0f} deg off straight down over ({x * 100:.1f}, {y * 100:.1f}) at {top_z * 1000:.0f} mm")
            return True
        status(f"top look ({tag}): no pose from {len(cands)} candidates; the old look instead")
        return False

    def support_look(self, sup):
        """A fresh close look at the block a place goes onto (6.5 cm back, 6 cm over its top); the look pose's frame
        when that does not plan. A raised block must pass the standing check (TowerFell when it fell; Fatal when two
        frames cannot tell). Any support must be seen at its own level where it is believed to be (else CycleFail:
        something may sit on it); the place is then centred on that fit."""
        b = self.belief[sup]
        raised = bot(b) > TABLE_Z + 0.005
        try:
            if not (raised and self.top_look(b["x"], b["y"], b["yaw"], top_of(b), "support look")):
                pl = Plan(self.op, self.start_q(), 0.08)
                pl.to([b["x"] - 0.065, b["y"], min(max(0.13, top_of(b) + 0.06), LOOK_MAX_Z)], math.degrees(b["yaw"]), 0.4, "support look")
                pl.check()
                self.sweep(pl, exempt=set())
                self.go(pl, 0.8, "support look", "level")
        except SystemExit as e:  # a tall tower near the reach limit: the high look pose's frame instead
            status(f"support look {sup}: no close look ({str(e)[:120]}); a frame from the look pose")
            try:
                self.to_pose(LOOK_TIP, "look")
            except SystemExit as e2:
                raise NoPlan(f"no fresh frame of {sup} before the place: {e2}")
        time.sleep(0.25)
        frame = self.op.snap(f"under-{sup}")
        img, q, grip, stem = frame
        shape = shape_of(b)
        with self.full_mask():
            f = self.op.view.find(img, sup, q, grip * 0.1, shape, (b["x"], b["y"], b["yaw"]))
        cv2.imwrite(str(self.op.run / f"{stem}-ov.jpg"), self.op.overlay(img, q, grip * 0.1, {sup: (f, shape)}, stem))
        if raised:
            v, info, tab = self.standing_twice(sup, b, frame, under=self.under(sup))
            if v == "fallen":
                raise TowerFell(f"{sup} is not standing at {bot(b) * 1000:.0f} mm: {info['why']}", sup, tab)
            if v != "standing":
                raise Fatal(f"cannot tell whether {sup} stands at {bot(b) * 1000:.0f} mm from two frames ({info['why']}): "
                            "stopped so nothing goes on it")
            f = info.get("wrist_level") or f
        if f is None or f["iou"] < 0.6 or math.hypot(f["x"] - b["x"], f["y"] - b["y"]) > 0.02:
            raise CycleFail(f"support look {sup}: not seen at its level where it should be (fit {None if f is None else round(f['iou'], 2)}"
                            f"{'' if f is None else ', ' + str(round(math.hypot(f['x'] - b['x'], f['y'] - b['y']) * 1000)) + ' mm off'}): "
                            "not placing on it")
        d = [round((f["x"] - b["x"]) * 1000, 1), round((f["y"] - b["y"]) * 1000, 1)]
        if f["iou"] >= 0.8:
            b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"], edge=f["edge"])
            self.put_blocks()
        O.record("support_look", run=self.dir.name, block=sup, fit=f, moved_from_last_mm=d, stem=stem)
        return b

    # ---- the standing check
    def full_mask(self):
        """A context: the wrist view's colour masks take in the pale, lit part of a block's face too (its own hue
        within 16, saturation from 18, touching the saturated blob): a top face half washed out by the light fits as
        a smaller, farther block, which reads as a fallen one."""
        import contextlib
        view = self.op.view
        base = view.colour_mask

        def pale(bgr, colour, jaws):
            m = base(bgr, colour, jaws)
            return full_colour_mask(bgr, m, jaws)

        @contextlib.contextmanager
        def ctx():
            view.colour_mask = pale
            try:
                yield
            finally:
                view.colour_mask = base
        return ctx()

    def under(self, bid):
        """The block a raised block sits on (the highest one under it, within 3 cm), or None."""
        b = self.belief[bid]
        cands = [(c, o) for c, o in self.belief.items() if c != bid and top_of(o) <= bot(b) + 0.005 and bot(o) < bot(b) - 0.005
                 and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.03]
        return max(cands, key=lambda co: top_of(co[1]))[1] if cands else None

    @staticmethod
    def _overlap(a, a_half, b, b_half, n=9):
        """The share of footprint a (x, y, yaw; half extents) that lies inside footprint b."""
        ca, sa = math.cos(a["yaw"]), math.sin(a["yaw"])
        cb, sb = math.cos(b["yaw"]), math.sin(b["yaw"])
        g = np.linspace(-1, 1, n)
        u, v = np.meshgrid(g * a_half[0], g * a_half[1])
        px = a["x"] + u * ca - v * sa - b["x"]
        py = a["y"] + u * sa + v * ca - b["y"]
        bu, bv = px * cb + py * sb, -px * sb + py * cb
        return float(np.mean((np.abs(bu) <= b_half[0]) & (np.abs(bv) <= b_half[1])))

    def _possible(self, fit, under):
        """A table-level explanation is physically possible only clear of the block under (it would sit inside it)."""
        if fit is None or under is None:
            return True
        return self._overlap(fit, half_of(dict(fit, lying=False)), under, half_of(under)) <= STAND_OH_OVERLAP

    def standing(self, bid, b, frame, under=None, xy_tol=STAND_XY):
        """Is bid standing at its believed level? b: its believed x, y, yaw, bottom and shape; frame: a fresh (wrist
        image, model joints, grip, stem) from op.snap (its overhead frame saved beside it); under: the block it should
        sit on. The block is fitted at the believed level and at table level (flat and on its side) in both frames.
          wrist: standing when the believed level fits at least STAND_W_IOU, within STAND_XY of where it should be,
                 and STAND_MARGIN better than the table; fallen when a possible table fit (clear of the block under)
                 beats it by STAND_MARGIN, or nothing fits at its level at all; else it cannot tell.
          overhead: the same, with STAND_OH_IOU and STAND_OH_XY; from 3 blocks up its parallax puts a table-level
                 explanation of the same blob clear of the stack, so there it often cannot tell.
        Verdict: "standing" only when the wrist says standing and the overhead does not say fallen (an overhead that
        cannot tell because its level fit is missing does not count as agreeing); "fallen" when either says fallen
        and the wrist does not say standing; else "unclear". Returns (verdict, info, best table fit)."""
        from hillclimb.runner.calib import overhead_fit as OF
        img, q, grip, stem = frame
        view, opening = self.op.view, grip * 0.1
        guess = (b["x"], b["y"], b["yaw"])
        lying = bool(b.get("lying"))
        n_below = int(round((bot(b) - TABLE_Z) / H))
        with self.full_mask():
            lvl = view.find(img, bid, q, opening, shape_of(b), guess)
            tabs = [view.find(img, bid, q, opening, shape_of({"lying": ly, "big": b.get("big")}, TABLE_Z), guess) for ly in (lying, not lying)]
            mids = [view.find(img, bid, q, opening, shape_of({"lying": False, "big": b.get("big")}, TABLE_Z + k * H), guess) for k in range(1, n_below)]
        tab_i = max(range(2), key=lambda i: -1 if tabs[i] is None else tabs[i]["iou"])
        tab = None if tabs[tab_i] is None else dict(tabs[tab_i], lying=(lying if tab_i == 0 else not lying), level=0, z0=TABLE_Z, big=b.get("big"))
        # a lower level (on a block lower in the stack) fitting clearly better is a fall too
        for k, f_ in enumerate(mids, start=1):
            if f_ is not None and (tab is None or f_["iou"] > tab["iou"]):
                tab = dict(f_, lying=False, level=k, z0=TABLE_Z + k * H, big=b.get("big"))
        info = {"block": bid, "z0": bot(b), "frame": stem, "wrist_level": lvl, "wrist_table": tab}
        why = []
        d_l = None if lvl is None else math.hypot(lvl["x"] - b["x"], lvl["y"] - b["y"])
        lvl_ok = lvl is not None and lvl["iou"] >= STAND_W_IOU and d_l <= xy_tol
        tab_poss = tab is not None and self._possible(tab, under)
        if lvl_ok and (tab is None or not tab_poss or lvl["iou"] >= tab["iou"] + STAND_MARGIN):
            w = "standing"
        elif (tab_poss and tab["iou"] >= STAND_W_IOU and tab["iou"] >= (lvl["iou"] if lvl else 0) + STAND_MARGIN) or \
                ((lvl is None or lvl["iou"] < 0.3) and (tab is None or tab_poss)):
            w = "fallen"
        else:
            w = "unclear"
        if w != "standing":
            why.append(f"wrist {w}: " + ("no fit at its level" if lvl is None else f"level fit {lvl['iou']:.2f} at {d_l * 1000:.0f} mm")
                       + (f", table fit {tab['iou']:.2f}{'' if tab_poss else ' (inside the block under)'}" if tab else ""))
        oh = "no frame"
        try:
            ohimg = cv2.imread(str(self.op.run / f"{stem}-overhead.jpg"))
            cam = OF.camera()
            col = bid.split("#")[0]
            m = full_colour_mask(ohimg if ohimg.shape[1] == cam.W else cv2.resize(ohimg, (cam.W, cam.H)), OF.colour_mask(
                ohimg if ohimg.shape[1] == cam.W else cv2.resize(ohimg, (cam.W, cam.H)), col), None)
            ol = OF.fit(ohimg, col, shape_of(b), guess, cam, mask=m)
            ots = [OF.fit(ohimg, col, shape_of({"lying": ly, "big": b.get("big")}, TABLE_Z), guess, cam, mask=m) for ly in (lying, not lying)]
            ot = max([o for o in ots if o is not None], key=lambda o: o["iou"], default=None)
            poss = self._possible(ot, under)
            info.update(overhead_level=ol, overhead_table=ot, overhead_table_possible=poss)
            d_o = None if ol is None else math.hypot(ol["x"] - b["x"], ol["y"] - b["y"])
            ol_ok = ol is not None and ol["iou"] >= STAND_OH_IOU and d_o <= max(STAND_OH_XY, xy_tol)
            if ol_ok and (ot is None or not poss or ol["iou"] >= ot["iou"] + STAND_MARGIN):
                oh = "standing"
            elif ot is not None and poss and n_below <= 1 and ot["iou"] >= STAND_OH_IOU and ot["iou"] >= (ol["iou"] if ol_ok else 0) + STAND_MARGIN:
                oh = "fallen"
            elif ol_ok:
                oh = "cannot tell"  # the same blob as good at table level, clear of the stack (parallax)
            else:
                oh = "no level fit"
            if oh != "standing":
                why.append(f"overhead {oh}: " + ("no fit at its level" if ol is None else f"level fit {ol['iou']:.2f} at {d_o * 1000:.0f} mm")
                           + (f", table fit {ot['iou']:.2f}{'' if poss else ' (inside the block under)'}" if ot else ""))
        except Exception as e:  # noqa: BLE001
            oh = f"no frame ({type(e).__name__})"
            why.append(f"overhead: {oh}")
        # Nov 8 night: the overhead's calibration is about 9 cm off (green fits 0.88 at 91 mm), so it often has no level
        # fit at all; then the wrist decides alone (its own margin, level fit 0.75 or better; 01:52: yellow stood on
        # orange at 0.93 against 0.86 from the far second frame). An overhead that sees it fallen still vetoes.
        wrist_clear = lvl is not None and lvl["iou"] >= 0.75 and (tab is None or lvl["iou"] >= tab["iou"] + STAND_MARGIN)
        # and up to 3 high a very clear wrist outweighs an overhead "fallen" (01:11: blue stood on orange in both frames,
        # wrist 0.97 against 0.74, while the shifted overhead fitted it on the table at 0.92)
        wrist_sure = lvl is not None and lvl["iou"] >= 0.9 and (tab is None or lvl["iou"] >= tab["iou"] + 0.15) and n_below <= 2
        if w == "standing" and (oh in ("standing", "cannot tell") or (oh == "no level fit" and wrist_clear)
                                or (oh == "fallen" and wrist_sure)):
            v = "standing"
        elif w == "fallen" or (oh == "fallen" and w != "standing"):
            v = "fallen"
        else:
            v = "unclear"
        if tab is None and info.get("overhead_table") is not None:
            tab = dict(info["overhead_table"], edge=False, lying=False, big=b.get("big"))
        info.update(wrist=w, overhead=oh, verdict=v, why="; ".join(why) or "ok")
        O.record("standing", run=self.dir.name, stem=stem, **info)
        status(f"standing check {bid} at {bot(b) * 1000:.0f} mm ({stem}): {v.upper()} (wrist {w}: level "
               f"{None if lvl is None else round(lvl['iou'], 2)}, table {None if tab is None else round(tab['iou'], 2)}; overhead {oh})"
               f"{'' if v == 'standing' else ': ' + info['why']}")
        return v, info, tab

    def support_still(self, sup):
        """Right before a place goes down: the support in a fresh overhead frame, at its believed level, within 2 cm
        of where its close look put it, and no possible table-level explanation clearly better. (ok, why)"""
        from hillclimb.runner.calib import overhead_fit as OF
        b = self.belief[sup]
        try:
            data, _, _ = fetch("overhead")
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            cam = OF.camera()
            if img.shape[1] != cam.W:
                img = cv2.resize(img, (cam.W, cam.H), interpolation=cv2.INTER_AREA)
            col = sup.split("#")[0]
            m = full_colour_mask(img, OF.colour_mask(img, col), None)
            ol = OF.fit(img, col, shape_of(b), (b["x"], b["y"], b["yaw"]), cam, mask=m)
            ot = None
            if bot(b) > TABLE_Z + 0.005:
                ots = [OF.fit(img, col, shape_of({"lying": ly, "big": b.get("big")}, TABLE_Z), (b["x"], b["y"], b["yaw"]), cam, mask=m) for ly in (False, True)]
                ot = max([o for o in ots if o is not None], key=lambda o: o["iou"], default=None)
        except Exception as e:  # noqa: BLE001
            return False, f"no overhead frame ({type(e).__name__}: {e})"
        d = None if ol is None else math.hypot(ol["x"] - b["x"], ol["y"] - b["y"])
        poss = self._possible(ot, self.under(sup)) if ot is not None else False
        ok = ol is not None and ol["iou"] >= 0.35 and d <= 0.02 and not (poss and ot["iou"] > ol["iou"] + STAND_MARGIN)
        why = ("no fit at its level" if ol is None else f"level fit {ol['iou']:.2f} at {d * 1000:.0f} mm") + \
            (f", table fit {ot['iou']:.2f}{'' if poss else ' (inside the block under)'}" if ot is not None else "")
        O.record("support_still", run=self.dir.name, block=sup, ok=ok, why=why, level=ol, table=ot)
        if not ok:
            status(f"over {sup} before going down: NOT where its close look put it ({why})")
        return ok, why

    def standing_twice(self, bid, b, frame, under=None, xy_tol=STAND_XY):
        """The standing check on this frame; when it cannot tell, once more from the high look pose (an oblique view
        of the stack, and a fresh overhead frame)."""
        v, info, tab = self.standing(bid, b, frame, under, xy_tol)
        if v != "unclear":
            return v, info, tab
        try:
            if not (bot(b) > TABLE_Z + 0.005 and self.top_look(b["x"], b["y"], b["yaw"], top_of(b), "stand2", skip=1)):
                self.to_pose(LOOK_TIP, "look")
        except SystemExit as e:
            status(f"standing check {bid}: no second frame ({str(e)[:120]})")
            return v, info, tab
        time.sleep(0.25)
        return self.standing(bid, b, self.op.snap(f"stand2-{bid}"), under, xy_tol)

    def relook_fallen(self, ids):
        """Blocks that fell: found again from the high look pose, on the table (flat or on its side) near where they
        were; a block not found keeps its x, y at table level, marked lost (it stays something to keep clear of)."""
        try:
            self.to_pose(LOOK_TIP, "look")
            time.sleep(0.25)
            img, q, grip, stem = self.op.snap("relook-fallen")
        except SystemExit as e:
            status(f"re-look of the fallen blocks skipped: {str(e)[:160]}")
            img = None
        for c in ids:
            b = self.belief[c]
            best = None
            if img is not None:
                for ly in (False, True):
                    f = self.op.view.find(img, c, q, grip * 0.1, shape_of({"lying": ly, "big": b.get("big")}, TABLE_Z), (b["x"], b["y"], b["yaw"]))
                    if f is not None and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) < 0.15 and (best is None or f["iou"] > best["iou"]):
                        best = dict(f, lying=ly)
            dup = [o for o, ob in self.belief.items() if best is not None and o != c and o.split("#")[0] == c.split("#")[0]
                   and abs(bot(ob) - TABLE_Z) < 0.005 and math.hypot(ob["x"] - best["x"], ob["y"] - best["y"]) < 0.03]
            if dup:
                # the look already has it as a free block (a new id): that entry stays the block; this one is parked
                # far off the table in the belief (out of every path and sweep) so the block is not there twice
                self.belief[c] = dict(b, x=0.6, y=-0.5, level=0, z0=TABLE_Z, iou=0.0, fallen=True, lost=True, placed=False, same_as=dup[0])
                status(f"fallen {c}: it is {dup[0]} on the table at ({best['x'] * 100:.1f}, {best['y'] * 100:.1f})")
            elif best is not None and best["iou"] >= 0.55:
                self.belief[c] = dict(b, x=best["x"], y=best["y"], yaw=best["yaw"], level=0, z0=TABLE_Z, lying=best["lying"],
                                      iou=best["iou"], edge=best["edge"], fallen=True, placed=False, t_seen=time.time())
                status(f"fallen {c}: on the table at ({best['x'] * 100:.1f}, {best['y'] * 100:.1f}){' on its side' if best['lying'] else ''} iou {best['iou']:.2f}")
            else:
                self.belief[c] = dict(b, level=0, z0=TABLE_Z, iou=0.0, fallen=True, lost=True, placed=False)
                status(f"fallen {c}: not found from the look pose; kept at ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) on the table as lost")
        self.put_blocks()

    def after_fall(self, top, err):
        """The stack under a place is not standing: the highest block of it that still stands (each checked from its
        own fresh frame), the ones above it re-looked as fallen. Returns that block's id, or None."""
        stack = sorted(self.below(top), key=lambda c: -bot(self.belief[c]))
        fallen, new = [top], None
        for c in stack[1:]:
            try:
                self.support_look(c)  # its own fresh frame: the standing check if raised, seen at its level if not
                new = c
                break
            except TowerFell:
                fallen.append(c)
            except (SystemExit, CycleFail) as e:
                # not confirmed (not seen where it should be, or no frame): nothing under the fall is placed on
                status(f"after the fall: {c} not confirmed ({str(e)[:160]}); nothing to place on")
                break
        O.record("tower_fell", run=self.dir.name, top=top, fallen=fallen, stands=new, why=str(err))
        self.relook_fallen(fallen)
        return new

    def take_off(self, colour, cycle, k):
        """A block left off on its stack after the re-places: set down alone on the table (it is the stack's top)."""
        try:
            x, y, yaw = self.pick_spot(colour, [0])
            self.block(colour, (x, y, yaw, 0, None), cycle, k, "off-stack")
        except CycleFail as e:
            raise Fatal(f"{colour}: off on its stack and could not be taken off ({e}): stopped so nothing goes on it")

    def grip_to(self, s, g, wait_close=False):
        _post("/move", {"q": s["goal"], "grip": g, "rate": 0.1})
        t_c = time.time()
        if not wait_close:
            time.sleep(0.35)
            return
        g_prev, still = None, 0
        while time.time() - t_c < 1.5:
            gg = arm_state()["grip"]
            still = still + 1 if g_prev is not None and abs(gg - g_prev) < 0.002 else 0
            g_prev = gg
            if still >= 2 and time.time() - t_c > 0.15:
                break
            time.sleep(0.05)

    # ---- one block
    def check_place(self, colour, target, level, z0=None, moved=False, under=None):
        """The check photo after a place. On a stack (z0 over the table) the placed block must pass the standing check
        at its believed level (self.last_standing False and its table fit returned when not); on the table it is fitted
        at table level (flat or on its side)."""
        tx, ty, tyaw = target[:3]
        z0 = TABLE_Z + level * H if z0 is None else z0
        held = dict(self.belief.get(colour, {}), z0=z0)
        self.last_standing = None
        if not moved and not (z0 > TABLE_Z + 0.005 and self.top_look(tx, ty, tyaw, top_of(held), "check")):
            try:
                pl = Plan(self.op, np.array(self.op.st["target"]), 0.08)
                pl.to([tx - 0.065, ty, min(max(0.13, top_of(held) + 0.06), LOOK_MAX_Z)], math.degrees(tyaw), 0.4, "check")
                pl.check()
                self.sweep(pl, exempt=set())
                self.go(pl, 0.8, "check", "level")
            except SystemExit as e:  # the photo from the back-off pose instead
                status(f"check pose skipped ({str(e)[:120]}); photo from the back-off pose")
        time.sleep(0.1)
        img, q, grip, stem = self.op.snap(f"check-{colour}")
        known = bool(held.get("lying", False))
        best = None
        if z0 > TABLE_Z + 0.005:
            b = dict(held, x=tx, y=ty, yaw=tyaw)
            sup = self.belief.get(under) if under else None
            v, info, tab = self.standing_twice(colour, b, (img, q, grip, stem), under=sup, xy_tol=STAND_XY_PLACED)
            if v == "unclear":
                raise Fatal(f"cannot tell whether {colour} stands on {under} after the place, from two frames ({info['why']}): "
                            "stopped so nothing goes on it")
            self.last_standing = v == "standing"
            if self.last_standing and info.get("wrist_level") is not None:
                best = (dict(info["wrist_level"], level=level, z0=z0, lying=known), shape_of({"lying": known, "big": held.get("big")}, z0))
            elif not self.last_standing and tab is not None:
                best = (dict(tab, level=tab.get("level", 0), z0=tab.get("z0", TABLE_Z)),
                        shape_of({"lying": tab.get("lying", False), "big": held.get("big")}, tab.get("z0", TABLE_Z)))
        else:
            for lying in (known, not known):  # the shape it was picked as, unless the other fits clearly better
                shape = shape_of({"lying": lying, "big": held.get("big")}, z0)
                with self.full_mask():
                    f = self.op.view.find(img, colour, q, grip * 0.1, shape, (tx, ty, tyaw))
                if f is not None and (best is None or f["iou"] > best[0]["iou"] + (0.05 if lying != known else 0)):
                    best = (dict(f, level=level, z0=z0, lying=lying), shape)
        fits = {colour: best} if best else {}
        cv2.imwrite(str(self.op.run / f"{stem}-ov.jpg"), self.op.overlay(img, q, grip * 0.1, fits, stem))
        # the next blocks, refreshed when this frame shows them well
        for c, b in self.belief.items():
            if c == colour:
                continue
            f = self.op.view.find(img, c, q, grip * 0.1, shape_of(b), (b["x"], b["y"], b["yaw"]))
            if f and f["iou"] >= 0.88 and not f["edge"] and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) < 0.015:
                b.update(x=f["x"], y=f["y"], yaw=f["yaw"])
        return (best[0] if best else None), stem

    def legal_place(self, colour, q0, R, tgt, grasp_h):
        """The place plan with the block pressed 2 mm into the surface; if the jaws rule refuses it, the set may stop up
        to 4 mm higher (the block's bottom at most 2 mm over the surface: set down, never dropped). (plan, mm higher)."""
        why = None
        for k_up in range(5):
            try:
                return self.plan_place(colour, q0, R, tgt, grasp_h + 0.001 * k_up), k_up, None
            except SystemExit as e:
                why = str(e)
        return None, None, why

    def block(self, colour, target, cycle, k, phase, recovery=False, regrip=0.0, attempt=0, replace=False):
        """Pick colour (belief) and place it at target (x, y, yaw, level, support colour or None). Returns ok.
        replace: a re-place of a block that landed off on its stack (the support is under it: no support look; the
        check photo just taken stands for it). attempt: re-places so far."""
        op = self.op
        op.held = None
        self.held_ext = 0.0
        times = {}
        last = [time.time()]

        def mark(name):
            now_ = time.time()
            times[name] = round(times.get(name, 0.0) + now_ - last[0], 2)
            last[0] = now_
        self.phase_times = times
        tx, ty, tyaw, tlevel, sup = target
        # carry high enough that the held block's bottom clears the stack it goes onto by 2 cm
        sup_top = top_of(self.belief[sup]) if sup else TABLE_Z
        # carry high enough that the held block clears the stack it goes onto, inside the arm's reach
        src_top = top_of(self.belief[colour]) if colour in self.belief else TABLE_Z
        # carry high enough that the held block clears the stack it goes onto and the one it came off, inside reach
        self.dr = {"carry_z": round(min(max(float(self.rng.uniform(*CARRY)), sup_top + 0.05, src_top + 0.02), max(REACH_Z, sup_top + 0.035)), 4),
                   "over_dz": round(float(self.rng.uniform(0, 0.02)), 4)}
        if sup and not replace:
            # the block below, from a fresh frame: standing (or the place goes on what still stands), then measured at
            # its own level: the place squared and centred on it
            try:
                sb = self.support_look(sup)
            except TowerFell as e:
                new = self.after_fall(sup, e)
                if new is None:
                    raise CycleFail(f"{sup}: the stack under the place is not standing and nothing under it stands: {e}")
                status(f"{colour}: {sup} is not standing ({e}); placing on {new}, the highest block that stands")
                sup, sb = new, self.belief[new]
                tlevel = int(sb.get("level", round((bot(sb) - TABLE_Z) / H))) + 1
                sup_top = top_of(sb)
                self.dr["carry_z"] = round(min(max(self.dr["carry_z"], sup_top + 0.05), max(REACH_Z, sup_top + 0.035)), 4)
            tx, ty, tyaw = sb["x"], sb["y"], sb["yaw"]
            target = (tx, ty, tyaw, tlevel, sup)
        if not recovery:
            self.relook(colour)
            mark("re-look")
        src = dict(self.belief[colour])
        top = top_of(self.belief[sup]) if sup else TABLE_Z
        q0 = self.start_q()
        corr = self.correction(tlevel, tx, ty)  # aim off by the learned bias at this level and place (mm, arm frame)
        ax, ay = tx - corr[0] / 1000, ty - corr[1] / 1000
        try:
            pl = self.plan_pick(colour, q0, regrip)
            # the place must solve too, before anything moves (nominal grasp height)
            lift = self.plan_lift(pl.nodes[-1][0], pl.R, math.degrees(src["yaw"]), colour)
            self.plan_place(colour, lift.nodes[-1][0], lift.R, (ax, ay, tyaw, top, sup), 0.015)
        except SystemExit as e:
            raise NoPlan(f"{colour} to ({tx * 100:.1f}, {ty * 100:.1f}) L{tlevel}: {e}")
        T0 = now()
        mark("plan pick")
        t_wall = time.time()
        self.go(pl, getattr(self, "pick_open", {}).get(colour, 0.8), "pick", "down")
        mark("pick stream")
        s = arm_state()
        self.grip_to(s, GRIP_CLOSE, wait_close=True)
        mark("close")
        s = arm_state()
        q_m = np.array(s["q"]) + op.dq
        tip_grasp, yaw_grasp = op.tip(q_m)
        grasp = {"run": self.dir.name, "block": colour, "grip": s["grip"], "tip_fk_mm": (tip_grasp * 1000).round(1),
                 "jaw_yaw_deg": round(yaw_grasp, 1), "wrist_look": src, "q_measured": s["q"], "cycle": cycle}
        prompt = f"pick up the {colour} block"
        if s["grip"] < GRIP_HELD:
            O.record("grasp_failed", **grasp)
            self.grip_to(s, 0.8)
            back = Plan(op, np.array(op.st["target"]), 0.08)
            back.R = pl.R
            back.to(op.tip(pl.nodes[-1][0])[0] + [0, 0, 0.06], math.degrees(src["yaw"]), 0.3, "back up", line=0.01, keep=True)
            back.check(skip=colour)
            self.go(back, 0.8, "back up", "up")
            self.episode(T0, now(), skill="pick", prompt=prompt, success=False, colour=colour, start_pose=src, grasp_pose=None,
                         end_pose=None, error="grip closed empty", duration_s=round(time.time() - t_wall, 2), cycle=cycle,
                         block=k, recovery=recovery, phase=phase)
            status(f"cycle {cycle} block {k} {phase} {colour}: grasp missed (grip {s['grip']:.3f})")
            if recovery:
                raise CycleFail(f"{colour}: grasp missed twice")
            # one recovery: look from right over it
            img, q, grip, stem = op.snap(f"relook-{colour}")
            f = op.view.find(img, colour, q, grip * 0.1, shape_of(src), (src["x"], src["y"], src["yaw"]))
            if f is None or f["iou"] < self.relook_iou:
                # not seen from right over it (jaws and neighbours hide it): the high look pose and a refit by id
                status(f"{colour}: not seen from over it after the miss; re-looking from the look pose")
                self.relook(colour)
            else:
                self.belief[colour].update(x=f["x"], y=f["y"], yaw=f["yaw"])
                self.put_blocks()
            return self.block(colour, target, cycle, k, phase, recovery=True)
        # held: the wrist frame of the block between the pads, while the lift streams
        held_frame = {}
        threading.Thread(target=lambda: held_frame.update(data=fetch("wrist")), daemon=True).start()
        z_bottom = bot(src)
        grasp_h = tip_grasp[2] - z_bottom
        op.held = (colour, float(grasp_h))  # the motion rule clears stacks by the held block's bottom from here on
        self.held_ext = ext(src)  # a held brick reaches 16.5 mm further: the sweep keeps that much more clear
        lift = self.plan_lift(np.array(op.st["target"]), pl.R, math.degrees(src["yaw"]), colour)
        self.go(lift, GRIP_CLOSE, "lift", "up")
        mark("lift")
        s = arm_state()
        T1 = now()
        held_ok = s["grip"] >= GRIP_HELD
        truth, held_fit = None, None
        for _ in range(20):
            if "data" in held_frame:
                break
            time.sleep(0.05)
        if "data" in held_frame:
            data, jq, jg = held_frame["data"]
            (op.run / f"held-{cycle}-{k}-wrist.jpg").write_bytes(data)
            img = cv2.resize(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), (640, 360), interpolation=cv2.INTER_AREA)
            qh = np.array(jq) + op.dq
            tip_h = op.tip(qh)[0]
            held_fit = op.view.find(img, colour, qh, jg * 0.1, shape_of(src, tip_h[2] - grasp_h), None)
            if held_fit:
                held_fit["minus_tip_mm"] = [round((held_fit["x"] - tip_h[0]) * 1000, 1), round((held_fit["y"] - tip_h[1]) * 1000, 1)]
        truth = tip_grasp[:2] + (np.array(held_fit["minus_tip_mm"]) / 1000 if held_fit and held_fit["iou"] > 0.6 else 0)
        grasp.update(held_in_wrist=held_fit, truth_mm=(truth * 1000).round(1), grasp_h_mm=round(grasp_h * 1000, 1),
                     wrist_err_mm=((np.array([src["x"], src["y"]]) - truth) * 1000).round(1))
        oh = [o for o in (self.look_overhead.get(colour) or [])] if isinstance(self.look_overhead, dict) else []
        if oh:
            o = min(oh, key=lambda o: math.hypot(o["x"] - truth[0], o["y"] - truth[1]))
            grasp["overhead"] = o
            grasp["overhead_err_mm"] = ((np.array([o["x"], o["y"]]) - truth) * 1000).round(1)
        O.record("grasp", **grasp)
        grasp_pose = {"x": truth[0], "y": truth[1], "yaw": math.radians(yaw_grasp), "level": src["level"], "z_tip": tip_grasp[2]}
        self.episode(T0, T1, skill="pick", prompt=prompt, success=bool(held_ok), colour=colour, start_pose=src, grasp_pose=grasp_pose,
                     end_pose={"tip_mm": (op.tip(np.array(s["q"]) + op.dq)[0] * 1000).round(1), "grip": s["grip"]},
                     error={"tracker_minus_truth_mm": grasp["wrist_err_mm"]}, duration_s=round(time.time() - t_wall, 2),
                     cycle=cycle, block=k, recovery=recovery, phase=phase)
        if not held_ok:
            raise CycleFail(f"{colour} slipped out during the lift (grip {s['grip']:.3f})")
        # place
        t_place = time.time()
        pl3, k_up, why = self.legal_place(colour, np.array(op.st["target"]), lift.R, (ax, ay, tyaw, top, sup), grasp_h)
        hf = held_fit if (held_fit and held_fit.get("iou", 0) >= 0.75 and held_fit.get("minus_tip_mm") is not None) else None
        if pl3 is not None and hf is not None and sup:
            off = np.array(hf["minus_tip_mm"], float) / 1000.0
            if 0.003 < float(np.linalg.norm(off)) <= 0.03:
                a = math.radians(op.tip(pl3.nodes[-1][0])[1] - op.tip(qh)[1])
                offp = np.array([math.cos(a) * off[0] - math.sin(a) * off[1], math.sin(a) * off[0] + math.cos(a) * off[1]])
                pl4, k4, why4 = self.legal_place(colour, np.array(op.st["target"]), lift.R, (ax - offp[0], ay - offp[1], tyaw, top, sup), grasp_h)
                O.record("held_offset", run=self.dir.name, block=colour, level=tlevel, held_mm=(off * 1000).round(1).tolist(),
                         shift_mm=(-offp * 1000).round(1).tolist(), planned=pl4 is not None)
                if pl4 is not None:
                    status(f"{colour}: held {float(np.linalg.norm(off)) * 1000:.0f} mm off the fingertips; place aim shifted by "
                           f"{(-offp * 1000).round(1).tolist()} mm")
                    pl3, k_up, why = pl4, k4, why4
                    ax, ay = ax - offp[0], ay - offp[1]
        mark("plan place")
        if pl3 is not None and src.get("big"):
            # a brick lands with its long side across the jaws' final frame (any quarter turn of the wrist plans):
            # its target yaw is that one, for the belief, the check photo and the labels
            tyaw = (math.radians(op.tip(pl3.nodes[-1][0])[1]) + math.pi / 2 + math.pi / 2) % math.pi - math.pi / 2
            target = (tx, ty, tyaw) + tuple(target[3:])
        if pl3 is not None and k_up:
            status(f"{colour}: grasp {grasp_h * 1000:.1f} mm over its bottom; set {k_up} mm over the 2 mm press (bottom within 2 mm of contact)")
        if pl3 is None:
            # too low to set down on contact here: set it back where it came from (same level), open, regrip 10 mm higher
            if regrip > 0:
                raise Fatal(f"{colour} held: still no legal set-down after a regrip ({why})")
            src_sup = next((c for c, o in self.belief.items() if c != colour and o["level"] == src["level"] - 1
                            and math.hypot(o["x"] - src["x"], o["y"] - src["y"]) < 0.03), None)
            status(f"{colour}: grasp {grasp_h * 1000:.1f} mm over its bottom is too low for a contact set-down at the target ({why}); "
                   f"setting it back at L{src['level']} to regrip 10 mm higher")
            plb, kb, whyb = self.legal_place(colour, np.array(op.st["target"]), lift.R, (src["x"], src["y"], src["yaw"], z_bottom, src_sup), grasp_h)
            if plb is None:
                raise Fatal(f"{colour} held: no contact set-down at the target or back at its source ({whyb})")
            Tb = now()
            self.go(plb, GRIP_CLOSE, "set back", "down")
            self.grip_to(arm_state(), 0.8)
            op.held = None
            self.held_ext = 0.0
            back = Plan(op, np.array(op.st["target"]), 0.08)
            back.R = plb.R
            back.to(op.tip(plb.nodes[-1][0])[0] + [0, 0, 0.05], math.degrees(src["yaw"]), 0.3, "back off", line=0.01, keep=True)
            back.check(skip=colour)
            self.go(back, 0.8, "back off", "up")
            pp = STACK_PHRASES[0].format(a=colour, b=src_sup) if src_sup else f"put the {colour} block down on the table"
            self.episode(Tb, now(), skill="place", prompt=pp, success=False, colour=colour, start_pose=grasp_pose, target=src,
                         end_pose=None, error="set back unchecked, to regrip higher", duration_s=None, cycle=cycle, block=k,
                         recovery=True, phase=phase + "-setback")
            return self.block(colour, target, cycle, k, phase, recovery=recovery, regrip=0.010)
        if sup:
            # the carry first, stopping right over the support; one more look at it (the overhead frame: the held
            # block hides it from the wrist) before going down: it may have been knocked since its close look
            k_c = max(i for i, n in enumerate(pl3.nodes) if n[2] == "carry")
            carry = Plan(op, pl3.nodes[0][0], pl3.opening, held=pl3.held)
            carry.nodes, carry.R = pl3.nodes[:k_c + 1], pl3.R
            down = Plan(op, pl3.nodes[k_c][0], pl3.opening, held=pl3.held)
            down.nodes, down.R = [pl3.nodes[k_c]] + pl3.nodes[k_c + 1:], pl3.R
            self.go(carry, GRIP_CLOSE, "carry", "level")
            ok_sup, why_sup = self.support_still(sup)
            if not ok_sup:
                # not placing: the block goes back where it came from (same level), and the caller re-plans
                src_sup = next((c for c, o in self.belief.items() if c not in (colour, sup) and abs(top_of(o) - z_bottom) < 0.006
                                and math.hypot(o["x"] - src["x"], o["y"] - src["y"]) < 0.03), None)
                plb, kb, whyb = self.legal_place(colour, np.array(op.st["target"]), lift.R, (src["x"], src["y"], src["yaw"], z_bottom, src_sup), grasp_h)
                if plb is None:
                    raise Fatal(f"{colour} held over {sup}, which moved ({why_sup}); no set-down back at its source ({whyb})")
                self.go(plb, GRIP_CLOSE, "set back", "down")
                self.grip_to(arm_state(), 0.8)
                op.held = None
                self.held_ext = 0.0
                self.belief[colour] = dict(src)
                self.put_blocks()
                back = Plan(op, np.array(op.st["target"]), 0.08)
                back.R = plb.R
                back.to(op.tip(plb.nodes[-1][0])[0] + [0, 0, 0.05], math.degrees(src["yaw"]), 0.3, "back off", line=0.01, keep=True)
                back.check(skip=colour)
                self.go(back, 0.8, "back off", "up")
                raise CycleFail(f"{colour}: not placed: {sup} moved during the carry ({why_sup}); set back at its source")
            self.go(down, GRIP_CLOSE, "place", "down")
        else:
            self.go(pl3, GRIP_CLOSE, "place", "down")
        mark("carry and set")
        s = arm_state()
        self.release(s)
        op.held = None
        self.held_ext = 0.0
        # where it was set, until the check photo measures it: the back-off and the check move clear it as a stack
        self.belief[colour] = dict(self.belief.get(colour, {}), x=tx, y=ty, yaw=tyaw, level=tlevel, z0=top,
                                   lying=src.get("lying", False), iou=0.0, edge=False, big=src.get("big", False))
        self.put_blocks()
        mark("open")
        back = Plan(op, np.array(op.st["target"]), 0.08)
        back.R = pl3.R
        back.to(op.tip(pl3.nodes[-1][0])[0] + [0, 0, 0.05], math.degrees(tyaw), 0.3, "back off", line=0.01, keep=True)
        back.check(skip=colour)
        # the check pose (6.5 cm back) in the same stream: no stop between the back-off and the check photo
        held_now = dict(self.belief.get(colour, {}), z0=top)
        chk = Plan(op, back.nodes[-1][0], 0.08)
        chk.to([tx - 0.065, ty, min(max(0.13, top_of(held_now) + 0.06), LOOK_MAX_Z)], math.degrees(tyaw), V_HIGH, "check")
        try:
            chk.check()
            self.sweep(chk, exempt=set())
            back.nodes = back.nodes + chk.nodes[1:]
            at_check = True
        except SystemExit:
            at_check = False
        self.go(back, 0.8, "back off", "up")
        mark("back-off and check move")
        T2 = now()
        t_block = time.time() - t_wall
        # check
        t_c = time.time()
        fit, stem = self.check_place(colour, (tx, ty, tyaw), tlevel, top, moved=at_check, under=sup)
        t_check = time.time() - t_c
        mark("check")
        if sup:
            phrase = STACK_PHRASES[self.nstack % 2]
            self.nstack += 1
            pprompt = phrase.format(a=colour, b=sup)
        else:
            pprompt = f"put the {colour} block down on the table"
        off = yaw_off = None
        ok = False
        if fit is not None:
            off = [round((fit["x"] - tx) * 1000, 1), round((fit["y"] - ty) * 1000, 1)]
            yaw_off = round((math.degrees(fit["yaw"] - tyaw) + 45) % 90 - 45, 1)
            ok = fit["level"] == tlevel and fit["iou"] >= IOU_MIN
            if sup:  # neatness: relative to the measured block directly below
                ok = ok and math.hypot(*off) <= NEAT_MM and abs(yaw_off) <= NEAT_DEG
            else:
                ok = ok and math.hypot(*off) <= OFF_TOL * 1000
        O.record("place", run=self.dir.name, block=colour, target=[tx, ty, tyaw, tlevel], support=sup, fit=fit, offset_mm=off,
                 yaw_off_deg=yaw_off, stem=stem, cycle=cycle, recovery=recovery,
                 neat=({"rel_offset_mm": off, "rel_yaw_deg": yaw_off, "under": sup} if sup else None),
                 correction_mm=[round(c, 2) for c in corr], aimed=[ax, ay])
        if fit is not None and off is not None and fit["level"] == tlevel:
            self.bias.setdefault(tlevel, []).append([tx, ty, off[0] + corr[0], off[1] + corr[1]])
        self.episode(T1, T2, skill="place", prompt=pprompt, success=bool(ok), colour=colour, start_pose=grasp_pose,
                     target={"x": tx, "y": ty, "yaw": tyaw, "level": tlevel, "support": sup}, end_pose=fit,
                     error={"offset_mm": off, "yaw_off_deg": yaw_off, "relative_to": sup or "target spot"},
                     role=({0: "base", 1: "middle", 2: "top"}.get(tlevel, f"level {tlevel}") if phase == "stack" else "scatter"),
                     duration_s=round(time.time() - t_place - t_check, 2),
                     cycle=cycle, block=k, recovery=recovery, phase=phase)
        # the whole block move as one task episode (Dream fine-tuning): still at the look (or check) pose, pick, carry,
        # place, back off, still; stack moves and table moves in their own folders, failures apart
        task = "stack" if sup else "place_on_table"
        tprompt = f"stack the {colour} block on top of the {sup} block" if sup else f"put the {colour} block down on the table"
        whole_ok = bool(ok and held_ok)
        self.episode(T0, T2, root=f"dream_ft/{task}" if whole_ok else f"dream_ft_failures/{task}", skill="task", prompt=tprompt,
                     success=whole_ok, colour=colour, start_pose=src, grasp_pose=grasp_pose,
                     target={"x": tx, "y": ty, "yaw": tyaw, "level": tlevel, "support": sup}, end_pose=fit,
                     error={"offset_mm": off, "yaw_off_deg": yaw_off, "relative_to": sup or "target spot"},
                     role=({0: "base", 1: "middle", 2: "top"}.get(tlevel, f"level {tlevel}") if phase == "stack" else phase),
                     duration_s=round(t_block, 2), cycle=cycle, block=k, recovery=recovery, phase=phase, task=task)
        self.block_times.append(t_block)
        status(f"cycle {cycle} block {k} {phase} {colour}: ({src['x'] * 100:.1f}, {src['y'] * 100:.1f}) L{src['level']} -> "
               f"({tx * 100:.1f}, {ty * 100:.1f}) L{tlevel}{' on ' + sup if sup else ''}: grip {grasp['grip']:.3f}, tracker err "
               f"{grasp['wrist_err_mm'].tolist()} mm; placed {'OK' if ok else 'FAIL'} off {off} mm yaw {yaw_off} deg iou "
               f"{None if fit is None else round(fit['iou'], 2)}; {t_block:.1f} s (+{t_check:.1f} s check){' [re-place]' if recovery else ''}; carry {self.dr['carry_z'] * 100:.1f} cm; aim corrected by {[-round(c, 1) for c in corr]} mm; phases {times}")
        if fit is not None and fit["iou"] >= 0.5:  # a poor check fit (a misread) does not move the belief
            self.belief[colour] = {"x": fit["x"], "y": fit["y"], "yaw": fit["yaw"], "level": fit["level"], "iou": fit["iou"], "edge": fit["edge"],
                                   "z0": fit.get("z0"), "lying": fit.get("lying", src.get("lying", False)), "placed": True, "big": src.get("big", False)}
        else:
            self.belief[colour] = {"x": tx, "y": ty, "yaw": tyaw, "level": tlevel, "iou": 0.0, "edge": False, "z0": top, "lying": src.get("lying", False),
                                   "big": src.get("big", False)}
        self.put_blocks()
        if sup and getattr(self, "last_standing", None) is False:
            # not standing on the stack after the place: it fell (the belief holds its table fit, or where it was set
            # with no fit); nothing goes on it, the caller re-plans from the next place's own standing check
            if fit is None:
                self.belief[colour].update(level=0, z0=TABLE_Z, iou=0.0, lost=True)
                self.put_blocks()
            raise CycleFail(f"{colour}: not standing on {sup} after the place (it fell)")
        if ok:
            return True
        if fit is None or (fit["iou"] < IOU_MIN and not (sup and getattr(self, "last_standing", None) is True
                                                      and fit["level"] == tlevel and fit["iou"] >= 0.5)
                                              and not (not sup and fit["level"] == 0 and fit["iou"] >= 0.60
                                                       and math.hypot(fit["x"] - tx, fit["y"] - ty) <= 0.02)):
            raise CycleFail(f"{colour}: not found flat after the place (tipped or lost)")
        if fit["iou"] < IOU_MIN:
            status(f"{colour}: weak fit {fit['iou']:.2f} after the place, at its level where it should be: taken")
        if sup:
            big = off is not None and (math.hypot(*off) > REPLACE_MM or abs(yaw_off) > REPLACE_DEG)
            if not recovery or (big and attempt < REPLACE_TRIES):
                status(f"{colour}: {off} mm / {yaw_off} deg off the {sup} block: re-placing it (re-place {attempt + 1})")
                return self.block(colour, target, cycle, k, phase, recovery=True, attempt=attempt + 1, replace=True)
            if big:
                # still over 8 mm or 10 deg off: nothing may go on it; it comes off the stack
                status(f"{colour}: still {off} mm / {yaw_off} deg off the {sup} block after {attempt} re-places: taking it off the stack")
                self.take_off(colour, cycle, k)
                raise CycleFail(f"{colour}: {off} mm / {yaw_off} deg off after {attempt} re-places: taken off the stack")
            status(f"{colour}: still {off} mm / {yaw_off} deg off the {sup} block after a re-place (within {REPLACE_MM:.0f} mm, {REPLACE_DEG:.0f} deg); kept")
            return False
        if recovery:
            raise CycleFail(f"{colour}: still off after one recovery ({off} mm)")
        return self.block(colour, target, cycle, k, phase, recovery=True)

    # ---- spots
    def visible(self, x, y, yaw, levels):
        view = self.op.view
        cam = view.pose(self.q_look, 0.08)
        for lvl in levels:
            uv, ok = view.project(cam, Shape((FACE, FACE, H), TABLE_Z + lvl * H).corners(x, y, yaw))
            if not ok.all() or (uv[:, 0] < 12).any() or (uv[:, 0] > view.W - 12).any() or (uv[:, 1] < 12).any() or (uv[:, 1] > view.H - 12).any():
                return False
            c = np.round(uv.mean(0)).astype(int)
            if self.jaws_look[min(c[1], view.H - 1), min(c[0], view.W - 1)]:
                return False
        return True

    def finger_clear(self, x, y, yaw, moving):
        """Open fingers (along either face axis) of this block and of each neighbour keep 4 cm from the other's centre."""
        def fingers(cx, cy, a):
            return [(cx + s * FINGER * math.cos(a + k * math.pi / 2), cy + s * FINGER * math.sin(a + k * math.pi / 2)) for s in (-1, 1) for k in (0, 1)]
        def inside(fx, fy, b, margin=0.012):
            """A finger point within margin of b's footprint (a brick is 83 mm long: 4 cm from its centre is not clear)."""
            c, s_ = math.cos(b.get("yaw", 0.0)), math.sin(b.get("yaw", 0.0))
            u, v = (fx - b["x"]) * c + (fy - b["y"]) * s_, -(fx - b["x"]) * s_ + (fy - b["y"]) * c
            hu, hv = half_of(b)
            return abs(u) < hu + margin and abs(v) < hv + margin
        me = dict(self.belief.get(moving, {}), x=x, y=y, yaw=yaw)
        for c, b in self.belief.items():
            if c == moving:
                continue
            if any(math.hypot(fx - b["x"], fy - b["y"]) < 0.04 or inside(fx, fy, b) for fx, fy in fingers(x, y, yaw)) or \
                    any(math.hypot(fx - x, fy - y) < 0.04 or inside(fx, fy, me) for fx, fy in fingers(b["x"], b["y"], b["yaw"])):
                return False
        return True

    def spot(self, moving, levels, avoid=(), gap=GAP):
        """Candidates: the spots that worked today (jittered 2 cm) first, then uniform in the area."""
        for i in range(3000):
            if i < len(GOOD) and AREA[0][0] <= GOOD[i][0] <= AREA[0][1] and AREA[1][0] <= GOOD[i][1] <= AREA[1][1]:
                x, y = GOOD[i][0] + self.rng.uniform(-0.02, 0.02), GOOD[i][1] + self.rng.uniform(-0.02, 0.02)
            else:
                x, y = self.rng.uniform(*AREA[0]), self.rng.uniform(*AREA[1])
            yaw = self.rng.uniform(-math.pi / 4, math.pi / 4)
            ext_m = ext(self.belief[moving]) if moving in self.belief else 0.0
            if any(math.hypot(x - b["x"], y - b["y"]) < gap + ext(b) + ext_m for c, b in self.belief.items() if c != moving) or \
                    any(math.hypot(x - a[0], y - a[1]) < gap for a in avoid):
                self.why["block gap"] += 1
                continue
            if moving and math.hypot(x - self.belief[moving]["x"], y - self.belief[moving]["y"]) < 0.06:
                self.why["too close to where it is"] += 1
                continue
            if not self.finger_clear(x, y, yaw, moving):
                self.why["jaws near a neighbour"] += 1
                continue
            if len(self.obstacles) and np.min(np.hypot(self.obstacles[:, 0] - x, self.obstacles[:, 1] - y)) < CLEAR:
                self.why["obstacle keep-out"] += 1
                continue
            if not self.visible(x, y, yaw, levels):
                self.why["out of the look's view"] += 1
                continue
            yield x, y, yaw

    def feasible(self, colour, target):
        """The whole block, planned from where the arm is (no motion)."""
        try:
            pl = self.plan_pick(colour, self.start_q())
            lift = self.plan_lift(pl.nodes[-1][0], pl.R, math.degrees(self.belief[colour]["yaw"]), colour)
            tx, ty, tyaw, tl, sup = target
            p3 = self.plan_place(colour, lift.nodes[-1][0], lift.R, (tx, ty, tyaw, TABLE_Z + tl * H, sup), 0.015)
            ck = Plan(self.op, p3.nodes[-1][0], 0.08)
            ck.to([tx - 0.065, ty, look_z(tl)], math.degrees(tyaw), 0.4, "check")
            ck.check(skip=colour)
            return True
        except SystemExit as e:
            self.last_refusal = str(e)
            return False

    def pick_spot(self, colour, levels, avoid=()):
        import collections
        self.why = collections.Counter()
        # the pick first, once: when it does not plan no spot can help (Nov 8 02:21: 61 spots planned over 7 minutes,
        # every one refused at "pick blue#1")
        try:
            self.plan_pick(colour, self.start_q())
        except SystemExit as e:
            raise NoPlan(f"{colour}: its pick does not plan from here: {e}")
        n = 0
        for gap in (GAP,):
            m = 0
            for x, y, yaw in self.spot(colour, levels, avoid, gap):
                n, m = n + 1, m + 1
                if self.feasible(colour, (x, y, yaw, 0, None)):
                    return x, y, yaw
                self.why["plan: " + self.last_refusal.split(":")[0][:40]] += 1
                if m > CANDIDATES:
                    break
        raise NoSpot(f"no feasible free spot for {colour} ({n} planned; rejections {dict(self.why)}; last plan refusal: {getattr(self, 'last_refusal', '')})")

    # ---- a cycle
    def cycle(self, n):
        t0 = time.time()
        lt = self.jitter_look()
        bel, stem = self.look(f"look-c{n}", lt)
        if bel is None:
            bel, stem = self.look(f"look2-c{n}", (0.31, 0.06, 0.21, 90.0))
        if bel is None:
            raise CycleFail("the look did not find three flat blocks")
        status(f"cycle {n} look {stem}: " + ", ".join(f"{c} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) L{b['level']} iou {b['iou']:.2f}"
                                                   for c, b in bel.items()) + f"; {len(self.obstacles)} obstacle points")
        k = 0
        old_tower = [(b["x"], b["y"]) for c, b in bel.items() if any(o["level"] > b["level"] and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.03
                                                                    for o in bel.values())]
        order = sorted(COLOURS, key=lambda c: (-bel[c]["level"], self.rng.random()))
        self.cyc = {"look_tip": [round(v, 4) for v in lt], "scatter_order": order, "start_levels": {c: bel[c]["level"] for c in COLOURS},
                    "fallbacks": []}
        left = set()
        for c in order:
            k += 1
            if any(o["level"] > self.belief[c]["level"] and math.hypot(o["x"] - self.belief[c]["x"], o["y"] - self.belief[c]["y"]) < 0.03
                   for o2, o in self.belief.items() if o2 != c):
                self.fallback(n, f"scatter {c}: a block still sits on it; left in place")
                left.add(c)
                continue
            if self.belief[c]["iou"] < 0.5:
                self.relook(c)  # hidden in the look: seen before its spot is chosen
            try:
                x, y, yaw = self.pick_spot(c, [0])
                self.block(c, (x, y, yaw, 0, None), n, k, "scatter")
            except (NoSpot, NoPlan) as e:
                done = False
                if self.belief[c]["level"] > 0:  # (b) onto a loose block: a 2-stack
                    for o in self.loose(exclude={c}):
                        try:
                            b = self.belief[o]
                            self.block(c, (b["x"], b["y"], b["yaw"], 1, o), n, k, "scatter-2stack")
                            self.fallback(n, f"scatter {c}: {e}; (b) set on the loose {o} block")
                            done = True
                            break
                        except NoPlan as e2:
                            e = e2
                if not done:
                    self.fallback(n, f"scatter {c}: {e}; (a) left where it is")
                    left.add(c)
            self.stop_check()
        # restack: on the tallest stack if one is left, else a new base spot, else the base where it is, else the old spot
        stacks = self.stacks()
        tower = max(stacks, key=len)
        if len(tower) >= 2:
            self.fallback(n, f"restack on the existing stack {' / '.join(tower)}")
        else:
            base = str(self.rng.choice(COLOURS))
            try:
                x, y, yaw = self.pick_spot(base, [0, 1, 2], avoid=old_tower)  # a new spot: not where the old tower stood
                k += 1
                self.block(base, (x, y, yaw, 0, None), n, k, "stack")
            except (NoSpot, NoPlan) as e:
                self.fallback(n, f"tower base {base}: {e}; (a) built on {base} where it is")
            tower = [base]
        rest = [c for c in self.rng.permutation(COLOURS) if c not in tower]
        self.cyc["tower_order"] = {"base": tower[0], "middle": (tower + rest)[1], "top": (tower + rest)[2]}
        b0 = self.belief[tower[0]]
        self.cyc["tower_spot"] = [round(b0["x"], 4), round(b0["y"], 4), round(b0["yaw"], 4)]
        for c in rest:
            k += 1
            s_ = self.belief[tower[-1]]
            self.block(str(c), (s_["x"], s_["y"], s_["yaw"], len(tower), tower[-1]), n, k, "stack")
            tower.append(str(c))
            self.stop_check()
        self.to_pose(LOOK_TIP, "end")
        status(f"cycle {n} done: tower {' / '.join(tower)} at ({b0['x'] * 100:.1f}, {b0['y'] * 100:.1f}); {time.time() - t0:.0f} s")

    def fallback(self, n, what):
        self.cyc.setdefault("fallbacks", []).append(what)
        status(f"cycle {n} fallback: {what}")

    def stacks(self):
        out = []
        for c, b in self.belief.items():
            if b["level"] != 0:
                continue
            st = [c]
            while True:
                nxt = [o for o, ob in self.belief.items() if ob["level"] == len(st)
                       and math.hypot(ob["x"] - self.belief[st[-1]]["x"], ob["y"] - self.belief[st[-1]]["y"]) < 0.03]
                if not nxt:
                    break
                st.append(nxt[0])
            out.append(st)
        return out

    def loose(self, exclude=()):
        return [st[0] for st in self.stacks() if len(st) == 1 and st[0] not in exclude]

    def relook(self, colour):
        """A fresh look at the block about to be picked, from a jittered high pose (then the plain one if it fails)."""
        b = self.belief[colour]
        for tip in (self.jitter_look(0.015, 5.0), LOOK_TIP):
            self.to_pose(tip, "look")
            time.sleep(0.25)
            img, q, grip, stem = self.op.snap(f"relook-{colour}")
            f = self.op.view.find(img, colour, q, grip * 0.1, shape_of(b), (b["x"], b["y"], b["yaw"]))
            if f is not None and f["iou"] >= self.relook_iou and not f["edge"] and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) <= 0.04:
                break
        if f is not None and not f["edge"] and f["iou"] >= 0.5 and f["iou"] < self.relook_iou and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) <= 0.02:
            status(f"relook {colour}: weak fit {f['iou']:.2f} within 2 cm of where it was: taken")
        elif f is None or f["iou"] < self.relook_iou or math.hypot(f["x"] - b["x"], f["y"] - b["y"]) > 0.04:
            raise CycleFail(f"{colour}: not found where it was hidden ({f})")
        b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"], edge=f["edge"])
        self.put_blocks()
        status(f"relook {colour}: ({f['x'] * 100:.1f}, {f['y'] * 100:.1f}) iou {f['iou']:.2f}")

    def stop_check(self):
        if STOPFILE.exists():
            raise Fatal("STOP file")


def run(cycles=20):
    LOOP.mkdir(parents=True, exist_ok=True)
    session = LOOP / time.strftime("%m%d-%H%M%S")
    take_token()
    lp = Loop(session)
    lp.op.st["target"] = None
    status(f"data loop session {session.name} started; token taken")
    rec("/start")
    done, fails, reason = 0, 0, "done"
    held_ok = True
    try:
        s = health()
        if s.get("grip_holding"):
            raise Fatal("the jaws are holding something at the start")
        mtime = pathlib.Path(__file__).stat().st_mtime
        for n in range(1, cycles + 1):
            if pathlib.Path(__file__).stat().st_mtime != mtime:  # edited: new methods from the next cycle on
                import importlib
                mtime = pathlib.Path(__file__).stat().st_mtime
                try:
                    mod = importlib.reload(sys.modules[__name__])
                    lp.__class__ = mod.Loop
                    lp.bias = lp.load_bias()
                    status("data_loop.py reloaded at the cycle boundary")
                except Exception as e:  # noqa: BLE001
                    status(f"reload failed, keeping the running code: {e!r}")
            try:
                lp.cycle(n)
                done += 1
                fails = 0
            except (CycleFail, SystemExit, OSError, http.client.HTTPException) as e:
                link = isinstance(e, (OSError, http.client.HTTPException)) or "could not read" in str(e)
                if isinstance(e, SystemExit) and not link:
                    raise  # a refused path or a closed gate: not retried here
                fails += 0 if link else 1
                status(f"cycle {n} STOPPED: {type(e).__name__}: {e}" + ("; link hiccup: waiting for the gate, then the next cycle" if link else ""))
                if link:
                    wait_gate()
                st = arm_state_retry()
                if st["grip"] < 0.75 and st["goal_grip"] < 0.75:
                    raise Fatal(f"stopped holding a block: {e}")
                lp.belief = {}
                lp.op.st["target"] = None
                lp.to_pose(LOOK_TIP, "end")
                if fails >= 2:
                    raise Fatal("two cycles in a row stopped")
            if n % 3 == 0 and n < cycles:
                lp.saves.join()
                status("pause 30 s (3 cycles)")
                time.sleep(30)
            lp.stop_check()
    except (Fatal, SystemExit, KeyboardInterrupt, Exception) as e:  # noqa: BLE001
        reason = f"{type(e).__name__}: {e}"
        try:
            st = arm_state()
            held_ok = st["grip"] > 0.75 or st["goal_grip"] > 0.75 and not st.get("grip_holding")
            if not st["moving"] and held_ok and "GATE" not in str(e):
                lp.op.st["target"] = None
                lp.to_pose(LOOK_TIP, "end")
        except BaseException as e2:  # noqa: BLE001
            reason += f"; end pose failed: {e2}"
            held_ok = False
    finally:
        lp.saves.join()
        rec("/stop")
        if held_ok:
            release_token()
            status(f"session {session.name} ended ({reason}); {done} cycles; token released")
        else:
            (TOKEN / "who.txt").write_text(f"data-loop HELD {time.strftime('%H:%M:%S')}: stopped in an unclean state ({reason})\n")
            status(f"session {session.name} ended ({reason}); {done} cycles; TOKEN KEPT (arm not in a clean state)")
        lp.sync()
        report(session)


def report(session):
    session = pathlib.Path(session)
    if not session.is_absolute():
        session = LOOP / session
    labs = []
    for d in ("dataset", "failures"):
        p = session / d / "labels.jsonl"
        if p.exists():
            labs += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    for sk in ("pick", "place"):
        r = [x for x in labs if x["skill"] == sk]
        ok = sum(x["success"] for x in r)
        print(f"{sk}: {len(r)} episodes, {ok} success, {len(r) - ok} failure" + (f", {ok / len(r) * 100:.0f}%" if r else ""))
    rows = [json.loads(x) for x in open(session / "calib_records.jsonl")] if (session / "calib_records.jsonl").exists() else []
    g = [r for r in rows if r["kind"] == "grasp"]
    for key in ("wrist_err_mm", "overhead_err_mm"):
        e = np.array([r[key] for r in g if r.get(key) is not None])
        if len(e):
            print(f"{key}: {len(e)} grasps, mean {e.mean(0).round(1).tolist()}, sd {e.std(0).round(1).tolist()}, rms "
                  f"{np.sqrt((e ** 2).sum(1).mean()):.1f} mm")
            for name, lo, hi, ax in (("x<33", 0, 0.33, 0), ("x>=33", 0.33, 1, 0), ("y<8", -1, 0.08, 1), ("y>=8", 0.08, 1, 1)):
                sel = [i for i, r in enumerate(r_ for r_ in g if r_.get(key) is not None) if lo <= r["truth_mm"][ax] / 1000 < hi]
                sel = [i for i, r in enumerate([r_ for r_ in g if r_.get(key) is not None]) if lo <= r["truth_mm"][ax] / 1000 < hi]
                if sel:
                    print(f"   {name}: {len(sel)}, mean {e[sel].mean(0).round(1).tolist()} mm")
    m = [r for r in rows if r["kind"] == "move_end"]
    by = {}
    for r in m:
        by.setdefault(r["direction"], []).append(r["tip_err_mm"])
    for k, v in by.items():
        v = np.array(v)
        print(f"command vs actual {k}: {len(v)} moves, mean {v.mean(0).round(1).tolist()} sd {v.std(0).round(1).tolist()}")
    pl = [r for r in rows if r["kind"] == "place" and r.get("offset_mm")]
    if pl:
        o = np.array([r["offset_mm"] for r in pl])
        print(f"place offsets: {len(o)}, mean |off| {np.hypot(*o.T).mean():.1f} mm, max {np.hypot(*o.T).max():.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif sys.argv[1] == "report":
        report(sys.argv[2])
