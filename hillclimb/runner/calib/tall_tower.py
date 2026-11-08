"""One tower, as tall as it goes, from every foam block on the table (real left Metal arm).

Built on calib/data_loop.py's Loop (re-look before every pick, close look at the block below before every place,
contact set-downs, check photo, re-place once, the place bias, episodes recorded on the station) and so on opstack.py.
What is new here:
  - several blocks of one colour: ids like yellow#1, the colour from the prefix; hue bins for the new blocks (yellow
    with its own saturation floor, orange split from the red sticks at H 5); round blobs are balls, poor box fits are
    sticks, both obstacles;
  - picks choose the jaw axis whose open fingers clear the neighbours (blocks can touch), checked along every path
    point low enough to meet a neighbour;
  - the tower spot: every level's place, close look and check photo plan from a carry above it, before anything moves.

    python -m calib.tall_tower look                 # the overlay of every block on the current wrist look (no motion)
    python -m calib.tall_tower dry                  # the whole tower planned on the current look (no motion)
    python -m calib.tall_tower run                  # build it
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import os
import pathlib
import re
import sys
import time

import cv2
import numpy as np

from hillclimb.runner.calib import data_loop as D
from hillclimb.runner.calib import opstack as O
from hillclimb.runner.calib.opstack import GRIP_TIP, LOOK_TIP, PICK_DOWN, TABLE_Z, Op, Plan, arm_state
from hillclimb.runner.calib.tower_plan import FACE, H, Shape, WristView, frame_down
from hillclimb.runner.calib.data_loop import V_DOWN, V_HIGH, V_LAST, bot, hgt, shape_of, top_of

ROOT = O.ROOT
_RUNS_ROOT = pathlib.Path(__import__("os").environ.get("HILLCLIMB_RUNS", str(pathlib.Path.home() / "hillclimb_runs")))
TALL = _RUNS_ROOT / "station" / "tall"
LEARN = _RUNS_ROOT / "station" / "learn"
# the looks that see the whole reachable table (x 15-50, y -30..30 cm) between them, by the wrist camera model
# (python -m calib.look_cover): fingertips x, y, z (m) and the jaw yaw (deg) of the tool frame, held exactly
LOOKS = [(0.37, 0.0, 0.19, 65.0), (0.37, 0.02, 0.19, 105.0)]  # 100 % of the region, 99.9 % with either 1 cm or 3 deg off
LOOKS_OLD = [(LOOK_TIP[0], LOOK_TIP[1] + dy, LOOK_TIP[2], LOOK_TIP[3]) for dy in (0.0, 0.10, -0.10)]  # before Nov 8: 73-81 %
MERGE_R = 0.035  # sightings of one colour this close are one block (two blocks' centres are 5 cm apart or more)
# the push: jaws closed, fingertips 1 cm over the table, down outside the cluster, a 6 cm sweep, straight up
GRIP_PUSH = 0.30
PUSH_LEN = 0.06
PUSH_GAP = 0.010  # the pusher's face starts this far off the block's face
PUSH_HALF = (0.014, 0.022)  # the closed jaws, pads abreast across the push (8 x 26 mm pads 30 mm apart): half along it, half across
TOWER_CLEAR = 0.12
TOUCH = 0.012  # footprints closer than this count as touching (fit error is about 5 mm)
PUSH_REACH = 0.46  # fingertips this far from the base at most, 1 cm over the table (IK solves to 0.46 there)
PUSH_AREA = ((0.17, 0.42), (-0.13, 0.25))  # where a pushed block may end up (reachable for a pick)
STICK = Shape((0.1005, 0.02265, 0.02265), -0.008, period=math.pi)  # a stick lying on the table
BLOCK_COLOURS = ("yellow", "blue", "orange", "purple", "green")  # sticks and balls of these colours fail the box fit
# OpenCV HSV: hue bins, saturation floor (measured on the station's light, Nov 7 19:40)
HUES = {"yellow": ([(22, 38)], 35), "orange": ([(5, 20)], 70), "blue": ([(88, 125)], 70), "red": ([(0, 5), (176, 181)], 70),
        "purple": ([(135, 176)], 55), "green": ([(40, 88)], 55)}
VAL_MIN = 60
PAD_HALF = (0.007, 0.016)  # a pad's half extent (along the closing axis, across it) with 3 mm margin
FINGER = 0.043  # open fingers from the jaw centre at grip 0.8
D.NEAT_MM, D.NEAT_DEG = 6.0, 5.0
# where the whole-move episodes go on the station, under ~/pah/loop/ (FT_ROOT=night_nov8 for the Nov 8 night run;
# its failures in FT_ROOT_failures, the pick and place parts in FT_ROOT/parts/SESSION)
FT_ROOT = os.environ.get("FT_ROOT", "dream_ft_tall")
# the blocks on the table on Nov 8 (Isaiah): green is the big brick, blue, yellow and purple are small; of the two
# orange blocks one is big and one small (the fits decide which)
KNOWN_BIG = {"green": True, "blue": False, "yellow": False, "purple": False}
# TALL_KNOWN_BIG overrides it per table, e.g. '{"green": false, "orange": false}' for the sim (every block a small cube) or
# the venue's own pieces: a wrong entry forces the wrong box on every block of that colour (sim green fitted 0.59, 13 mm off)
KNOWN_BIG.update(json.loads(os.environ.get("TALL_KNOWN_BIG", "{}")))


def colour_of(bid):
    return bid.split("#")[0]


def colour_mask(bgr, colour, jaws):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bins, sat = HUES[colour_of(colour)]
    m = np.zeros(hsv.shape[:2], bool)
    for lo, hi in bins:
        m |= (hsv[..., 0] >= lo) & (hsv[..., 0] < hi)
    m = (m & (hsv[..., 1] >= sat) & (hsv[..., 2] >= VAL_MIN) & (jaws == 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


WristView.colour_mask = staticmethod(colour_mask)


def roundness(blob):
    cs, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    a = cv2.contourArea(c)
    (_, _), r = cv2.minEnclosingCircle(c)
    return a / max(math.pi * r * r, 1), a / max(cv2.contourArea(cv2.convexHull(c)), 1)


def merge_sightings(sights, r=MERGE_R):
    """One entry per block from the sightings of several looks: same colour, same level and within r is the same
    block; the best fit is kept (an edge fit counts 0.1 less), n_looks counts the looks that saw it."""
    out = []
    for s in sorted(sights, key=lambda s: -(s.get("iou", 0) - (0.1 if s.get("edge") else 0))):
        same = [m for m in out if m["colour"] == s["colour"] and abs(bot(m) - bot(s)) < 0.01 and math.hypot(m["x"] - s["x"], m["y"] - s["y"]) < r]
        if same:
            same[0]["n_looks"] = same[0].get("n_looks", 1) + 1
        else:
            out.append(dict(s, n_looks=1))
    return out


def renumber(blocks):
    """colour#n ids, per colour nearest the arm first."""
    out = {}
    for c in sorted({m["colour"] for m in blocks}):
        for j, m in enumerate(sorted([m for m in blocks if m["colour"] == c], key=lambda m: (m["x"], m["y"]))):
            out[f"{c}#{j + 1}"] = {k: v for k, v in m.items() if k != "colour"}
    return out


# ---- footprints on the table, for the cluster phase: (centre, axes as rows, half extents)
def box_of(b, half=None):
    a = b.get("yaw", 0.0)
    ax = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    h = np.array(half if half is not None else D.half_of(b), float)
    return np.array([b["x"], b["y"]], float), ax, h


def box_gap(A, B):
    """Separating-axis gap between two footprints (m): over 0 apart by at least that, under 0 overlapping."""
    (ca, aa, ha), (cb, ab, hb) = A, B
    d = cb - ca
    return max(abs(d @ n) - np.abs(aa @ n) @ ha - np.abs(ab @ n) @ hb for n in (*aa, *ab))


def support(A, d):
    return float(np.abs(A[1] @ d) @ A[2])


def pads_clear(b, others, opening=0.63, margin=0.002):
    """plan_pick's test on footprints: on one jaw axis at this opening, both open pads (PAD_HALF) clear of every other
    block by margin (a block on its side closes across only)."""
    c, ax, h = box_of(b)
    finger = opening * 0.05 + 0.003
    for k in ((1,) if b.get("lying") or b.get("big") else (0, 1)):
        a = ax[k]
        pads = [(c + s * finger * a, np.array([a, [-a[1], a[0]]]), np.array(PAD_HALF)) for s in (-1, 1)]
        if all(box_gap(p, (o[0], o[1], o[2] + margin)) > 0 for p in pads for o in others):
            return True
    return False


def area_out(p):
    """How far (m) a point is outside PUSH_AREA (0 inside): a push may bring a block in, never take it further out."""
    (x0, x1), (y0, y1) = PUSH_AREA
    return math.hypot(max(x0 - p[0], 0, p[0] - x1), max(y0 - p[1], 0, p[1] - y1))


def split_blob(b):
    """The blocks in a weakly fitted blob: n_est (its pixels over one block's silhouette there) centres by k-means on
    its points on the table, each square to the blob's long axis; one block: the fit itself."""
    P = np.asarray(b.get("blob") or [], float)
    n = int(b.get("n_est", 1) or 1)
    if len(P) < 8 or n < 2:
        return [(b["x"], b["y"], b.get("yaw", 0.0))]
    c = P.mean(0)
    yaw = math.radians(cv2.minAreaRect((P * 1000).astype(np.float32))[2])  # the outline's edges (an L's too) are the blocks'
    C = [P[np.argmax(np.linalg.norm(P - c, axis=1))]]  # farthest-point start, then Lloyd
    while len(C) < n:
        C.append(P[np.argmax(np.min([np.linalg.norm(P - q, axis=1) for q in C], 0))])
    C = np.array(C)
    for _ in range(15):
        lab = np.argmin(np.linalg.norm(P[:, None] - C[None], axis=2), 1)
        C = np.array([P[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(n)])
    return [(float(p[0]), float(p[1]), yaw) for p in C]


class AxisPlan(Plan):
    """A Plan whose free frame choice keeps to the jaw axes allowed (k parity: 0 closes along the block's yaw, 1
    across it)."""
    parity = None

    def solve_from(self, ref, xyz, yaw_deg, keep):
        if keep and self.R is not None or self.parity is None:
            return super().solve_from(ref, xyz, yaw_deg, keep)
        from hillclimb.runner.calib.tower_safety import outward_ok
        best = None
        for k in range(4):
            if k % 2 not in self.parity:
                continue
            R = frame_down(math.radians(yaw_deg) + k * math.pi / 2)
            if not outward_ok(R, xyz[:2]):
                continue
            q, ok = self.op.kin.ik_one(np.asarray(xyz, float) + [0, 0, self.op.kin.tip], R, ref)
            if not ok:
                q, _ = self.op.kin.ik(np.asarray(xyz, float) + [0, 0, self.op.kin.tip], [R], ref)
                if q is None:
                    continue
            if not self.op.safe.check_pose(q, self.opening)[0]:
                continue
            c = float(np.abs(q - ref).sum())
            if best is None or c < best[0]:
                best = (c, q, R)
        if best is None:
            raise SystemExit(f"no safe pose for fingertips {np.round(np.asarray(xyz) * 1000).tolist()} yaw {yaw_deg:.0f} on jaw axes {self.parity}")
        self.R = best[2]
        return best[1]


class Tall(D.Loop):
    relook_iou = 0.65  # blocks touching a neighbour fit about 0.7
    # ---- labels: prompts by colour, not id
    def overhead_now(self):
        try:
            data, _, _ = O.fetch("overhead")
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:  # noqa: BLE001
            D.status(f"overhead frame failed: {e}")
            return None

    def knock_check(self, colour, when):
        """The overhead now against the one before this move: (changed, what). Pictures in TALL/SESSION/knock/."""
        from hillclimb.runner.calib import overhead_change as OC
        before, after = getattr(self, "oh_before", None), self.overhead_now()
        if before is None or after is None:
            return True, "no overhead frame to compare"  # cannot tell: treated as a change
        d = self.dir / "knock"
        d.mkdir(exist_ok=True)
        path = d / f"{time.strftime('%H%M%S')}-{colour.replace('#', '')}.jpg"
        ch, what = OC.changed(before, after, colour_of(colour), path)
        self.last_knock_pic = path
        if ch:
            D.status(f"OVERHEAD CHANGE after {when} of {colour}: {what} ({path})")
            self.knocked = (colour, what, str(path))
        return ch, what

    def support_still(self, sup):
        """Right before a place goes down: nothing but the held block changed in the overhead since this move began
        (Nov 8: the overhead fit of the support, its camera about 9 cm off, took the other orange block or found no
        pale yellow and refused every stack)."""
        held = (self.op.held or (sup,))[0]
        ch, what = self.knock_check(held, f"the carry over {sup}")
        O.record("support_still", run=self.dir.name, block=sup, ok=not ch, why=str(what))
        return (not ch), ("overhead unchanged since the move began" if not ch else f"overhead changed: {what}")

    def describe(self, bid, b=None):
        """A block in a prompt: its size, colour and, on its side, that (the big bricks are one green and one orange)."""
        b = self.belief.get(bid, {}) if b is None else b
        return f"{D.size_word(b)} {colour_of(bid)} block" + (", standing on its end," if b.get("upright") else ", on its side," if b.get("lying") else "")

    def under_of(self, pose):
        """The block a pose sat on (its bottom on that block's top, within 3 cm), or None."""
        if not pose or bot(pose) < TABLE_Z + 0.005:
            return None
        return next((c for c, o in self.belief.items() if abs(top_of(o) - bot(pose)) < 0.008
                     and math.hypot(o["x"] - pose["x"], o["y"] - pose["y"]) < 0.03), None)

    def episode(self, t0, t1, root=None, **label):
        bid = label.get("colour")
        src = label.get("start_pose") or {}
        now_b = self.belief.get(bid, {})
        me = dict(now_b, big=src.get("big", now_b.get("big")), lying=src.get("lying", now_b.get("lying", False)))
        sup = (label.get("target") or {}).get("support") if isinstance(label.get("target"), dict) else None
        label["block_id"] = bid
        label["size"] = D.size_word(me)
        label["orientation"] = "on its end" if me.get("upright") else "on its side" if me.get("lying") else "flat"
        label["support_size"] = D.size_word(self.belief.get(sup, {})) if sup else None
        label["speed_mode"] = "A" if D.V_HIGH == SPEEDS["A"][0] else "B"
        label["style"] = getattr(self, "style", None)
        a = self.describe(bid, me)
        if label.get("skill") == "pick":
            label["prompt"] = f"pick up the {a}"
        elif sup and label.get("skill") in ("place", "task"):
            label["prompt"] = f"stack the {a} on top of the {self.describe(sup)}"
        elif label.get("skill") in ("place", "task") and label.get("phase") == "recover":
            label["prompt"] = f"take the {a} out of the pile and put it on the table"
        elif label.get("skill") in ("place", "task"):
            frm = self.under_of(src) if label.get("skill") == "task" else None
            label["prompt"] = (f"take the {a} off the {self.describe(frm)} and put it on the table" if frm
                               else f"put the {a} down on the table")
            label["from_block"] = frm
        label["prompt"] = re.sub(r"#\d+", "", label["prompt"]).replace(",,", ",")
        knocked = None
        if label.get("skill") == "task":
            task = "recover" if label.get("phase") == "recover" else "stack" if sup else ("unstack" if bot(src) > TABLE_Z + 0.005 else "move")
            label["task"] = task
            ch, what = self.knock_check(bid, "the move")
            label["overhead_check"] = {"changed": ch, "what": what, "picture": str(getattr(self, "last_knock_pic", ""))}
            if ch:
                label["success"], knocked = False, what
            root = f"{FT_ROOT}/{task}" if label.get("success") else f"{FT_ROOT}_failures/{task}"
        elif root is None:
            root = f"{FT_ROOT}/parts/{self.dir.name}/" + ("dataset" if label.get("success") else "failures")
        elif root.startswith("dream_ft"):
            root = root.replace("dream_ft", FT_ROOT, 1)
        out = super().episode(t0, t1, root=root, **label)
        if knocked is not None:
            raise D.Fatal(f"a block that was not moved changed in the overhead during the {bid} move: {knocked}")
        return out

    def relook(self, colour):
        """Re-look before a pick; a fit under 0.8 that moved more than 2 cm is not believed: a block the arm placed
        keeps its placed pose, anything else is refused."""
        b = self.belief[colour]
        before = dict(b)
        recent = [pl for pl in getattr(self, "places", []) if pl[2] > b.get("t_seen", 0) and math.hypot(pl[0] - b["x"], pl[1] - b["y"]) < 0.10]
        if time.time() - b.get("t_seen", 0) < 30 and not recent:
            return
        from hillclimb.runner.calib.look_cover import in_view
        q_std = getattr(self, "q_std", None)
        if q_std is not None and not in_view(self.op.view, q_std, [b])[0]:
            self.relook_far(colour)  # out of the standard look's view: from the LOOKS pose that sees it
        else:
            super().relook(colour)
        a = self.belief[colour]
        if a.get("iou", 1.0) < 0.55 and math.hypot(a["x"] - before["x"], a["y"] - before["y"]) > 0.02:
            if before.get("placed"):
                D.status(f"relook {colour}: fit {a['iou']:.2f} moved {math.hypot(a['x'] - before['x'], a['y'] - before['y']) * 1000:.0f} mm: keeping its placed pose")
                self.belief[colour] = before
            else:
                raise D.CycleFail(f"{colour}: re-look fit {a['iou']:.2f} is {math.hypot(a['x'] - before['x'], a['y'] - before['y']) * 1000:.0f} mm from where it was: refused")
        self.belief[colour]["t_seen"] = time.time()

    def relook_far(self, colour):
        """The re-look of a block the standard look does not see (far out, close in by the base): from the first LOOKS
        pose that has it in full view, in that pose's own frame."""
        from hillclimb.runner.calib.look_cover import in_view
        b = self.belief[colour]
        poses = [p for p in LOOKS if self.look_q(p) is not None and in_view(self.op.view, self.look_q(p), [b])[0]]
        if not poses:
            raise D.CycleFail(f"{colour}: no look pose sees ({b['x'] * 100:.1f}, {b['y'] * 100:.1f})")
        self.look_to(poses[0], "look")
        time.sleep(0.25)
        img, q, grip, stem = self.op.snap(f"relook-{colour.replace('#', '')}")
        f = self.op.view.find(img, colour, q, grip * 0.1, shape_of(b), (b["x"], b["y"], b["yaw"]))
        if f is None or f["iou"] < self.relook_iou or math.hypot(f["x"] - b["x"], f["y"] - b["y"]) > 0.04:
            raise D.CycleFail(f"{colour}: not found where it was seen ({f})")
        b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"], edge=f["edge"])
        self.put_blocks()
        D.status(f"relook {colour} from ({poses[0][0] * 100:.0f}, {poses[0][1] * 100:.0f}) yaw {poses[0][3]:.0f}: "
                 f"({f['x'] * 100:.1f}, {f['y'] * 100:.1f}) iou {f['iou']:.2f}")

    def failure(self, bid, target, err):
        """runs/station/learn/failures/<time>-<id>.json: the refusal or error, a category, the belief and the frames
        around it (the last two snaps of this session)."""
        try:
            txt = f"{type(err).__name__}: {err}"
            low = txt.lower()
            cat = ("reach" if "no safe pose" in low or "no ik" in low else "clearance" if "pad meets" in low or "neighbours" in low or "sweep" in low
                   or "camera within" in low else "perception" if "not found" in low or "not seen" in low else "grasp" if "grasp" in low or "slipped" in low
                   else "place" if "off" in low or "tipped" in low else "code")
            d = LEARN / "failures"
            d.mkdir(parents=True, exist_ok=True)
            snaps = sorted(self.op.run.glob("*.json"))
            snaps = [str(x.with_suffix("")) for x in snaps if x.name != "op.json"][-2:]
            (d / f"{time.strftime('%H%M%S')}-{bid.replace('#', '')}.json").write_text(json.dumps(D.jdump({
                "t": time.strftime("%Y-%m-%d %H:%M:%S"), "block": bid, "target": target, "category": cat, "error": txt,
                "belief": self.belief, "frames": snaps, "session": self.dir.name}), indent=1))
        except Exception as e:  # noqa: BLE001
            print("failure record failed:", repr(e), flush=True)

    def block(self, colour, target, *a, **k):
        outer = not getattr(self, "in_block", False)
        if outer:  # the overhead before the move (a recovery inside it keeps the first one)
            self.in_block, self.oh_before = True, self.overhead_now()
            self.speed_scale()
        try:
            out = super().block(colour, target, *a, **k)
        except (D.CycleFail, SystemExit) as e:
            self.failure(colour, target, e)
            if outer and self.knock_check(colour, "failed move")[0]:
                raise D.Fatal(f"a block that was not moved changed in the overhead during the failed {colour} move") from e
            raise
        finally:
            if outer:
                self.in_block = False
        self.places = getattr(self, "places", []) + [(target[0], target[1], time.time())]
        import threading
        beliefs = {kk: dict(v) for kk, v in self.belief.items()}
        threading.Thread(target=self.after_place, args=(colour, beliefs), daemon=True).start()
        if target[4] and not getattr(self, "no_perception", False):  # a stack place: the perception set at this tower height
            try:
                self.perception_set(colour, beliefs)
            except (SystemExit, D.CycleFail) as e:
                D.status(f"perception set skipped: {str(e)[:200]}")
                self.op.st["target"] = None
        return out

    def perception_set(self, top, beliefs):
        """runs/station/learn/perception/<time>-h<N>/: wrist frames from 3 jittered look poses and the overhead frame,
        with the truth (every block's id, level, bottom, flat or on its side, x, y, yaw from the place records and
        fits). The scene reader runs on the first frame in the background and its answer is saved beside the truth."""
        import threading
        h = 1 + int(round((bot(beliefs[top]) - TABLE_Z) / H))
        d = LEARN / "perception" / f"{time.strftime('%H%M%S')}-h{h}"
        d.mkdir(parents=True, exist_ok=True)
        truth = {k: {kk: b.get(kk) for kk in ("x", "y", "yaw", "level", "z0", "lying", "iou")} for k, b in beliefs.items() if not k.startswith("pile")}
        (d / "truth.json").write_text(json.dumps(D.jdump({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "top": top, "height_blocks": h, "blocks": truth}), indent=1))
        for i in range(3):
            tip = self.jitter_look(0.03, 10.0)
            self.to_pose(tip, "look")
            time.sleep(0.15)
            img, q, grip, stem = self.op.snap(f"perc{i}")
            for cam in ("wrist", "overhead"):
                (d / f"{i}-{cam}.jpg").write_bytes((self.op.run / f"{stem}-{cam}.jpg").read_bytes())
            (d / f"{i}.json").write_text((self.op.run / f"{stem}.json").read_text())
        known = "tower bottom first: " + ", ".join(k for k, _ in sorted(truth.items(), key=lambda kv: kv[1].get("z0") or 0)
                                                  if math.hypot(truth[k]["x"] - beliefs[top]["x"], truth[k]["y"] - beliefs[top]["y"]) < 0.03)

        # (a slower scene description was saved here in the original runner; not used by the harness)

    def after_place(self, bid, beliefs):
        """runs/station/learn/after_place/<time>-<id>.jpg: the check wrist frame and the overhead frame side by side,
        every believed block as a 3D wireframe box (level, flat or on its side) labelled by id, the jaw CAD in the
        wrist view. Overhead through calib/arm_left.json's camera (about 10 cm off, shown as is)."""
        try:
            from vision.locate import Camera
            stems = sorted(self.op.run.glob(f"*-check-{bid}.json"))
            if not stems:
                return
            stem = stems[-1].name[:-5]
            meta = json.loads(stems[-1].read_text())
            q = np.array(meta["wrist"]["joints"], float) + self.op.dq
            grip = float(meta["wrist"]["grip"] or 0.8)
            wr = cv2.resize(cv2.imread(str(self.op.run / f"{stem}-wrist.jpg")), (640, 360), interpolation=cv2.INTER_AREA)
            ov = cv2.resize(cv2.imread(str(self.op.run / f"{stem}-overhead.jpg")), (640, 360), interpolation=cv2.INTER_AREA)
            cam_w = self.op.view.pose(q, grip * 0.1)
            js = json.loads((ROOT / "calib" / "arm_left.json").read_text())["camera"]
            cam_o = Camera.from_calib(js)
            edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
            cs, _ = cv2.findContours(self.op.view.jaws_mask(q, grip * 0.1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(wr, cs, -1, (255, 224, 63), 1, cv2.LINE_AA)
            for k, b in beliefs.items():
                sh = shape_of(b)
                P = sh.corners(b["x"], b["y"], b["yaw"])
                col = (40, 170, 245) if k != bid else (60, 255, 60)
                uvw, okw = self.op.view.project(cam_w, P)
                uvo = cam_o.project(P) * 0.5
                for uv, ok, img in ((uvw, okw, wr), (uvo, np.ones(len(P), bool), ov)):
                    if not ok.all():
                        continue
                    for i, j in edges:
                        a_, b_ = uv[i], uv[j]
                        if np.all(np.abs([a_, b_]) < 5000):
                            cv2.line(img, tuple(int(v) for v in a_), tuple(int(v) for v in b_), col, 1, cv2.LINE_AA)
                    c = uv.mean(0)
                    if np.all(np.abs(c) < 5000):  # labels to the right of the box, one line per level so stacks stay readable
                        lv = int(round((bot(b) - TABLE_Z) / H))
                        at = (int(uv[:, 0].max()) + 4, int(c[1]) - 12 * lv)
                        txt = f"{k} {'side' if b.get('lying') else 'flat'} z{bot(b) * 1000:.0f}"
                        cv2.putText(img, txt, at, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(img, txt, at, cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            out = np.hstack([wr, ov])
            cv2.putText(out, f"{time.strftime('%H:%M:%S')} after placing {bid} (wrist check, overhead)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            d = LEARN / "after_place"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}-{bid.replace('#', '')}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:  # noqa: BLE001
            print("after_place failed:", repr(e), flush=True)

    def sync(self):
        """The original runner copied datasets to a laptop here; on the station there is nothing to copy."""
        self.saves.join()

    # ---- finding every block
    def find_blocks(self, img, q, grip):
        """Every block in one wrist frame, fitted flat or on its side at table level; a flat fit one level up when it
        is clearly better and a weakly fitted block of another colour sits under it (a pile). Rows of touching blocks of
        one colour (blobs over 2.6 long but block wide) are fitted too, weakly: the cluster phase pushes them apart.
        Each block keeps its blob's outline on the table ("blob", m) for that."""
        view = self.op.view
        opening = grip * 0.1
        cam = view.pose(q, opening)
        jaws = view.jaws_mask(q, opening)
        blocks, obst, rejected = [], [], []
        flat = Shape((FACE, FACE, H), TABLE_Z)
        flat1 = Shape((FACE, FACE, H), TABLE_Z + H)
        lying = shape_of({"lying": True}, TABLE_Z)
        bigs = (shape_of({"big": True}, TABLE_Z), shape_of({"big": True, "lying": True}, TABLE_Z))
        upright = shape_of({"big": True, "upright": True}, TABLE_Z)  # a brick on its end (Nov 8 03:03: green tipped up)
        for c in BLOCK_COLOURS + ("red",):
            m = colour_mask(img, c, jaws)
            n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < 150:
                    continue
                blob = lab == i
                ys, xs = np.nonzero(blob)
                pts = view.ray_on_plane(cam, np.stack([xs, ys], 1)[:: max(1, len(xs) // 200)], TABLE_Z + 0.015)[:, :2]
                fill, solid = roundness(blob)
                kind = "ball" if (fill > 0.78 and solid > 0.93) else None
                f = None
                (_, _), (rw, rh), _ = cv2.minAreaRect((pts * 1000).astype(np.float32))
                aspect = max(rw, rh) / max(min(rw, rh), 1e-3)
                wide = min(rw, rh) >= 38  # mm: a block's width or more (the sticks are 23 mm)
                if c in BLOCK_COLOURS and kind is None and (aspect <= 2.6 or wide):
                    f = view._fit(cam, jaws, lab, stats, i, flat, None)
                    fl = view._fit(cam, jaws, lab, stats, i, lying, None)
                    fs = view._fit(cam, jaws, lab, stats, i, STICK, None) if not wide else None
                    # on its side only when clearly better: a pale lit top (yellow) drops out of the mask and a flat block
                    # then fits "on its side" a little better (Nov 8 01:10); in the loop blocks only lie on a side after a fall
                    on_side = fl is not None and (f is None or fl["iou"] > f["iou"] + 0.10)
                    if on_side:
                        f = fl
                    # a big brick (83 x 50 mm): only when it fits well and clearly better than the small block's fits
                    big, small_fit, big_fit, up_end = False, (f, on_side), None, False
                    if (wide or KNOWN_BIG.get(c)) and KNOWN_BIG.get(c) is not False:
                        fbs = [view._fit(cam, jaws, lab, stats, i, sh, None) for sh in bigs]
                        # on its side only when clearly better than flat (as for the small blocks: 02:40 green lay flat, its
                        # hole up, and fitted "side")
                        jb = 1 if fbs[1] is not None and (fbs[0] is None or fbs[1]["iou"] > fbs[0]["iou"] + 0.10) else 0
                        jb = jb if fbs[jb] is not None else 1 - jb
                        big_fit = (fbs[jb], jb == 1) if fbs[jb] is not None else None
                        fu = view._fit(cam, jaws, lab, stats, i, upright, None)
                        best_b = max((x for x in fbs if x is not None), key=lambda x: x["iou"], default=None)
                        if fu is not None and fu["iou"] >= 0.7 and (best_b is None or fu["iou"] > best_b["iou"] + 0.08) and \
                                (f is None or fu["iou"] > f["iou"] + 0.08):
                            f, on_side, big, up_end = fu, False, True, True  # standing on its end: clearly better than lying
                        elif KNOWN_BIG.get(c) and fbs[jb] is not None and fbs[jb]["iou"] >= 0.5:
                            f, on_side, big = fbs[jb], jb == 1, True  # the only green block is the big one
                        elif fbs[jb] is not None and fbs[jb]["iou"] >= 0.8 and (f is None or fbs[jb]["iou"] > f["iou"] + 0.06):
                            f, on_side, big = fbs[jb], jb == 1, True
                    if fs is not None and f is not None and fs["iou"] > f["iou"]:  # a lying stick fits it better
                        f = None
                    up = None
                    if f is not None and not on_side and not big and f["iou"] < 0.9:
                        f1 = view._fit(cam, jaws, lab, stats, i, flat1, None)
                        if f1 is not None and f1["iou"] >= 0.8 and f1["iou"] > f["iou"] + 0.06:
                            up = f1
                    # 0.62: a block partly hidden behind a touching neighbour still fits about 0.7; a row of touching
                    # blocks fits worse still, but is kept (weak) for the cluster phase
                    if f is not None and (f["iou"] >= 0.50 or (wide and f["iou"] >= 0.25)) and 0.07 < f["x"] < 0.52 and -0.32 < f["y"] < 0.40:
                        # how many blocks the blob is: its pixels over one block's silhouette there
                        one = view.silhouette(cam, bigs[0] if big else flat, f["x"], f["y"], f["yaw"])
                        n_est = int(np.clip(round(stats[i, cv2.CC_STAT_AREA] / max(cv2.contourArea(one), 1.0)), 1, 4)) if one is not None else 1
                        margin = (big_fit[0]["iou"] if big_fit else 0.0) - (small_fit[0]["iou"] if small_fit[0] else 0.0)
                        blocks.append(dict(f, colour=c, level=0, z0=TABLE_Z, lying=bool(on_side), big=big, upright=up_end, px=int(stats[i, cv2.CC_STAT_AREA]),
                                           big_margin=margin, alt=(small_fit if big else big_fit),
                                           blob=np.round(pts[:: max(1, len(pts) // 80)], 4).tolist(), n_est=n_est, up=up))
                        continue
                    kind = "stick or cut block"
                elif aspect > 2.6:
                    kind = f"stick (aspect {aspect:.1f})"
                obst.append(pts)
                rejected.append({"colour": c, "kind": kind or "stick", "px": int(stats[i, cv2.CC_STAT_AREA]),
                                 "iou": None if f is None else round(f["iou"], 2), "at_mm": (pts.mean(0) * 1000).round().tolist()})
        # a better fit one level up is a pile only over a weakly fitted block of another colour (the one under it), or
        # over a sliver of another block colour too small to fit at all (Nov 8 01:59: blue on the green brick hid all
        # but 750 px of it, fitted 0.29 and dropped; blue fitted 0.88 one up, 0.70 on the table)
        slivers = [(r["colour"], np.array(r["at_mm"]) / 1000) for r in rejected if r["colour"] in BLOCK_COLOURS and r.get("at_mm")]
        for b in blocks:
            up = b.pop("up", None)
            if up is not None and (any(o is not b and o["colour"] != b["colour"] and o["iou"] < 0.8 and
                                       math.hypot(o["x"] - up["x"], o["y"] - up["y"]) < 0.045 for o in blocks)
                                   or any(c != b["colour"] and math.hypot(at[0] - up["x"], at[1] - up["y"]) < 0.06 for c, at in slivers)):
                b.update(x=up["x"], y=up["y"], yaw=up["yaw"], iou=up["iou"], edge=up["edge"], level=1, z0=TABLE_Z + H)
        # two orange blocks read the same size: the one the brick fits better (over its small fit) is the big one
        oranges = [b for b in blocks if b["colour"] == "orange" and b.get("level", 0) == 0]
        if len(oranges) == 2 and oranges[0]["big"] == oranges[1]["big"]:
            want = max(oranges, key=lambda b: b["big_margin"])
            for b in oranges:
                if b["big"] != (b is want) and b.get("alt") and b["alt"][0] is not None:
                    fit, side = b["alt"]
                    b.update(x=fit["x"], y=fit["y"], yaw=fit["yaw"], iou=fit["iou"], edge=fit["edge"], lying=bool(side), big=b is want)
                else:
                    b["big"] = b is want
        # ids: per colour, nearest the arm first
        out = {}
        for c in BLOCK_COLOURS:
            for j, b in enumerate(sorted([b for b in blocks if b["colour"] == c], key=lambda b: (b["x"], b["y"]))):
                out[f"{c}#{j + 1}"] = {k: b[k] for k in ("x", "y", "yaw", "level", "iou", "edge", "z0", "lying", "big", "upright", "blob", "n_est")}
        o = np.concatenate(obst) if obst else np.zeros((0, 2))
        o = o[(o[:, 0] > 0.15) & (o[:, 0] < 0.55) & (o[:, 1] > -0.32) & (o[:, 1] < 0.35)]
        for b in out.values():
            o = o[np.hypot(o[:, 0] - b["x"], o[:, 1] - b["y"]) > 0.045]
        return out, o, rejected, jaws

    def look_q(self, pose):
        """Model joints of a look pose in its held frame (IK from the look seed), cached; None if it has none."""
        self._look_q = getattr(self, "_look_q", {})
        key = tuple(round(v, 4) for v in pose[:4])
        if key not in self._look_q:
            from hillclimb.runner.calib.look_cover import Cover
            cv_ = Cover.__new__(Cover)
            cv_.kin, cv_.view, cv_.safe = self.op.kin, self.op.view, self.op.safe
            self._look_q[key] = cv_.solve(pose[:3], pose[3])
        return self._look_q[key]

    def look_to(self, pose, tag="look"):
        """Fingertips to a look pose in exactly its frame (frame_down(yaw): the camera sits on the jaws' +y side, so
        another of the four symmetric frames sees another part of the table). If that frame does not plan from here,
        to_pose's free frame (logged: the look then sees less)."""
        x, y, z, yaw = pose[:4]
        R = frame_down(math.radians(yaw))
        last = None
        for attempt in range(2):
            q0 = self.start_q()
            if np.linalg.norm(self.op.tip(q0)[0] - np.array([x, y, z])) < 0.005 and np.allclose(self.op.kin.fk(q0)[1], R, atol=0.02):
                return
            try:
                pl = Plan(self.op, q0, 0.08)
                pl.R = R
                pl.to([x, y, z], yaw, D.V_HIGH, tag, keep=True)
                if not np.allclose(pl.R, R, atol=1e-6):
                    raise SystemExit("the held frame has no safe pose")
                pl.check()
                self.sweep(pl, exempt=set())
                self.go(pl, 0.8, tag, "level")
                return
            except SystemExit as e:
                last = e
                if attempt == 0:  # from the folded rest or a low pose: to the spot in the nearest frame, then turn
                    self.to_pose((x, y, z, yaw), tag)
        D.status(f"{tag}: ({x * 100:.0f}, {y * 100:.0f}, {z * 100:.0f}) cm yaw {yaw:.0f} in its own frame does not plan ({str(last)[:120]}); "
                 "looking in the nearest frame (this look sees less)")

    def multi_look(self, poses=None, tag="look", held=True):
        """The looks in LOOKS (between them the whole reachable table), each in its held frame (held=False: to_pose's
        nearest frame, as before Nov 8); every block of every look fitted (no extra vision call per look), sightings merged by
        position (one colour within 3.5 cm: the better fit, edge fits last), ids renumbered per colour; then one batched
        vet of the weak green and purple fits; obstacles from every look. Ends at the standard look pose (its view is
        the one the spot visibility uses)."""
        poses = LOOKS if poses is None else poses
        self.clear_up()
        sights, obst, frames = [], [], []
        for i, pose in enumerate(poses):
            try:
                self.look_to(pose, tag) if held else self.to_pose(pose, tag)
            except SystemExit as e:
                D.status(f"look {i} skipped: {str(e)[:120]}")
                continue
            time.sleep(0.25)
            blocks, rejected, path = self.look_all(f"{tag}{i}", move=False, vet=False)
            frames.append({"path": str(self.last_look_img), "q": self.q_look, "grip": self.look_grip})
            if len(self.obstacles):
                obst.append(self.obstacles)
            for k, b in blocks.items():
                sights.append(dict(b, colour=colour_of(k), frame=len(frames) - 1))
        out = renumber(merge_sightings(sights))
        out = self.vet_batch(out, frames)
        self.look_frames = frames
        self.to_pose(LOOK_TIP, "look")
        self.belief = out
        self.obstacles = np.concatenate(obst) if obst else np.zeros((0, 2))
        for b in out.values():
            if len(self.obstacles):
                self.obstacles = self.obstacles[np.hypot(self.obstacles[:, 0] - b["x"], self.obstacles[:, 1] - b["y"]) > 0.045]
        img, q, grip, stem = self.op.snap("look-std")
        self.q_look, self.jaws_look = q, self.op.view.jaws_mask(q, grip * 0.1)
        self.q_std = q
        self.put_blocks()
        D.status(f"{len(frames)} looks: {len(out)} blocks " + ", ".join(
            f"{k} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}){' L1' if bot(b) > TABLE_Z + 0.005 else ''} iou {b.get('iou', 0):.2f} x{b.get('n_looks', 1)}"
            for k, b in out.items()))
        return out

    def vet_batch(self, blocks, frames):
        """Optional second check of weak green or purple fits. Off in this package: every fit is kept (the harness's
        vision checks run outside the runner)."""
        return blocks

    def clear_up(self):
        """At a session's start nothing is known of the scene yet: if the fingertips are under 20 cm, the first move is
        straight up (the same orientation, checked), so nothing standing under or beside the hand is swept."""
        if getattr(self, "cleared", False):
            return
        self.cleared = True
        q0 = self.start_q()
        p0, yaw0 = self.op.tip(q0)
        if p0[2] >= 0.15:  # at or near the look height already (19 cm)
            return
        s0 = arm_state()
        near = [np.abs(np.array(s0["q"]) - np.array(s0[k])).max() for k in ("start_joints", "zero_joints") if s0.get(k)]
        if near and min(near) < 0.15:
            return  # folded at rest: the look move unfolds it under the home rules
        up = Plan(self.op, q0, 0.08)
        try:
            up.to([p0[0], p0[1], min(p0[2] + 0.06, 0.19)], yaw0, 0.25, "straight up", line=0.01)
            up.check()
        except SystemExit as e:
            D.status(f"session start: no straight-up move from {p0[2] * 1000:.0f} mm ({str(e)[:120]}); the look move's own sweep decides")
            return
        D.status(f"session start: fingertips at {p0[2] * 1000:.0f} mm, going straight up first")
        self.go(up, float(np.clip(s0["goal_grip"], 0.0, 1.0)), "straight up", "up")

    def look_all(self, tag, tip=LOOK_TIP, move=True, vet=True):
        if move:
            self.clear_up()
            self.to_pose(tip, "look")
            time.sleep(0.25)
        img, q, grip, stem = self.op.snap(tag)
        self.last_look_img = self.op.run / f"{stem}-wrist.jpg"
        self.look_grip = grip
        blocks, obst, rejected, jaws = self.find_blocks(img, q, grip)
        if vet and not getattr(self, "no_vet", False):
            blocks = self.vet_blocks(blocks, q, grip, self.op.run / f"{stem}-wrist.jpg")
        for b in blocks.values():
            b["t_seen"] = time.time()
        self.belief, self.obstacles = blocks, obst
        self.q_look, self.jaws_look = q, jaws
        self.put_blocks()
        try:
            self.look_overhead = O.overhead_blocks((self.op.run / f"{stem}-overhead.jpg").read_bytes())
        except Exception:  # noqa: BLE001
            self.look_overhead = {}
        out = self.op.overlay(img, q, grip * 0.1, {bid: (dict(b), shape_of(b)) for bid, b in blocks.items()}, stem)
        cam = self.op.view.pose(q, grip * 0.1)
        for p in obst[::4]:
            uv, ok = self.op.view.project(cam, [[p[0], p[1], TABLE_Z + 0.011]])
            if ok.all():
                cv2.circle(out, tuple(int(v) for v in uv[0]), 2, (0, 0, 255), -1)
        for bid, b in blocks.items():
            uv, ok = self.op.view.project(cam, [[b["x"], b["y"], TABLE_Z + H]])
            if ok.all():
                cv2.putText(out, bid, (int(uv[0][0]) - 20, int(uv[0][1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(out, bid, (int(uv[0][0]) - 20, int(uv[0][1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        path = self.op.run / f"{stem}-blocks.jpg"
        cv2.imwrite(str(path), out)
        O.record("look", run=self.dir.name, stem=stem, wrist=blocks, rejected=rejected, overhead=self.look_overhead)
        return blocks, rejected, path

    def vet_blocks(self, blocks, q, grip, wrist_path):
        """Optional second check of weak green or purple fits in one wrist frame. Off in this package: every fit kept."""
        return blocks

    def towers(self):
        """Centres (x, y) of everything standing two or more high (recorded stacks, piles), to keep clear of."""
        out = []
        for k, b in self.belief.items():
            if bot(b) > TABLE_Z + 0.005 or k.startswith("pile"):
                if not any(math.hypot(b["x"] - x, b["y"] - y) < 0.04 for x, y in out):
                    out.append((b["x"], b["y"]))
        return out

    def objects(self):
        """The blocks on the table as footprints for push planning; a weak fit whose blob is two or three blocks long
        becomes its parts (id/0, id/1, ...)."""
        objs = []
        for k, b in self.belief.items():
            if k.startswith("pile") or bot(b) > TABLE_Z + 0.005 or any(math.hypot(b["x"] - tx, b["y"] - ty) < 0.03 for tx, ty in self.towers()):
                continue
            parts = split_blob(b) if b.get("iou", 1.0) < 0.8 and not b.get("lying") else [(b["x"], b["y"], b["yaw"])]
            for j, (x, y, yaw) in enumerate(parts):
                objs.append({"id": k if len(parts) == 1 else f"{k}/{j}", "block": k, "x": x, "y": y, "yaw": yaw, "lying": b.get("lying", False),
                             "iou": b.get("iou", 1.0), "part": len(parts) > 1})
        return objs

    def needs_push(self, objs):
        """Blocks to separate: a weak fit that is several blocks or has a neighbour within 7.5 cm, a block touching
        another (footprints under TOUCH apart), or one no jaw axis can pick (plan_pick's pad test at 63 mm)."""
        need = []
        for o in objs:
            others = [box_of(p) for p in objs if p is not o]
            near = any(math.hypot(p["x"] - o["x"], p["y"] - o["y"]) < 0.075 for p in objs if p is not o)
            touch = any(box_gap(box_of(o), q) < TOUCH for q in others)
            if o["part"] or (o["iou"] < 0.8 and near) or touch or not pads_clear(o, others):
                need.append(o)
        return need

    def push_candidates(self, need, objs):
        """Every push of a block to separate, scored. A push: the closed jaws come down PUSH_GAP off the block's face on
        the side away from the push, then sweep PUSH_LEN (the block moves about 5 cm); the pads abreast across the push
        (PUSH_HALF) or in line with it (narrower: it fits between blocks). Kept only when the jaws come down clear of
        every footprint, sweep into no block but this one, the block slides into no other block, ends in PUSH_AREA, the
        fingertips keep to y over -14.5 cm, and jaws and block stay TOWER_CLEAR from every tower. Score: blocks of the
        cluster pickable after (pads_clear on the moved footprints), blocks left touching nothing, room round the
        pushed block (ending against another block costs 1.5), near the reach's middle, pads abreast."""
        towers = self.towers()
        obst = self.obstacles if len(self.obstacles) else np.zeros((0, 2))
        out = []
        for o in need:
            A = box_of(o)
            cluster = [p for p in objs if p is not o and box_gap(A, box_of(p)) < 0.03]
            for th in range(0, 360, 20):
                d = np.array([math.cos(math.radians(th)), math.sin(math.radians(th))])
                perp = np.array([-d[1], d[0]])
                for inline in (False, True):
                    half = np.array(PUSH_HALF[::-1] if inline else PUSH_HALF)
                    start = A[0] - d * (support(A, d) + PUSH_GAP + half[0])
                    end = start + d * PUSH_LEN
                    disp = PUSH_LEN - PUSH_GAP
                    sweep_pts = [start + d * PUSH_LEN * f for f in np.linspace(0, 1, 7)]
                    why = None
                    # footprints that already overlap (estimates of touching blocks) may slide along each other, never
                    # press further in
                    g0 = {p["id"]: min(box_gap(A, box_of(p)), 0.0) for p in objs if p is not o}
                    if any(box_gap((start, np.array([d, perp]), half), box_of(p)) < 0.004 for p in objs):
                        why = "jaws come down on a block"
                    elif any(box_gap((c_, np.array([d, perp]), half), box_of(p)) < 0.002 for c_ in sweep_pts for p in objs if p is not o):
                        why = "jaws sweep into another block"
                    elif any(box_gap((A[0] + d * disp * f, A[1], A[2]), box_of(p)) < g0[p["id"]] - 0.004
                             for f in (0.25, 0.5, 0.75, 1.0) for p in objs if p is not o):
                        why = "block shoves another"
                    new = A[0] + d * disp
                    if why is None and area_out(new) > max(area_out(A[0]), 0.0) + 1e-6:
                        why = "block leaves the reach"
                    if why is None and min(start[1], end[1]) < O.TIP_Y_MIN + 0.005:
                        why = "fingertips past y -14.5 cm"
                    if why is None and max(math.hypot(*start), math.hypot(*end)) > PUSH_REACH:
                        why = "past the reach 1 cm over the table"
                    if why is None and any(math.hypot(p[0] - tx, p[1] - ty) < TOWER_CLEAR for p in sweep_pts + [new] for tx, ty in towers):
                        why = "near a tower"
                    if why is None and len(obst):
                        for c_ in sweep_pts:
                            dd = obst - c_
                            if np.any((np.abs(dd @ d) < half[0] + 0.01) & (np.abs(dd @ perp) < half[1] + 0.01)):
                                why = "a stick or ball in the way"
                                break
                        dd = obst - new
                        if why is None and np.any(np.max(np.abs(dd @ A[1].T) - A[2], 1) < 0.01):
                            why = "block ends on a stick or ball"
                    if why:
                        self.push_why[why] += 1
                        continue
                    moved = [dict(p) for p in objs]
                    for p in moved:
                        if p["id"] == o["id"]:
                            p["x"], p["y"] = float(new[0]), float(new[1])
                    mine = [p for p in moved if p["id"] == o["id"] or any(p["id"] == q["id"] for q in cluster)]
                    n_ok = sum(pads_clear(p, [box_of(r) for r in moved if r is not p]) for p in mine)
                    room = min([box_gap((new, A[1], A[2]), box_of(p)) for p in objs if p is not o] or [0.05])
                    n_free = sum(min([box_gap(box_of(p), box_of(r)) for r in moved if r is not p] or [1]) >= TOUCH for p in mine)
                    score = n_ok + n_free + min(max(room, 0), 0.04) / 0.04 - (1.5 if room < TOUCH else 0) \
                        - 0.3 * math.hypot(new[0] - 0.28, new[1] - 0.03) / 0.2 - (0.1 if inline else 0)
                    out.append((score, o, start, end, inline, n_ok, len(mine)))
        out.sort(key=lambda c: -c[0])
        return out

    def plan_push(self, start, end, exempt, inline=False):
        """Closed jaws over the start, straight down to 1 cm over the table (1.3 or 1.6 if the table rule needs it),
        the sweep, straight up. Pads abreast: the jaw axis across the push; in line: along it (either of the two
        frames that way, whichever plans)."""
        q0 = self.start_q()
        d = (np.asarray(end) - start) / np.linalg.norm(np.asarray(end) - start)
        ang = math.degrees(math.atan2(d[1], d[0]))
        top = max([top_of(b) for b in self.belief.values() if math.hypot(b["x"] - start[0], b["y"] - start[1]) < 0.12] or [TABLE_Z])
        last = None
        # over the start 4 cm above what is near (3 cm when far out, where the reach is low), then straight down
        for z_hi, dz in ((max(0.07, top + 0.04), 0.010), (max(0.07, top + 0.04), 0.013), (top + 0.03, 0.010), (max(0.07, top + 0.04), 0.016)):
            z = TABLE_Z + dz
            for k in ((1, 3) if inline else (0, 2)):
                yaw = ang + 90 + 90 * k
                try:
                    R = frame_down(math.radians(yaw))
                    pl = Plan(self.op, q0, 0.03)
                    pl.R = R
                    pl.to([start[0], start[1], z_hi], yaw, D.V_HIGH, "over push", keep=True)
                    if not np.allclose(pl.R, R, atol=1e-6):
                        raise SystemExit("frame has no safe pose")
                    pl.to([start[0], start[1], z], yaw, D.V_DOWN, "push down", line=0.01, keep=True)
                    pl.to([end[0], end[1], z], yaw, 0.12, "push", line=0.005, keep=True)
                    pl.to([end[0], end[1], z_hi], yaw, D.V_DOWN, "push up", line=0.01, keep=True)
                    pl.check()
                    self.sweep(pl, exempt)
                    return pl, dz, yaw
                except SystemExit as ex:
                    last = ex
        raise SystemExit(f"push: {last}")

    def push(self, start, end, exempt, inline=False, note=""):
        pl, dz, yaw = self.plan_push(start, end, exempt, inline)
        D.status(f"push{note}: closed jaws down at ({start[0] * 100:.1f}, {start[1] * 100:.1f}) cm, {dz * 1000:.0f} mm over the table, "
                 f"sweep to ({end[0] * 100:.1f}, {end[1] * 100:.1f}), pads {'in line' if inline else 'abreast'}, jaw yaw {yaw % 360:.0f}")
        self.go(pl, GRIP_PUSH, "push", "push")
        self.pushes = getattr(self, "pushes", 0) + 1
        O.record("push", run=self.dir.name, start=start, end=end, height=dz, yaw=yaw, inline=inline, note=note)
        for k in exempt:  # whatever was pushed has moved: a re-look before it is picked
            if k in self.belief:
                self.belief[k]["t_seen"] = 0

    def relook_region(self, pts, tag="look-region"):
        """One look from the LOOKS pose that sees the most of these points (blocks' footprints), the standard look
        first when it sees them all; the belief's table blocks that look sees are replaced by what it finds (ids kept
        by colour and nearest position, 10 cm at most), the rest kept."""
        cands = [LOOK_TIP] + list(LOOKS)
        best = None
        from hillclimb.runner.calib.look_cover import in_view
        for pose in cands:
            q = self.look_q(pose)
            if q is None:
                continue
            n = sum(in_view(self.op.view, q, [{"x": p[0], "y": p[1]} for p in pts]))
            if best is None or n > best[0]:
                best = (n, pose)
        pose = best[1] if best else LOOK_TIP
        self.look_to(pose, tag)
        time.sleep(0.25)
        old = {k: dict(v) for k, v in self.belief.items()}
        old_obst = self.obstacles
        blocks, rejected, path = self.look_all(tag, move=False, vet=False)
        q = self.q_look
        keep = {k: b for k, b in old.items() if k.startswith("pile") or k in getattr(self, "tower_ids", ()) or bot(b) > TABLE_Z + 0.005
                or not in_view(self.op.view, q, [b])[0]}
        gone = {k: b for k, b in old.items() if k not in keep}
        new = dict(keep)
        for k, b in sorted(blocks.items(), key=lambda kv: -kv[1].get("iou", 0)):
            c = colour_of(k)
            if b.get("edge") or not in_view(self.op.view, q, [b])[0]:
                continue  # cut by the frame's edge: what the earlier looks saw of it stands
            if any(colour_of(kk) == c and math.hypot(v["x"] - b["x"], v["y"] - b["y"]) < MERGE_R for kk, v in keep.items()):
                continue  # seen by an earlier look too, where it was in full view
            cands_ = [g for g in gone if colour_of(g) == c and g not in new and math.hypot(gone[g]["x"] - b["x"], gone[g]["y"] - b["y"]) < 0.10]
            bid = min(cands_, key=lambda g: math.hypot(gone[g]["x"] - b["x"], gone[g]["y"] - b["y"])) if cands_ else None
            if bid is None:
                n = 1
                while f"{c}#{n}" in new or f"{c}#{n}" in old:
                    n += 1
                bid = f"{c}#{n}"
            new[bid] = dict(b, t_seen=time.time())
        self.belief = new
        if len(old_obst):  # sticks and balls this look did not cover are kept
            P = np.asarray(pts, float)
            far = np.min(np.hypot(old_obst[:, 0, None] - P[None, :, 0], old_obst[:, 1, None] - P[None, :, 1]), 1) > 0.12
            self.obstacles = np.concatenate([old_obst[far], self.obstacles]) if len(self.obstacles) else old_obst[far]
        self.put_blocks()
        D.status(f"re-look ({pose[0] * 100:.0f}, {pose[1] * 100:.0f}) yaw {pose[3]:.0f}: " + ", ".join(
            f"{k} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) iou {b.get('iou', 0):.2f}" for k, b in new.items() if k not in keep))

    def piles(self):
        """Blocks seen sitting on others (level 1 fits) that are not a recorded stack: highest first."""
        tw = [(self.belief[k]["x"], self.belief[k]["y"]) for k in getattr(self, "tower_ids", ()) if k in self.belief]
        return sorted([k for k, b in self.belief.items() if not k.startswith("pile") and bot(b) > TABLE_Z + 0.005 and k not in getattr(self, "tower_ids", ())
                       and not any(math.hypot(b["x"] - x, b["y"] - y) < 0.10 for x, y in tw)], key=lambda k: -bot(self.belief[k]))

    def separate(self, max_pushes=8, max_untangle=3):
        """The first phase of tallest: piles taken apart top-down (each top block picked and set down alone at a free
        spot), then touching blocks and weak blobs pushed apart, one push at a time with a re-look after each, until
        every block can be picked on its own or no push plans."""
        import collections
        self.pushes, n_untangle = 0, 0
        self.push_why = collections.Counter()
        while True:
            D.Loop.stop_check(self)
            pl_ = self.piles()
            if pl_ and n_untangle < max_untangle:
                k = pl_[0]
                area = D.AREA
                D.AREA = ((0.22, 0.34), (-0.10, 0.15))
                try:
                    px, py, pyaw = self.pick_spot(k, [0])
                    D.status(f"separate: {k} sits on a pile ({bot(self.belief[k]) * 1000:.0f} mm up): taken off first, to ({px * 100:.1f}, {py * 100:.1f})")
                    self.block(k, (px, py, pyaw, 0, None), 1, n_untangle + 1, "untangle")
                except D.CycleFail as e:
                    D.status(f"separate: untangling {k} failed ({str(e)[:200]})")
                    self.op.st["target"] = None
                    st = arm_state()
                    if st["grip"] < 0.75 and st["goal_grip"] < 0.75:
                        raise D.Fatal(f"holding after an untangle failure: {e}")
                finally:
                    D.AREA = area
                n_untangle += 1
                self.untangled = n_untangle
                b = self.belief.get(k, {})
                self.relook_region([(b.get("x", 0.28), b.get("y", 0.0))] + [(v["x"], v["y"]) for v in self.belief.values()
                                                                             if math.hypot(v["x"] - b.get("x", 0.28), v["y"] - b.get("y", 0)) < 0.15])
                continue
            objs = self.objects()
            need = self.needs_push(objs)
            if not need:
                D.status(f"separate: every block can be picked on its own ({self.pushes} pushes)")
                return True
            if self.pushes >= max_pushes:
                D.status(f"separate: {self.pushes} pushes made, still to separate: {[o['id'] for o in need]}")
                return False
            cands = self.push_candidates(need, objs)
            done = None
            tried = collections.Counter()
            for score, o, start, end, inline, n_ok, n_cl in cands:
                # planning costs seconds of arm time: at most 6 plans, 2 per block
                if sum(tried.values()) >= 6 or tried[o["id"]] >= 2:
                    continue
                tried[o["id"]] += 1
                ex = {p["block"] for p in objs if math.hypot(p["x"] - o["x"], p["y"] - o["y"]) < 0.15}
                try:
                    self.push(start, end, ex, inline, f" {self.pushes + 1} ({o['id']}: {n_ok} of {n_cl} pickable after)")
                    done = (o, start, end)
                    break
                except SystemExit as e:
                    self.push_why["plan: " + str(e)[:60]] += 1
            if done is None:
                D.status(f"separate: no push plans for {[o['id'] for o in need]} ({dict(self.push_why)})")
                return False
            o, start, end = done
            pts = [tuple(end), (o["x"], o["y"])] + [(p["x"], p["y"]) for p in objs if math.hypot(p["x"] - o["x"], p["y"] - o["y"]) < 0.12]
            self.relook_region(pts)

    # ---- picks that keep the open fingers off touching neighbours
    def pad_hits(self, p, xaxis, exempt, finger=FINGER):
        """Neighbour blocks a pad would overlap at fingertips p with the jaws closing along xaxis."""
        hits = []
        yaxis = np.array([-xaxis[1], xaxis[0]])
        for c, b in self.belief.items():
            if c in exempt:
                continue
            if p[2] > top_of(b) + 0.015 or bot(b) > p[2] + 0.06:
                continue
            cb, sb = math.cos(b["yaw"]), math.sin(b["yaw"])
            hu, hv = D.half_of(b)
            for s in (-1, 1):
                ctr = p[:2] + s * finger * xaxis[:2]
                corners = [ctr + a * PAD_HALF[0] * xaxis[:2] + w * PAD_HALF[1] * yaxis for a in (-1, 1) for w in (-1, 1)] + [ctr]
                for pt in corners:
                    d = pt - [b["x"], b["y"]]
                    u, v = d[0] * cb + d[1] * sb, -d[0] * sb + d[1] * cb
                    if abs(u) < hu + 0.002 and abs(v) < hv + 0.002:
                        hits.append(c)
                        break
        # sticks and balls lie up to 43 mm high: no pad within 5 mm of one of their points while that low
        if len(self.obstacles) and p[2] < TABLE_Z + 0.06:
            for s in (-1, 1):
                ctr = p[:2] + s * finger * xaxis[:2]
                d = self.obstacles - ctr
                u, v = np.abs(d @ xaxis[:2]), np.abs(d @ yaxis)
                if np.any((u < PAD_HALF[0]) & (v < PAD_HALF[1])):  # foam sticks: 3 mm past the pad's outline
                    hits.append("an obstacle")
                    break
        return hits

    def finger_sweep(self, plan, exempt, finger=FINGER):
        for (q0, _, _), (q1, _, tag) in zip(plan.nodes[:-1], plan.nodes[1:]):
            n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / math.radians(2))) + 1)
            for t in np.linspace(0, 1, n):
                pos, R = self.op.kin.fk(q0 + (q1 - q0) * t)
                p = pos + self.op.kin.tip * R[:, 2]
                hits = self.pad_hits(p, R[:, 0], exempt, finger)
                if hits:
                    raise SystemExit(f"fingers refused at {tag}: a pad meets {hits}")

    def plan_pick(self, colour, q0, regrip=0.0):
        b = self.belief[colour]
        jy = math.degrees(b["yaw"])
        cxy = np.array([b["x"], b["y"]]) - PICK_DOWN[:2]
        z_grip = GRIP_TIP + (bot(b) - TABLE_Z) - PICK_DOWN[2] + (0.010 if bot(b) > TABLE_Z + 0.005 or b.get("lying") else 0.0) + regrip
        # the highest fingers-down reach falls from about 238 mm near x 16-24 cm to about 190 mm at x 36 cm
        cap = D.REACH_Z if b["x"] < 0.25 else 0.19
        z_over = min(top_of(b) + 0.08 + self.dr["over_dz"], cap)
        ex = self.below(colour)
        last = None
        # the widest opening (80, 70 or 63 mm) with a jaw axis whose open fingers clear the neighbours at the grip
        axes = []
        for g in (0.8, 0.7, 0.63):
            finger = g * 0.05 + 0.003
            # on its side the block is 34.5 mm along its yaw (the hole) and 50.3 across: the pads close across, on the
            # 50 mm span, or the 0.44 close would stop short of a 34.5 mm grip
            for par in ((1,) if b.get("lying") or b.get("big") else (0, 1)):
                a = math.radians(jy) + par * math.pi / 2
                if not self.pad_hits(np.array([b["x"], b["y"], z_grip]), np.array([math.cos(a), math.sin(a), 0.0]), ex, finger):
                    axes.append(par)
            if axes:
                break
        if not axes:
            # touching neighbours on every side (a pile of foam): the axis with the fewest, at 63 mm; a pad may brush
            # or press on a neighbour (Isaiah, Nov 7: foam, it is fine)
            g = 0.63
            finger = g * 0.05 + 0.003
            counts = []
            for par in ((1,) if b.get("lying") or b.get("big") else (0, 1)):
                a = math.radians(jy) + par * math.pi / 2
                counts.append((len(self.pad_hits(np.array([b["x"], b["y"], z_grip]), np.array([math.cos(a), math.sin(a), 0.0]), ex, finger)), par))
            axes = [min(counts)[1]]
            self.pad_tolerant = True
            D.status(f"pick {colour}: neighbours on both jaw axes; going in on axis {axes[0]} at 63 mm (a pad may touch a neighbour)")
        else:
            self.pad_tolerant = False
        self.pick_open = getattr(self, "pick_open", {})
        self.pick_open[colour] = g
        z_high = z_over
        for route in ("direct", "high", "low", "lower"):
            # low: over at the top + 3 cm (the place's own approach height); lower: + 1.5 cm, the last 3 cm then shorter
            z_over = min(top_of(b) + 0.03, cap) if route == "low" else min(top_of(b) + 0.015, cap) if route == "lower" else z_high
            pl = AxisPlan(self.op, q0, 0.08)
            pl.parity = axes
            try:
                if route == "high":
                    p0 = self.op.tip(q0)[0]
                    pl.to([p0[0], p0[1], min(max(self.dr["carry_z"], p0[2]), cap)], self.op.tip(q0)[1], 0.4, "up", line=0.02)
                    pl.R = None
                    pl.to([cxy[0], cxy[1], min(max(self.dr["carry_z"], z_over), cap)], jy, 0.5, "over high")
                    pl.to([cxy[0], cxy[1], z_over], jy, 0.4, "over", line=0.02, keep=True)
                else:
                    pl.to([cxy[0], cxy[1], z_over], jy, V_HIGH, "over")
                if z_over > z_grip + 0.03 + 1e-4:
                    pl.to([cxy[0], cxy[1], z_grip + 0.03], jy, V_DOWN, "down", line=0.01, keep=True)
                pl.to([cxy[0], cxy[1], z_grip], jy, V_LAST, "last 3 cm", line=0.005, keep=True)
                pl.check(skip=colour)
                if not getattr(self, "pad_tolerant", False):
                    self.finger_sweep(pl, ex, finger)
                self.sweep_high(pl, ex)
                return pl
            except SystemExit as e:
                last = e
        raise SystemExit(f"pick {colour}: {last}")

    def sweep_high(self, plan, exempt):
        """The data loop's sweep for the parts of a pick path above the neighbours' grip zone (fingertips more than
        1.5 cm over a block's top are never in the way; lower, the pad check above decides)."""
        for (q0, _, _), (q1, _, tag) in zip(plan.nodes[:-1], plan.nodes[1:]):
            n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / math.radians(2))) + 1)
            for t in np.linspace(0, 1, n):
                p = self.op.tip(q0 + (q1 - q0) * t)[0]
                for c, b in self.belief.items():
                    if c in exempt:
                        continue
                    top = top_of(b)
                    if math.hypot(p[0] - b["x"], p[1] - b["y"]) < 0.03 and p[2] < top + 0.012:
                        raise SystemExit(f"sweep refused at {tag}: fingertips over the {c} block")

    # ---- the tower
    def tower_spot(self, order, levels=5):
        """A spot where the most levels plan (all of them if it can). Returns (x, y, yaw, order that plans)."""
        import collections
        best = None
        self.why = collections.Counter()
        tried = 0
        t_end = time.time() + float(os.environ.get("TOWER_SPOT_S", "10"))  # the arm waits while this runs (Nov 8: 3 s planned 2-4 of 5)
        area = D.AREA
        D.AREA = getattr(self, "spot_area", ((0.18, 0.30), (-0.10, 0.15)))  # the reach is highest close in: the tower goes there
        for x, y, yaw in self.spot(None, list(range(levels)), avoid=[], gap=0.075):
            tried += 1
            got = self.plan_tower((x, y, yaw), order)
            if best is None or len(got) > len(best[3]):
                best = (x, y, yaw, got, getattr(self, "last_refusal", None))
            if len(got) == levels or tried >= 120 or (time.time() > t_end and best and best[3]):
                break
        D.AREA = area
        if best:
            self.last_refusal = best[4]
            return best[:4]
        return None

    def plan_level(self, bid, lvl, spot, sup, q):
        x, y, yaw = spot
        st = top_of(self.belief[sup]) if sup else TABLE_Z
        self.dr = {"carry_z": min(max(0.15, st + 0.05), max(D.REACH_Z, st + 0.035)), "over_dz": 0.01}
        self.op.st["target"] = q.tolist()
        pl = self.plan_pick(bid, q)
        lift = self.plan_lift(pl.nodes[-1][0], pl.R, math.degrees(self.belief[bid]["yaw"]), bid)
        grasp_h = 0.025 if self.belief[bid].get("lying") else 0.015
        p3 = self.plan_place(bid, lift.nodes[-1][0], lift.R, (x, y, yaw, st, sup), grasp_h)
        for zt, tag in ((st + hgt(self.belief[bid]), "check"),) + (((st, "support look"),) if sup else ()):
            ck = Plan(self.op, p3.nodes[-1][0], 0.08)
            ck.to([x - 0.065, y, min(max(0.13, zt + 0.06), D.LOOK_MAX_Z)], math.degrees(yaw), 0.4, tag)
            ck.check(skip=bid)

    def plan_tower(self, spot, order, base=()):
        """Greedy: at each level the first block (in this order) whose pick, place, close look and check photo all plan,
        with the blocks already placed moved onto the tower. base: the tower already standing (bottom first).
        Returns the order that planned (after base)."""
        saved = {k: dict(v) for k, v in self.belief.items()}
        q = self.q_look if self.q_look is not None else self.start_q()
        got, why = [], {}
        stack = list(base)
        try:
            for lvl in range(len(stack), len(stack) + len(order)):
                ok = None
                top = self.belief[stack[-1]] if stack else None
                at = (top["x"], top["y"], top["yaw"]) if top else spot
                for bid in [b for b in order if b not in got]:
                    try:
                        self.plan_level(bid, lvl, at, stack[-1] if stack else None, q)
                        ok = bid
                        break
                    except SystemExit as e:
                        why[bid] = str(e)[:160]
                if ok is None:
                    self.last_refusal = f"level {lvl} (block {lvl + 1}): " + "; ".join(f"{b}: {w}" for b, w in why.items() if b not in got)
                    break
                st = top_of(self.belief[stack[-1]]) if stack else TABLE_Z
                self.belief[ok] = {"x": at[0], "y": at[1], "yaw": at[2], "level": lvl, "iou": 0.95, "edge": False, "z0": st,
                                   "lying": self.belief[ok].get("lying", False), "big": self.belief[ok].get("big", False)}
                self.put_blocks()
                got.append(ok)
                stack.append(ok)
                why = {}
                if self.belief[ok].get("lying"):
                    break  # a block on its side is the top: nothing goes on its 34.5 mm face
        finally:
            self.belief = saved
            self.put_blocks()
            self.op.st["target"] = None
        return got

    # ---- the whole sequence, planned before anything moves
    def free_top(self, bid):
        """Nothing sits on this block (a pile's lower blocks come free when the ones above have moved)."""
        b = self.belief[bid]
        return not any(k != bid and bot(o) > bot(b) + 0.005 and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.04
                       for k, o in self.belief.items())

    def park_move(self, bid, spot, q, avoid_r=TOWER_CLEAR):
        """A table spot for a block that is in the way: free (data_loop's spot rules), TOWER_CLEAR from the tower spot,
        and its whole move planned from q (pick, lift, place, check). (x, y, yaw) or None."""
        import collections
        area = D.AREA
        D.AREA = ((0.17, 0.42), (-0.13, 0.25))  # the whole reachable table: a crowded one has room only at its edges
        self.why = getattr(self, "why", None) or collections.Counter()
        try:
            n = 0
            for x, y, yaw in self.spot(bid, [0], avoid=[(spot[0], spot[1])], gap=0.065):
                if math.hypot(x - spot[0], y - spot[1]) < avoid_r:
                    continue
                n += 1
                if n > 12:
                    return None
                try:
                    self.op.st["target"] = q.tolist()
                    self.plan_level(bid, 0, (x, y, yaw), None, q)
                    return x, y, yaw
                except SystemExit:
                    continue
            return None
        finally:
            D.AREA = area

    def plan_sequence(self, spot, order, base=(), deadline=None, max_extra=6):
        """Every move of the tower dry-planned in turn against the belief as it will be (each block moved as the plan
        moves it): at each level the first block in `order` whose pick, lift, place and check look all plan (data_loop's
        sweeps: the final straight-down segment of a place keeps the held block 12 mm over every block within 7.5 cm);
        when none does, the blocks the refusals name are cleared first (parked TOWER_CLEAR from the spot; a pile's top
        first), or a block no jaw axis can pick is pushed (push_candidates and plan_push, its predicted landing put in
        the belief). Returns (tower blocks after base, the moves [(kind, id, target)], why it stopped)."""
        saved = {k: dict(v) for k, v in self.belief.items()}
        q = self.q_look if self.q_look is not None else self.start_q()
        stack, seq, extra, stop = list(base), [], 0, None
        try:
            while True:
                if deadline and time.time() > deadline:
                    stop = "out of planning time"
                    break
                lvl = len(stack)
                top = self.belief[stack[-1]] if stack else None
                if top is not None and top.get("lying"):
                    stop = "a block on its side is the top; flat ones left: " + "; ".join(f"{b}: {w[:100]}" for b, w in getattr(self, "_why_flat", {}).items())
                    break
                at = (top["x"], top["y"], top["yaw"]) if top else spot
                sup = stack[-1] if stack else None
                why, ok = {}, None
                # nearest the tower first: a block left beside a growing tower can no longer be picked (its pads meet the
                # tower) and the tower's places cannot clear it
                free = sorted([b for b in order if b not in stack and b in self.belief and self.free_top(b)
                               and not (stack and colour_of(b) == colour_of(stack[-1]))],
                              key=lambda b: (0 if (math.hypot(self.belief[b]["x"] - spot[0], self.belief[b]["y"] - spot[1]) < 0.11
                                                   or abs(self.belief[b]["y"]) > 0.12 or math.hypot(self.belief[b]["x"], self.belief[b]["y"]) > 0.28) else 1,
                                             math.hypot(self.belief[b]["x"] - spot[0], self.belief[b]["y"] - spot[1])))
                # a block on its side ends the tower: only when no flat block plans, even after clearing moves
                for bid in [b for b in free if not self.belief[b].get("lying")]:
                    try:
                        self.plan_level(bid, lvl, at, sup, q)
                        ok = bid
                        break
                    except SystemExit as e:
                        why[bid] = str(e)
                if ok is None and any(not self.belief[b].get("lying") for b in order if b not in stack and b in self.belief) and extra < max_extra:
                    fix = self.clearing_move(why, at, spot, lvl, stack, [b for b in order if b not in stack and b in self.belief], q)
                    if fix is not None:
                        seq.append(fix)
                        extra += 1
                        continue
                if ok is None:
                    self._why_flat = dict(why)
                    for bid in [b for b in free if self.belief[b].get("lying")]:
                        try:
                            self.plan_level(bid, lvl, at, sup, q)
                            ok = bid
                            break
                        except SystemExit as e:
                            why[bid] = str(e)
                if ok is not None:
                    st_ = top_of(self.belief[sup]) if sup else TABLE_Z
                    self.belief[ok] = {"x": at[0], "y": at[1], "yaw": at[2], "level": lvl, "iou": 0.95, "edge": False, "z0": st_,
                                       "lying": self.belief[ok].get("lying", False), "big": self.belief[ok].get("big", False)}
                    self.put_blocks()
                    seq.append(("stack", ok, (at[0], at[1], at[2], lvl, sup)))
                    stack.append(ok)
                    if len(stack) >= len([b for b in order if b in self.belief]) + len(base):
                        stop = "every block"
                        break
                    continue
                left = [b for b in order if b not in stack and b in self.belief]
                if not left:
                    stop = "every block"
                    break
                if extra >= max_extra:
                    stop = f"level {lvl}: {max_extra} clearing moves made; " + "; ".join(f"{b}: {w[:100]}" for b, w in why.items())
                    break
                fix = self.clearing_move(why, at, spot, lvl, stack, left, q)
                if fix is None:
                    stop = f"level {lvl} (block {lvl + 1}): " + "; ".join(f"{b}: {w[:120]}" for b, w in why.items())
                    break
                seq.append(fix)
                extra += 1
        finally:
            self.belief = saved
            self.put_blocks()
            self.op.st["target"] = None
        self.last_refusal = stop
        return stack[len(base):], seq, stop

    def clearing_move(self, why, at, spot, lvl, stack, left, q):
        """One move that unblocks the next level, applied to the belief: a block the refusals name (in the way of the
        place, its camera or a pick's pads; a pile's top before the block under it) parked away from the tower; else
        the block the most refusals were about pushed apart from its neighbours. None when nothing plans."""
        named = collections.Counter()
        clearance = ("sweep refused", "camera within", "pad meets", "fingertips over")
        for w in why.values():
            if any(c in w for c in clearance):  # reach refusals name no block worth moving
                for k in re.findall(r"([a-z]+#\d+)", w):
                    if k in self.belief and k not in stack:
                        named[k] += 1
        for k in [b for b in left if b not in named and not self.free_top(b)]:
            named[k] += 0  # under a pile: its top is in the way
        tried = set()
        for k, _ in named.most_common():
            # a named block under others: the pile's top first
            while not self.free_top(k):
                b = self.belief[k]
                k = max([o for o, ob in self.belief.items() if bot(ob) > bot(b) + 0.005 and math.hypot(ob["x"] - b["x"], ob["y"] - b["y"]) < 0.04],
                        key=lambda o: bot(self.belief[o]))
            if k in tried or k in stack:
                continue
            tried.add(k)
            dest = self.park_move(k, spot, q)
            if dest is not None:
                x, y, yaw = dest
                self.belief[k] = {"x": x, "y": y, "yaw": yaw, "level": 0, "iou": 0.95, "edge": False, "z0": TABLE_Z,
                                  "lying": self.belief[k].get("lying", False), "big": self.belief[k].get("big", False)}
                self.put_blocks()
                return ("clear", k, (x, y, yaw, 0, None))
        # pushes: a block whose pick refusals are about its neighbours
        objs = self.objects()
        need = [o for o in self.needs_push(objs) if o["block"] in left and any(c in why.get(o["block"], "") for c in clearance)]
        if need:
            self.push_why = getattr(self, "push_why", None) or collections.Counter()
            for score, o, s, e, inline, n_ok, n_cl in self.push_candidates(need, objs)[:3]:
                ex = {p["block"] for p in objs if math.hypot(p["x"] - o["x"], p["y"] - o["y"]) < 0.15}
                try:
                    self.op.st["target"] = q.tolist()
                    self.plan_push(s, e, ex, inline)
                except SystemExit:
                    continue
                d = (np.asarray(e) - s) / np.linalg.norm(np.asarray(e) - s)
                b = self.belief[o["block"]]
                if not o["part"]:
                    b["x"], b["y"] = float(b["x"] + d[0] * (PUSH_LEN - PUSH_GAP)), float(b["y"] + d[1] * (PUSH_LEN - PUSH_GAP))
                self.put_blocks()
                return ("push", o["block"], (tuple(s), tuple(e), inline))
        return None

    def best_sequence(self, order, base=(), budget=40.0):
        """The tower spot (a free one in the high-reach band, or a block where it lies as the base, or the standing
        tower) whose whole sequence plans the most blocks, then the fewest clearing moves. Spots are tried nearest the
        high-reach point first until the budget (seconds of planning) runs out."""
        import collections
        t_end = time.time() + budget
        cands = []
        if base:
            b0 = self.belief[base[0]]
            cands.append(((b0["x"], b0["y"], b0["yaw"]), list(base), list(order)))
        else:
            for k in order:  # a flat block on the table in the band, built on where it lies
                b = self.belief[k]
                if not b.get("lying") and abs(bot(b) - TABLE_Z) < 0.005 and 0.19 <= b["x"] <= 0.30 and -0.12 <= b["y"] <= 0.15 and self.free_top(k):
                    cands.append(((b["x"], b["y"], b["yaw"]), [k], [o for o in order if o != k]))
            area = D.AREA
            D.AREA = getattr(self, "spot_area", ((0.19, 0.29), (-0.12, 0.15)))
            self.why = collections.Counter()
            free = []
            for x, y, yaw in self.spot(None, [0, 1, 2], avoid=[], gap=0.0):  # spots with blocks near them: cleared in the plan
                if not any(math.hypot(x - b["x"], y - b["y"]) < 0.055 for b in self.belief.values()):
                    free.append(((x, y, yaw), [], list(order)))
                if len(free) >= 12:
                    break
            D.AREA = area
            cands += free
            cands.sort(key=lambda c: math.hypot(c[0][0] - 0.24, c[0][1] - 0.03))
        best = None
        n_tried = 0
        for spot, b_, o_ in cands:
            if time.time() > t_end and best is not None:
                break
            got, seq, stop = self.plan_sequence(spot, o_, b_, deadline=t_end + 20)
            n_tried += 1
            n = len(b_) + len(got)
            score = (n, -sum(1 for m in seq if m[0] != "stack"))
            if best is None or score > best[0]:
                best = (score, spot, b_, seq, stop)
            if n >= len(order) + len(base):
                break
        if best is None:
            return None
        (n, neg), spot, b_, seq, stop = best
        D.status(f"whole sequence: {n} blocks plan at ({spot[0] * 100:.1f}, {spot[1] * 100:.1f}) on base {b_ or 'a free spot'} "
                 f"with {-neg} clearing moves ({n_tried} spots planned); moves {[(m[0], m[1]) for m in seq]}; stop: {stop}")
        return spot, b_, seq, stop

    def run_sequence(self, spot, base, seq):
        """The planned moves in order; a move that fails or no longer plans: the rest re-planned from the belief as it
        is (once per level), then on."""
        tower = list(base)
        k = 0
        replans = 0
        moves = list(seq)
        while moves:
            kind, bid, tgt = moves.pop(0)
            k += 1
            try:
                if kind == "push":
                    s, e, inline = tgt
                    objs = self.objects()
                    ex = {p["block"] for p in objs if math.hypot(p["x"] - s[0], p["y"] - s[1]) < 0.15}
                    self.push(np.asarray(s), np.asarray(e), ex, inline, f" (planned, {bid})")
                    self.relook_region([tuple(e), tuple(s)] + [(p["x"], p["y"]) for p in objs if math.hypot(p["x"] - s[0], p["y"] - s[1]) < 0.12])
                elif kind == "clear":
                    self.block(bid, tgt, 1, k, "clear")
                else:
                    if tower:
                        s_ = self.belief[tower[-1]]
                        tgt = (s_["x"], s_["y"], s_["yaw"], len(tower), tower[-1])
                    self.block(bid, tgt, 1, len(tower) + 1, "stack")
                    tower.append(bid)
                    D.status(f"tall tower: {len(tower)} high ({' / '.join(tower)})")
                continue
            except (D.NoPlan, D.NoSpot, D.CycleFail, SystemExit) as e:
                st = arm_state()
                if st["grip"] < 0.75 and st["goal_grip"] < 0.75:
                    raise
                self.op.st["target"] = None
                D.status(f"sequence: {kind} {bid} failed ({str(e)[:200]})")
            if replans >= 4:
                break
            replans += 1
            left = [b for b in self.belief if b not in tower and not b.startswith("pile")]
            res = self.best_sequence(left, tuple(tower), budget=15.0) if tower else self.best_sequence(left, (), budget=15.0)
            if res is None:
                break
            spot, b_, moves, stop = res
            moves = list(moves)
            tower = list(b_) if not tower else tower
        return tower

    def refresh_free(self, tower):
        """A full look from the look pose: the blocks not in the tower refitted and keyed by their nearest id."""
        old = {k: dict(v) for k, v in self.belief.items()}
        blocks, rejected, path = self.look_all("look-refresh")
        new = {k: v for k, v in old.items() if k in tower}
        tx, ty = old[tower[0]]["x"], old[tower[0]]["y"]
        free = [b for b in blocks.values() if math.hypot(b["x"] - tx, b["y"] - ty) > 0.05]
        cols = {k: v for k, v in blocks.items() if v in free}
        prev = {k: v for k, v in old.items() if k not in tower}
        for k, b in sorted(cols.items(), key=lambda kv: kv[0]):
            c = colour_of(k)
            cands = [p for p in prev if colour_of(p) == c and p not in new]
            bid = min(cands, key=lambda p: math.hypot(prev[p]["x"] - b["x"], prev[p]["y"] - b["y"])) if cands else None
            if bid is None:
                n = 1
                while f"{c}#{n}" in new or f"{c}#{n}" in tower:
                    n += 1
                bid = f"{c}#{n}"
            new[bid] = b
        self.belief = new
        self.put_blocks()
        D.status("tall tower full look: " + ", ".join(f"{k} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) L{b['level']} iou {b['iou']:.2f}"
                                                     for k, b in new.items()) + f"; overlay {path}")

    def move_tower(self, tower, pile, parked=()):
        """Unstack the tower top-down to spread spots (18-30 cm out, 8 cm apart), rebuild it in the same order at the
        spot near x 18-22 cm where the most levels (tower then pile) plan, then add the pile blocks top first."""
        import collections
        area = D.AREA
        try:
            # where it goes: the most levels planned with every block at its parking spot unknown yet: plan from
            # the blocks where they stand (the tower's own blocks moved off it first in the plan)
            D.AREA = ((0.16, 0.26), (-0.14, 0.14))
            self.why = collections.Counter()
            best = None
            cands = []
            for x, y, yaw in self.spot(None, [0, 1, 2], avoid=[(self.belief[tower[0]]["x"], self.belief[tower[0]]["y"])], gap=0.06):
                cands.append((x, y, yaw))
                if len(cands) >= 40:
                    break
            # the reach is highest near x 18-22, y -5..10: nearest that first; 9 cm from a tall stack (the camera
            # passes over it), and the first block's whole move must plan
            first = (list(tower) + list(parked))[0]
            for x, y, yaw in sorted(cands, key=lambda c: math.hypot(c[0] - 0.20, c[1] - 0.025)):
                if any(top_of(b) > TABLE_Z + 0.05 and math.hypot(b["x"] - x, b["y"] - y) < 0.09 for k, b in self.belief.items() if k != first):
                    continue
                if self.feasible(first, (x, y, yaw, 0, None)):
                    best = (x, y, yaw, [])
                    break
            if best is None:
                raise D.Fatal(f"no new tower spot near x 18-22 cm: {dict(self.why)}")
            x, y, yaw, got = best
            D.status(f"tall tower: new spot ({x * 100:.1f}, {y * 100:.1f}) yaw {math.degrees(yaw):.0f}: {len(got)} levels plan ({got}); limit {getattr(self, 'last_refusal', None)}")
            # park the tower blocks top-down, 8 cm apart, 18-30 cm out, clear of the new spot
            D.AREA = ((0.18, 0.30), (-0.10, 0.20))
            park = []
            for k in reversed(tower):
                wait_no_hands(self)
                try:
                    px, py, pyaw = self.pick_spot(k, [0], avoid=[(x, y)] + park)
                except D.NoSpot as e:
                    D.status(f"tall tower: no parking spot for {k} ({str(e)[:200]}); re-looking it and the blocks around it")
                    self.belief[k]["t_seen"] = 0
                    super(Tall, self).relook(k)
                    px, py, pyaw = self.pick_spot(k, [0], avoid=[(x, y)] + park)
                self.block(k, (px, py, pyaw, 0, None), 1, 0, "unstack")
                park.append((px, py))
            # rebuild in the same order at the new spot
            built = []
            for k in list(tower) + list(parked):
                wait_no_hands(self)
                if built:
                    s_ = self.belief[built[-1]]
                    tgt = (s_["x"], s_["y"], s_["yaw"], len(built), built[-1])
                else:
                    tgt = (x, y, yaw, 0, None)
                self.block(k, tgt, 1, len(built) + 1, "stack")
                built.append(k)
                D.status(f"tall tower: {len(built)} high at the new spot ({' / '.join(built)}), top at {(top_of(self.belief[k]) - TABLE_Z) * 1000:.0f} mm")
        finally:
            D.AREA = area
        built = self.build_seq(pile, built)
        D.status(f"tall tower done: {len(built)} blocks ({' / '.join(built)}), top at {(top_of(self.belief[built[-1]]) - TABLE_Z) * 1000:.0f} mm over the table")
        return built

    def loop_look(self, avoid):
        """A full look; blocks by the base (x under 15 cm) and the avoided piles stay in the belief as things in the way."""
        self.look_all("look-loop")
        if not getattr(self, "seeded", False):
            self.seeded = True
            self.seed_levels()
        for n_, (ax, ay, atop) in enumerate(avoid):
            self.belief = {k: b for k, b in self.belief.items() if math.hypot(b["x"] - ax, b["y"] - ay) > 0.07}
            self.belief[f"pile#{n_}"] = {"x": ax, "y": ay, "yaw": 0.0, "level": 0, "z0": atop - H, "lying": False, "iou": 1.0, "edge": False,
                                         "t_seen": time.time()}
            if len(self.obstacles):
                self.obstacles = self.obstacles[np.hypot(self.obstacles[:, 0] - ax, self.obstacles[:, 1] - ay) > 0.06]
        self.put_blocks()
        # nothing on top: a block under another one (a stack left by a stopped run) is neither picked nor built on
        covered = {k for k, b in self.belief.items() for o in self.belief.values()
                   if o is not b and bot(o) > bot(b) + 0.01 and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.045 + D.ext(o) + D.ext(b)}
        # and within reach (purple at x 51 cm cannot be picked: choosing it aborted a cycle after its base was set)
        free = [k for k, b in self.belief.items() if not k.startswith("pile") and b["x"] >= 0.15 and abs(bot(b) - TABLE_Z) < 0.005
                and k not in covered and math.hypot(b["x"], b["y"]) <= PUSH_REACH]
        D.status("stack loop look: " + ", ".join(f"{k} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}){' side' if b.get('lying') else ''}"
                                                for k, b in self.belief.items()) + f"; free {free}")
        return free

    # ---- domain randomization (Isaiah, Nov 8 03:05: neater, tighter, aligned, rotated, at different spots and speeds)
    STYLES = {"tight": (4.0, 4.0), "aligned": (8.0, 10.0), "rotated": (8.0, 10.0)}

    def pick_style(self):
        """This cycle's stacking style: tight (square on the block below, re-placed over 4 mm or 4 deg), aligned (the
        usual 8 mm, 10 deg) or rotated (each block turned 10 to 35 deg, either way, against the one below)."""
        st = str(self.rng.choice(list(self.STYLES)))
        D.REPLACE_MM, D.REPLACE_DEG = self.STYLES[st]
        self.style = st
        self.cyc["style"] = st
        return st

    def style_yaw(self):
        if getattr(self, "style", "aligned") != "rotated":
            return 0.0
        return float(self.rng.choice([-1, 1]) * math.radians(self.rng.uniform(10, 35)))

    def speed_scale(self):
        """Each move at a random fraction of the streamed speeds (0.75 to 1.15), logged with the episode."""
        k = float(self.rng.uniform(0.75, 1.15))
        base = SPEEDS["A"]
        D.V_HIGH, D.V_DOWN, D.V_LAST, D.V_SET = (v * k for v in base)
        g = globals()
        g["V_HIGH"], g["V_DOWN"], g["V_LAST"] = (v * k for v in base[:3])
        self.dr["speed_scale"] = round(k, 3)
        return k

    def recover(self, max_moves=12):
        """From a tumbled table (a tower knocked over) to every block apart, flat or on its side, on the table: two looks;
        a block over another first (highest first), then a block touching or overlapping a neighbour (most neighbours
        first), each to an open spot; again until nothing is piled or touching. A block whose move does not plan is tried
        later; a push separates a pair no pick can take. Recorded as recover episodes (Isaiah, Nov 8 03:04: recovering
        from a knock-over is the data the judged run needs)."""
        moved = 0
        skip = collections.Counter()
        weak_done = collections.Counter()
        for _ in range(max_moves * 2):
            self.multi_look(tag="look-recover")
            bs = {k: b for k, b in self.belief.items() if not k.startswith("pile")}
            raised = [k for k, b in bs.items() if bot(b) > TABLE_Z + 0.005]
            touching = collections.Counter()
            for (ka, a), (kb, b) in itertools.combinations(bs.items(), 2):
                if box_gap(box_of(a), box_of(b)) < 0.012:
                    touching[ka] += 1
                    touching[kb] += 1
            # a poorly fitted block (fit under 0.65: leaning on something, tipped part way) is picked and set down again
            weak = [k for k, b in bs.items() if b.get("iou", 1.0) < 0.65 and k not in raised and k not in touching and not weak_done[k]]
            cands = sorted(raised, key=lambda k: -bot(bs[k])) + sorted(touching, key=lambda k: -touching[k]) + weak
            cands = [k for i, k in enumerate(cands) if k not in cands[:i] and skip[k] < 2]
            if not cands:
                D.status(f"recover: done after {moved} moves (" + ("nothing piled, touching or unclear" if not (raised or touching) else
                                                                 f"left: {sorted(set(raised) | set(touching))}") + ")")
                return moved
            done = False
            for k in cands:
                try:
                    px, py, pyaw = self.pick_spot(k, [0])
                    self.cyc = {"tower_order": [], "fallbacks": [], "mode": "recover", "raised": raised, "touching": dict(touching)}
                    weak_done[k] += 1
                    self.block(k, (px, py, pyaw, 0, None), 0, moved + 1, "recover")
                    moved += 1
                    done = True
                    break
                except (D.CycleFail, D.NoPlan, D.NoSpot) as e:
                    skip[k] += 1
                    D.status(f"recover: {k} not moved ({str(e)[:160]}); the next one")
                    self.op.st["target"] = None
                except D.Fatal as e:
                    # a neighbour moved with it (a pile): expected while recovering; the episode is a failure already.
                    # Look again and go on, unless the hand is not open
                    st = arm_state()
                    if "was not moved changed" not in str(e) or st["grip"] < 0.75:
                        raise
                    D.status(f"recover: {k} moved a neighbour too ({str(e)[:120]}); looking again")
                    self.op.st["target"] = None
                    moved += 1
                    done = True
                    break
            if not done:
                try:
                    self.separate(max_pushes=2, max_untangle=1)
                except Exception as e:  # noqa: BLE001
                    D.status(f"recover: push failed ({str(e)[:160]})")
                    return moved
        return moved

    def seed_levels(self):
        """A stack left by the run before (a stop) reads as blocks on the table when the block under the top one is hidden
        from the look (Nov 8 01:52: yellow on orange fitted 0.94 at table level, 0.83 one up, and its pick took the orange
        too). A block seen at table level takes the level the last run believed for a block of its colour within 3 cm."""
        prev = sorted((d for d in TALL.glob("1108-*") if d != self.dir and (d / "op" / "op.json").exists()),
                      key=lambda d: d.stat().st_mtime)
        if not prev:
            return
        last = json.loads((prev[-1] / "op" / "op.json").read_text()).get("blocks") or {}
        for k, b in self.belief.items():
            if bot(b) > TABLE_Z + 0.005:
                continue
            for pk, pb in last.items():
                if colour_of(pk) == colour_of(k) and pb.get("z0", TABLE_Z) > TABLE_Z + 0.005 and \
                        math.hypot(pb["x"] - b["x"], pb["y"] - b["y"]) < 0.03:
                    lvl = int(round((pb["z0"] - TABLE_Z) / H))
                    if lvl < 1:  # a block on its side (its z0 a little over the table): nothing to take
                        continue
                    b.update(level=lvl, z0=TABLE_Z + lvl * H)
                    D.status(f"{k}: the last run ({prev[-1].name}) had it {lvl} up at ({pb['x'] * 100:.1f}, {pb['y'] * 100:.1f}); taken as {lvl} up")
        self.put_blocks()

    def helper_cycle(self, n, free, avoid):
        """The second-stack maneuver (sim/stack9.py's helper, Isaiah Nov 8 02:15): a 2-high helper stack at one spot and
        a tower base at another; each other block goes onto the helper's top first, then from there onto the tower's
        top; then everything comes down to free spots. Fingers down throughout (the sim's 30 deg tilted regrasp is only
        needed past the fingers-down reach, about 7 high, and has not run on this arm)."""
        av = [(a[0], a[1]) for a in avoid]
        for _ in range(40):  # helper pair, tower base, the rest: never one colour on the same colour
            o = [str(c) for c in self.rng.permutation(free)]
            h0, h1, t0, rest = o[0], o[1], o[2], o[3:]
            chain = [t0] + rest
            if colour_of(h0) != colour_of(h1) and all(colour_of(r) != colour_of(h1) for r in rest) and \
                    all(colour_of(a) != colour_of(b) for a, b in zip(chain, chain[1:])):
                break
        else:
            raise D.NoSpot(f"no helper order without one colour on the same colour ({free})")
        self.cyc = {"tower_order": chain, "helper": [h0, h1], "fallbacks": [], "cycle": n, "mode": "helper"}
        D.status(f"helper cycle {n}: helper {h0} / {h1}, tower {chain}")
        k = 0
        hx, hy, hyaw = self.pick_spot(h0, [0, 1, 2], avoid=av)
        k += 1
        self.block(h0, (hx, hy, hyaw, 0, None), n, k, "helper")
        hb = self.belief[h0]
        k += 1
        self.block(h1, (hb["x"], hb["y"], hb["yaw"], 1, h0), n, k, "helper")
        tx, ty, tyaw = self.pick_spot(t0, [0, 1, 2], avoid=av + [(hb["x"], hb["y"])])
        k += 1
        self.block(t0, (tx, ty, tyaw, 0, None), n, k, "helper-tower")
        tower = [t0]
        for b in rest:
            top = self.belief[h1]
            k += 1
            self.block(b, (top["x"], top["y"], top["yaw"], 2, h1), n, k, "to-helper")
            tt = self.belief[tower[-1]]
            k += 1
            self.block(b, (tt["x"], tt["y"], tt["yaw"], len(tower), tower[-1]), n, k, "from-helper")
            tower.append(b)
        D.status(f"helper cycle {n}: tower {' / '.join(tower)} built through the helper {h0} / {h1}")
        for b in list(reversed(tower)) + [h1, h0]:  # down: the tower top first, then the helper
            if bot(self.belief[b]) <= TABLE_Z + 0.005 and b in (t0, h0):
                continue  # the bottoms stay where they are
            px, py, pyaw = self.pick_spot(b, [0], avoid=av)
            k += 1
            self.block(b, (px, py, pyaw, 0, None), n, k, "unstack")

    def stack_loop(self, avoid, cycles=1000):
        """Never exits on a block it cannot find or place: re-looks and goes on with another block or spot."""
        import collections
        t_report = time.time()
        fails = 0
        for n in range(1, cycles + 1):
            D.Loop.stop_check(self)
            try:
                free = self.loop_look(avoid)
                # a stack left standing (a stopped run): its top block first, down to a free table spot
                tops = [k for k, b in self.belief.items() if not k.startswith("pile") and bot(b) > TABLE_Z + 0.005
                        and not any(o is not b and bot(o) > bot(b) + 0.01 and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.045
                                    for o in self.belief.values())]
                # a top that failed twice is left standing (with whatever it sits on) rather than tried forever
                self.stuck = getattr(self, "stuck", collections.Counter())
                tops = [k for k in tops if self.stuck[k] < 2]
                if tops:
                    bid = max(tops, key=lambda k: bot(self.belief[k]))
                    self.stuck[bid] += 1
                    self.cyc = {"tower_order": [], "fallbacks": [], "cycle": n, "left_standing": bid}
                    px, py, pyaw = self.pick_spot(bid, [0], avoid=[(a[0], a[1]) for a in avoid])
                    self.block(bid, (px, py, pyaw, 0, None), n, 1, "unstack")
                    continue
                if len(free) < 2:
                    D.status("stack loop: fewer than 2 free blocks; looking again in 20 s")
                    time.sleep(20)
                    continue
                # STACK_MODE=alternate: every other cycle the helper maneuver (the sim's 9-high trick, fingers down here)
                if os.environ.get("STACK_MODE") == "alternate" and n % 2 == 0 and len(free) >= 4:
                    self.helper_cycle(n, free, avoid)
                    continue
                # STACK_K=lo,hi: tower heights (default 2-3); Isaiah, Nov 8 02:15: every reachable block, one tower
                lo, hi = (int(v) for v in os.environ.get("STACK_K", "2,3").split(","))
                k = min(int(self.rng.integers(lo, hi + 1)), len(free)) if len(free) >= 3 else 2
                # never one colour on the same colour: the two orange blocks touching read as one blob and the standing
                # check cannot tell them apart (Nov 8 02:04)
                for _ in range(30):
                    order = [str(c) for c in self.rng.permutation(free)][:k]
                    if all(colour_of(a) != colour_of(b) for a, b in zip(order, order[1:])):
                        break
                else:
                    order = order[:1] + [o for o in order[1:2] if colour_of(o) != colour_of(order[0])]
                self.cyc = {"tower_order": order, "fallbacks": [], "cycle": n}
                self.pick_style()
                self.why = collections.Counter()
                base = order[0]
                if k >= 4:  # a tall one: where every level's place plans before anything moves (the reach is highest close in)
                    got = self.tower_spot(order, levels=k)
                    if not got or len(got[3]) < 2:
                        raise D.NoSpot(f"no tower spot for {order} ({getattr(self, 'last_refusal', '')})")
                    x, y, yaw, planned = got
                    if len(planned) < k:
                        D.status(f"stack loop: {len(planned)} of {k} levels plan ({planned}); last refusal {str(getattr(self, 'last_refusal', ''))[:160]}")
                    order = [str(c) for c in planned]
                    for j in range(1, len(order)):  # the planner may reorder: never one colour on the same colour
                        if colour_of(order[j]) == colour_of(order[j - 1]):
                            order = order[:j]
                            break
                    self.cyc["tower_order"] = order
                else:
                    x, y, yaw = self.pick_spot(base, list(range(k)), avoid=[(a[0], a[1]) for a in avoid])
                built = []
                for bid in order:
                    wait_no_hands(self)
                    if built:
                        s_ = self.belief[built[-1]]
                        tgt = (s_["x"], s_["y"], s_["yaw"] + self.style_yaw(), len(built), built[-1])
                    else:
                        tgt = (x, y, yaw, 0, None)
                    try:
                        self.block(bid, tgt, n, len(built) + 1, "stack")
                        built.append(bid)
                    except (D.NoPlan, D.NoSpot) as e:
                        D.status(f"stack loop: {bid} skipped ({str(e)[:200]})")
                        self.op.st["target"] = None
                D.status(f"stack loop cycle {n}: built {len(built)} high ({' / '.join(built)}) at ({x * 100:.1f}, {y * 100:.1f})")
                # unstack top-down to random spots
                for j, bid in enumerate(reversed(built)):
                    wait_no_hands(self)
                    try:
                        px, py, pyaw = self.pick_spot(bid, [0], avoid=[(a[0], a[1]) for a in avoid])
                        self.block(bid, (px, py, pyaw, 0, None), n, len(built) + j + 1, "unstack")
                    except (D.NoPlan, D.NoSpot) as e:
                        D.status(f"stack loop: unstacking {bid} skipped ({str(e)[:200]}); re-looking")
                        self.op.st["target"] = None
                        break
                fails = 0
            except D.Fatal as e:
                # a knock-over (the overhead check): the move is a failure episode already; recover the table and go on
                st = arm_state()
                if "was not moved changed" not in str(e) or st["grip"] < 0.75 or os.environ.get("RECOVER") == "0":
                    raise
                D.status(f"stack loop cycle {n}: knocked ({str(e)[:160]}); recovering the table")
                self.op.st["target"] = None
                self.recover()
                continue
            except D.CycleFail as e:
                fails += 1
                st = arm_state()
                if st["grip"] < 0.75 and st["goal_grip"] < 0.75:
                    raise D.Fatal(f"stopped holding a block: {e}")
                D.status(f"stack loop cycle {n}: {type(e).__name__}: {str(e)[:300]}; re-looking")
                self.op.st["target"] = None
                if fails >= 6:
                    time.sleep(30)
            if time.time() - t_report > 600:
                t_report = time.time()
                D.status(f"stack loop counts: {self.counts}, blocks {len(self.block_times)}, "
                         f"mean {np.mean(self.block_times) if self.block_times else 0:.1f} s per block")

    def build_seq(self, order, base):
        """Strictly in this order (a pile comes off top first), each onto the tower's top; stops at the first failure."""
        built = list(base)
        self.cyc = {"tower_order": list(base) + list(order), "fallbacks": []}
        for bid in order:
            wait_no_hands(self)
            s_ = self.belief[built[-1]]
            try:
                self.block(bid, (s_["x"], s_["y"], s_["yaw"], len(built), built[-1]), 1, len(built) + 1, "stack")
            except (D.NoPlan, D.NoSpot) as e:
                D.status(f"tall tower: {bid} cannot go on at level {len(built)}: {str(e)[:300]}")
                break
            built.append(bid)
            D.status(f"tall tower: {len(built)} high ({' / '.join(built)}), top at {(top_of(self.belief[bid]) - TABLE_Z) * 1000:.0f} mm")
        return built

    def build(self, order, spot, base=()):
        """Places in the planned order; a block whose plan or grasp fails now gives way to the next one (each block
        retried at most twice). When none is left that plans, one full look for blocks moved within reach."""
        x, y, yaw = spot
        self.cyc = {"tower_spot": [x, y, yaw], "tower_order": list(base) + list(order), "fallbacks": []}
        built, left = list(base), list(order)
        tries = {}
        looked = False
        while True:
            lvl = len(built)
            sup = built[-1] if built else None
            if sup:
                s_ = self.belief[sup]
                target = (s_["x"], s_["y"], s_["yaw"], lvl, sup)
            else:
                target = (x, y, yaw, 0, None)
            done = None
            for bid in list(left):
                if tries.get(bid, 0) >= 3:
                    continue
                try:
                    tries[bid] = tries.get(bid, 0) + 1
                    self.block(bid, target, 1, lvl + 1, "stack")
                    done = bid
                    break
                except D.NoPlan as e:
                    D.status(f"tall tower: {bid} does not plan for level {lvl} now ({str(e)[:200]}); trying the next block")
                except D.CycleFail as e:
                    st = arm_state()
                    if st["grip"] < 0.75 and st["goal_grip"] < 0.75:
                        raise
                    D.status(f"tall tower: {bid} failed ({e}); trying the next block, then {bid} again")
                    self.op.st["target"] = None
            if done is None:
                if looked:
                    D.status(f"tall tower: no remaining block plans for level {lvl}: stopping at {lvl} high")
                    break
                looked = True
                self.refresh_free(built)
                left = [k for k in self.belief if k not in built]
                tries = {}
                continue
            left.remove(done)
            built.append(done)
            looked = False
            D.status(f"tall tower: {len(built)} high ({' / '.join(built)})")
            if self.belief[done].get("lying"):
                D.status("tall tower: a block on its side is the top; done")
                break
            if not left:
                self.refresh_free(built)  # a block slid within reach gets added too
                left = [k for k in self.belief if k not in built]
                looked = True
                if not left:
                    break
        return built


def hands_px(bgr, view=None, q=None):
    """Skin-coloured pixels (hue under 25 or over 170, saturation 35-90: the orange foam is 99, the red sticks 131) that
    land on the table inside the work area (x 5-45 cm, y -15..30) when cast through the wrist camera: a hand over the
    table. A leg or arm past the table's edge projects outside the area and does not count. Without the camera pose,
    the frame's bottom fifth (the old test)."""
    small = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    skin = ((hsv[..., 0] < 25) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 35) & (hsv[..., 1] < 90) & (hsv[..., 2] > 70)
    if view is None or q is None:
        return int(skin[288:].sum()) * 4
    cam = view.pose(q, 0.08)
    skin &= view.jaws_mask(q, 0.08) == 0
    ys, xs = np.nonzero(skin)
    if not len(xs):
        return 0
    sel = np.stack([xs, ys], 1)[:: max(1, len(xs) // 3000)]
    P = view.ray_on_plane(cam, sel, TABLE_Z)
    Rc, pc = cam
    down = (P[:, 2] - pc[2]) < 0  # rays that meet the table in front of the camera
    inside = down & (P[:, 0] > 0.05) & (P[:, 0] < 0.45) & (P[:, 1] > -0.15) & (P[:, 1] < 0.30)
    return int(inside.sum() * len(xs) / len(sel))


def wait_no_hands(t, limit=2500, timeout=300):
    if os.environ.get("HANDS") != "1":
        return 0  # off (Isaiah, Nov 7 20:50): people at the station keep clear themselves; false waits cost arm time
    t0 = time.time()
    while True:
        data, jq, jg = O.fetch("wrist")
        q = None if jq is None else np.array(jq, float) + t.op.dq
        n = hands_px(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), t.op.view, q)
        if n < limit:
            return n
        if time.time() - t0 > timeout:
            raise D.Fatal(f"a hand stayed over the table for {timeout} s ({n} px)")
        D.status(f"hand over the table ({n} px at 640x360): waiting")
        time.sleep(2)


# V_HIGH, V_DOWN, V_LAST, V_SET: A the 20:40 speeds, B high-up segments at 0.75 (stream rate about 0.98, cap 1.0 since
# opstack's 20:50 change); lowering and set-down the same in both
SPEEDS = {"A": (1.2, 0.9, 0.6, 0.18), "B": (1.2, 0.9, 0.6, 0.18)}  # 2x (Isaiah, 20:57)


def speed_mode():
    """A or B, alternating by the session's minute: the streamed speeds for this session (logged in every label)."""
    m = "A" if int(time.strftime("%M")) % 2 == 0 else "B"
    D.V_HIGH, D.V_DOWN, D.V_LAST, D.V_SET = SPEEDS[m]
    g = globals()
    g["V_HIGH"], g["V_DOWN"], g["V_LAST"] = SPEEDS[m][:3]
    return m


STAGE_BASE = (0.22, -0.05)
STAGE_FEED = [(0.17, -0.137), (0.17, 0.037), (0.27, -0.137), (0.27, 0.037), (0.22, 0.123), (0.32, -0.05)]
STAGE_PARK = [(0.37, 0.07), (0.125, 0.10), (0.125, -0.12)]  # 10 cm or more from every grid spot, |y| under 14.5 cm


def stage(t):
    """Move the table's flat blocks onto the staging grid with the runner's own block move. A block within 1.5 cm of
    its spot stays; a spot held by another block is freed first (its block moved on, or parked). Returns the base."""
    flat = [k for k, b in t.belief.items() if abs(bot(b) - TABLE_Z) < 0.005 and not b.get("lying") and b.get("iou", 0) >= 0.5
            and not k.startswith("pile") and t.free_top(k)]
    spots = [STAGE_BASE] + STAGE_FEED
    # same-colour neighbours on the grid are fine (10 cm apart); assign by nearest, blocks already close first
    assign, free_b = {}, list(flat)
    pairs = sorted(((math.hypot(t.belief[k]["x"] - s[0], t.belief[k]["y"] - s[1]), k, i) for k in flat for i, s in enumerate(spots)))
    used = set()
    for d, k, i in pairs:
        if k in assign or i in used:
            continue
        assign[k], _ = i, used.add(i)
    # a block past seven: parked only when it is within 10 cm of a grid spot, else left where it is
    extra = [k for k in flat if k not in assign and any(math.hypot(t.belief[k]["x"] - s[0], t.belief[k]["y"] - s[1]) < 0.10 for s in spots)]
    for j, k in enumerate(extra[:len(STAGE_PARK)]):
        assign[k] = len(spots) + j
    allspots = spots + STAGE_PARK

    def at_spot(k):
        s = allspots[assign[k]]
        return math.hypot(t.belief[k]["x"] - s[0], t.belief[k]["y"] - s[1]) < 0.015

    def blocker(k):
        s = allspots[assign[k]]
        return [o for o in t.belief if o != k and math.hypot(t.belief[o]["x"] - s[0], t.belief[o]["y"] - s[1]) < 0.10]

    moved, n = 0, 0
    todo = [k for k in assign if not at_spot(k)]
    D.status(f"stage: {len(flat)} flat blocks, {len(todo)} to move; base spot {STAGE_BASE}")
    for _ in range(3 * len(todo) + 3):
        todo = [k for k in todo if not at_spot(k)]
        if not todo:
            break
        ready = [k for k in todo if not blocker(k)]
        if not ready:
            # every target is held: the block holding the first one goes to a free parking spot
            k0 = todo[0]
            hold = blocker(k0)[0]
            free_p = [j for j, p in enumerate(STAGE_PARK) if not any(math.hypot(b["x"] - p[0], b["y"] - p[1]) < 0.10 for b in t.belief.values())]
            if not free_p:
                D.status("stage: no free parking spot; staging stops")
                break
            assign[hold] = len(spots) + free_p[0]
            todo.append(hold)
            continue
        k = min(ready, key=lambda k: math.hypot(t.belief[k]["x"] - allspots[assign[k]][0], t.belief[k]["y"] - allspots[assign[k]][1]))
        s = allspots[assign[k]]
        n += 1
        try:
            t.block(k, (s[0], s[1], 0.0, 0, None), 0, n, "stage")
            moved += 1
        except (D.CycleFail, D.NoPlan, D.NoSpot, SystemExit) as e:
            D.status(f"stage: {k} not moved ({str(e)[:140]}); left where it is")
            todo.remove(k)
            st = arm_state()
            if st["grip"] < 0.75:
                raise
        t.put_blocks()
    base = next((k for k, i in assign.items() if i == 0 and at_spot(k)), None)
    D.status(f"stage: {moved} moves; base {base}; " + ", ".join(f"{k}@({t.belief[k]['x'] * 100:.0f},{t.belief[k]['y'] * 100:.0f})" for k in assign))
    return base


def use_sim(port, sim=None):
    """The same script against calib/station_sim.py on PORT: its URLs and clock, runs/sim_station/ (or sim, or the
    folder in SIM_STATION_DIR, one per parallel run) for the token, status, sessions, records and learn sets, no rsync."""
    global TALL, LEARN, time
    from hillclimb.runner.calib import arm_runner as AR
    base = f"http://127.0.0.1:{port}"
    clock = AR.sim_clock(base)
    AR.use_backend(AR.sim_urls(base), clock)
    time = clock
    sim = pathlib.Path(sim or os.environ.get("SIM_STATION_DIR") or _RUNS_ROOT / "sim_station")
    D.TOKEN, D.LOOP = sim / "ARM_OWNER", sim / "loop"
    D.STATUS, D.STOPFILE = D.LOOP / "STATUS.txt", D.LOOP / "STOP"
    D.LOOP.mkdir(parents=True, exist_ok=True)
    O.RECORDS = sim / "opstack" / "calib_records.jsonl"
    TALL, LEARN = sim / "tall", sim / "learn"
    Tall.sync = lambda self: None


def main():
    if "--sim" in sys.argv:
        i = sys.argv.index("--sim")
        use_sim(int(sys.argv[i + 1]), (_RUNS_ROOT / "sim_station" / f"p{sys.argv[i + 1]}") if "--sim-own-dir" in sys.argv else None)
        del sys.argv[i:i + 2]
    cmd = sys.argv[1]
    TALL.mkdir(parents=True, exist_ok=True)
    SPEED = speed_mode()
    D.status(f"speed mode {SPEED}: {SPEEDS[SPEED]}")
    session = TALL / time.strftime("%m%d-%H%M%S")
    if cmd in ("look", "dry"):
        t = Tall(session)
        t.q_look = None
        blocks, rejected, path = t.look_all("look", move=False)
        print(json.dumps({"blocks": {k: {kk: round(v, 4) if isinstance(v, float) else v for kk, v in b.items()} for k, b in blocks.items()},
                          "not_blocks": rejected, "overlay": str(path)}, indent=1))
        if cmd == "dry":
            order = sorted(blocks, key=lambda k: -min([math.hypot(blocks[k]["x"] - o["x"], blocks[k]["y"] - o["y"]) for j, o in blocks.items() if j != k] or [1]))
            t.dr = {"carry_z": 0.15, "over_dz": 0.01}
            res = t.tower_spot(order, len(order))
            print("ORDER", order, "SPOT", None if res is None else [round(v, 4) for v in res[:3]], "PLANS", None if res is None else res[3])
            print("LIMIT", getattr(t, "last_refusal", None))
            print("SPOT REJECTIONS", dict(t.why))
        return
    if cmd == "continue":  # continue SESSION ID [ID ...]: the tower standing from SESSION (bottom first)
        prev = TALL / sys.argv[2]
        lying_ids = {a.split(":")[0] for a in sys.argv[3:] if a.endswith(":lying")}
        tower = [a.split(":")[0] for a in sys.argv[3:]]
        fits, looks = {}, None
        for f in sorted(TALL.glob("*/calib_records.jsonl")):  # every tall session, the latest fit of each block wins
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["kind"] == "place" and r.get("fit") and r["block"] in tower:
                    fits[r["block"]] = r["fit"]
                if r["kind"] == "look":
                    looks = r["wrist"]
        D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        D.status(f"tall tower session {session.name} (continuing {prev.name}, standing {' / '.join(tower)}): token taken")
        D.rec("/start")
        clean = True
        try:
            t.belief = {**{k: dict(v) for k, v in (looks or {}).items() if k not in tower},
                        **{k: {"x": f["x"], "y": f["y"], "yaw": f["yaw"], "level": f["level"], "iou": f["iou"], "edge": f["edge"]} for k, f in fits.items()}}
            z = TABLE_Z
            for k in tower:  # the standing tower's bottoms from its blocks' heights, bottom first
                t.belief[k]["z0"] = z
                t.belief[k]["lying"] = k in lying_ids
                z = top_of(t.belief[k])
            t.refresh_free(tower)
            for k in tower:
                if k in lying_ids:  # refit on its side at its real bottom from the fresh look
                    b = t.belief[k]
                    img = cv2.resize(cv2.imread(str(t.last_look_img)), (640, 360), interpolation=cv2.INTER_AREA)
                    f = t.op.view.find(img, k, t.q_look, 0.08, shape_of(b), (b["x"], b["y"], b["yaw"]))
                    D.status(f"tall tower: {k} refit on its side at {bot(b) * 1000:.0f} mm: {None if f is None else {kk: round(v, 4) for kk, v in f.items() if isinstance(v, float)}}")
                    if f is not None and f["iou"] >= 0.6 and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) < 0.03:
                        b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"])
            t.put_blocks()
            free = [k for k in t.belief if k not in tower]
            order = sorted(free, key=lambda k: -min([math.hypot(t.belief[k]["x"] - o["x"], t.belief[k]["y"] - o["y"]) for j, o in t.belief.items() if j != k] or [1]))
            t.dr = {"carry_z": 0.15, "over_dz": 0.01}
            base = t.belief[tower[0]]
            got = t.plan_tower((base["x"], base["y"], base["yaw"]), order, tower)
            D.status(f"tall tower plan: {len(got)} more on {len(tower)} plan, order {got}; limit: {getattr(t, 'last_refusal', None) if len(got) < len(order) else 'none'}")
            built = t.build(got + [k for k in order if k not in got], (base["x"], base["y"], base["yaw"]), tower)
            D.status(f"tall tower done: {len(built)} blocks ({' / '.join(built)}), top at {(top_of(t.belief[built[-1]]) - TABLE_Z) * 1000:.0f} mm over the table")
        except BaseException as e:  # noqa: BLE001
            D.status(f"tall tower stopped: {type(e).__name__}: {e}")
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            finish(t, clean)
        return
    if cmd == "home":  # to the look pose from wherever the arm stopped, clear of every stack the records know
        D.take_token() if not D.TOKEN.exists() else None
        t = Tall(session)
        t.op.st["target"] = None
        t.belief = world_from_records()
        t.put_blocks()
        t.to_pose(LOOK_TIP, "look")
        D.status("home: at the look pose")
        D.release_token()
        return
    if cmd == "tbrun":  # G7: the truth builder's plan on the runner's beliefs, run through the runner's own moves
        import subprocess as _sp
        from hillclimb.runner.calib.tower_plan import H as _H
        D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        t.no_perception = True
        t.no_vet = True
        D.status(f"tbrun session {session.name}: token taken")
        D.rec("/start")
        clean = True
        try:
            t.multi_look(None)
            do_scatter_setup(t)
            t.put_blocks()
            src = {k: dict(b) for k, b in t.belief.items() if abs(bot(b) - TABLE_Z) < 0.005 and b.get("iou", 0) >= 0.5 and not k.startswith("pile")}
            inp, outp = session / "tb_in.json", session / "tb_plan.json"
            session.mkdir(parents=True, exist_ok=True)
            inp.write_text(json.dumps([{"id": k, "x": b["x"], "y": b["y"], "yaw": b["yaw"], "side": bool(b.get("lying"))} for k, b in src.items()]))
            mdir = os.environ.get("TB_MIRROR_DIR", str(pathlib.Path.home() / "hspike" / "G" / "ps"))
            r = _sp.run([sys.executable, "tb_mirror.py", str(inp), str(outp)], cwd=mdir, env=dict(os.environ, MUJOCO_GL="egl"),
                        capture_output=True, text=True, timeout=1500)
            if not outp.exists():
                raise D.Fatal(f"tbrun: no mirror plan ({r.stderr[-300:]})")
            plan = json.loads(outp.read_text())
            D.status(f"tbrun: mirror height {plan['height']} of target {plan['target']} ({plan['cubes']} cubes), stop {plan['stop']}; "
                     f"moves {[(s_['kind'], s_['block']) for s_ in plan['segments']]}")
            done_n, column, ended = 0, [], None
            for i, sg in enumerate(plan["segments"]):
                bid = sg["block"]
                lvl = max(0, int(round((sg["end_bottom"] - TABLE_Z) / _H)))
                sup = sg.get("support") if lvl > 0 else None
                if bid not in t.belief or (sup and sup not in t.belief):
                    D.status(f"tbrun: move {i + 1} {sg['kind']} {bid}: not in the runner's belief; replay ends")
                    break
                if lvl >= 7 or sg["kind"] not in ("stack", "tower", "table"):
                    D.status(f"tbrun: move {i + 1} {sg['kind']} {bid} L{lvl}: not a fingers-down move the runner makes; replay ends")
                    break
                tgt = (sg["end_x"], sg["end_y"], sg["end_yaw"], lvl, sup)
                try:
                    t.block(bid, tgt, 1, i + 1, "stack" if lvl else "table")
                except (D.CycleFail, D.NoPlan, D.NoSpot, SystemExit) as e:
                    D.status(f"tbrun: move {i + 1} {bid} L{lvl} failed ({str(e)[:160]}); replay ends")
                    st = arm_state()
                    if st["grip"] < 0.75:
                        raise
                    break
                done_n += 1
                if lvl > 0:
                    if not column and sup:
                        column = [sup]
                    if column and column[-1] == sup:
                        column.append(bid)
                D.status(f"tbrun: move {i + 1} done ({sg['kind']} {bid} L{lvl}{' on ' + sup if sup else ''})")
            tops = sorted(t.belief.items(), key=lambda kv: -bot(kv[1]))
            D.status(f"tbrun replay: {done_n} of {len(plan['segments'])} moves; column {column}; highest believed block {tops[0][0]} at {bot(tops[0][1]) * 1000:.0f} mm")
            if done_n < len(plan["segments"]) or len(column) < int(plan.get("target") or 0):
                blocks = dict(t.belief)
                cand = [k for k, b in blocks.items() if b.get("iou", 0) >= 0.5 and b["x"] >= 0.15 and not k.startswith("pile")
                        and k not in column and abs(bot(b) - TABLE_Z) < 0.005]
                order = sorted(cand, key=lambda k: (bool(blocks[k].get("lying")), -bot(blocks[k])))
                t.dr = {"carry_z": 0.15, "over_dz": 0.01}
                t.spot_area = ((0.19, 0.29), (-0.12, 0.15))
                D.status(f"tbrun fallback: the runner's own planner from the column {column} with {order}")
                res = t.best_sequence(order, tuple(column))
                if res is None:
                    D.status("tbrun fallback: no sequence plans")
                else:
                    spot, base_, seq, stop = res
                    built = t.run_sequence(spot, base_, seq)
                    D.status(f"tbrun fallback done: {len(built)} blocks ({' / '.join(built)}); planner stop {stop}")
        except BaseException as e:  # noqa: BLE001
            D.status(f"tbrun stopped: {type(e).__name__}: {e}")
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            finish(t, clean)
        return
    if cmd == "tallest":  # one tower as high as the reach allows, from every block it can get, near (24, 5)
        D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        t.no_perception = True
        t.no_vet = "--no-vet" in sys.argv
        D.status(f"tallest tower session {session.name}: token taken")
        D.rec("/start")
        clean = True
        try:
            t.multi_look(LOOKS_OLD if "--old-looks" in sys.argv else None)
            do_scatter_setup(t)
            # first phase: piles taken apart top-down, touching blocks and weak blobs pushed apart (--no-separate: off)
            t.tower_ids = set(sum(t.scatter_list, []))
            if "--no-separate" not in sys.argv:
                t.separate()
                do_scatter_setup(t)
            # --on A:B (seen by a person): the A block sits on the B block; A is then picked at level 1 (its re-look fits it there)
            for k_, a in enumerate(sys.argv):
                if a == "--on":
                    ca, cb = sys.argv[k_ + 1].split(":")
                    pairs = [(ka, kb) for ka, ba in t.belief.items() for kb, bb in t.belief.items()
                             if colour_of(ka) == ca and colour_of(kb) == cb and ka != kb and math.hypot(ba["x"] - bb["x"], ba["y"] - bb["y"]) < 0.07]
                    if pairs:
                        ka, kb = min(pairs, key=lambda kk: math.hypot(t.belief[kk[0]]["x"] - t.belief[kk[1]]["x"], t.belief[kk[0]]["y"] - t.belief[kk[1]]["y"]))
                        t.belief[ka].update(z0=top_of(t.belief[kb]), level=1, x=t.belief[kb]["x"], y=t.belief[kb]["y"], t_seen=0)
                        D.status(f"tallest: {ka} taken as sitting on {kb}")
            t.put_blocks()
            blocks = dict(t.belief)
            # a tower already standing close in (x 20-28 cm) is built on, not taken apart
            stacked, sbase = t.scatter_list
            tower = []
            if "--stage" in sys.argv:  # G6: the staging grid first; its base block is the tower's base
                sb_ = stage(t)
                blocks = dict(t.belief)
                if sb_ is not None:
                    tower = [sb_]
                    D.status(f"tallest: staged base {sb_} at ({blocks[sb_]['x'] * 100:.1f}, {blocks[sb_]['y'] * 100:.1f})")
            for k_, a in enumerate(sys.argv):  # --base COLOUR@X,Y: that block is the base, left where it lies
                if a == "--base":
                    c, xy = sys.argv[k_ + 1].split("@")
                    bx, by = (float(v) for v in xy.split(","))
                    cands_b = [k for k, b in blocks.items() if colour_of(k) == c and abs(bot(b) - TABLE_Z) < 0.005]
                    if cands_b:
                        tower = [min(cands_b, key=lambda k: math.hypot(blocks[k]["x"] - bx, blocks[k]["y"] - by))]
                        D.status(f"tallest: base {tower[0]} where it lies ({blocks[tower[0]]['x'] * 100:.1f}, {blocks[tower[0]]['y'] * 100:.1f})")
            for b0 in ([] if tower else sbase):
                if 0.18 <= blocks[b0]["x"] <= 0.34:
                    tower = sorted([k for k in stacked + [b0] if math.hypot(blocks[k]["x"] - blocks[b0]["x"], blocks[k]["y"] - blocks[b0]["y"]) < 0.03],
                                   key=lambda k: bot(blocks[k]))
                    break
            if tower:
                D.status(f"tallest: building on the standing tower {tower}")
            cand = [k for k, b in blocks.items() if b.get("iou", 0) >= 0.5 and b["x"] >= 0.15 and not k.startswith("pile") and k not in tower]
            order = sorted(cand, key=lambda k: (bool(blocks[k].get("lying")), -bot(blocks[k])))  # flat low, side on top; tops of stacks first
            D.status("tallest: blocks " + ", ".join(f"{k} ({blocks[k]['x'] * 100:.1f}, {blocks[k]['y'] * 100:.1f}) z{bot(blocks[k]) * 1000:.0f}"
                                                   f"{' side' if blocks[k].get('lying') else ''} iou {blocks[k].get('iou', 0):.2f}" for k in order))
            t.dr = {"carry_z": 0.15, "over_dz": 0.01}
            t.spot_area = ((0.19, 0.29), (-0.12, 0.15))
            if "--next-move-only" not in sys.argv:
                # the whole sequence (clearing moves, pushes, every stack place) planned per spot before anything moves
                res = t.best_sequence(order, tuple(tower))
                if res is None:
                    raise D.Fatal(f"no tower spot plans: {dict(t.why)}")
                spot, base_, seq, stop = res
                built = t.run_sequence(spot, base_, seq)
                top = top_of(t.belief[built[-1]]) - TABLE_Z if built else 0
                D.status(f"tallest tower done: {len(built)} blocks ({' / '.join(built)}), top {top * 1000:.0f} mm over the table")
                return
            if tower:
                b0 = blocks[tower[0]]
                x, y, yaw = b0["x"], b0["y"], b0["yaw"]
                got = t.plan_tower((x, y, yaw), order, tower)
            else:
                flat_band = [k for k in order if not blocks[k].get("lying") and 0.19 <= blocks[k]["x"] <= 0.29 and -0.12 <= blocks[k]["y"] <= 0.15
                             and blocks[k].get("iou", 0) >= 0.8 and abs(bot(blocks[k]) - TABLE_Z) < 0.005]
                res = None if flat_band else t.tower_spot(order, len(order))
                if res is None or not res[3]:
                    # no free spot: the block nearest (24, 5) in the area is the base, built on where it is
                    near = [k for k in order if 0.18 <= blocks[k]["x"] <= 0.34 and -0.12 <= blocks[k]["y"] <= 0.18]
                    if not near:
                        raise D.Fatal(f"no tower spot and no block to build on near (24, 5): {dict(t.why)}")
                    base_k = min(flat_band or near, key=lambda k: (bool(blocks[k].get("lying")), math.hypot(blocks[k]["x"] - 0.24, blocks[k]["y"] - 0.05)))
                    tower = [base_k]
                    order = [k for k in order if k != base_k]
                    b0 = blocks[base_k]
                    x, y, yaw = b0["x"], b0["y"], b0["yaw"]
                    got = t.plan_tower((x, y, yaw), order, tower)
                    D.status(f"tallest: building on {base_k} where it lies")
                else:
                    x, y, yaw, got = res
            D.status(f"tallest plan: {len(got)} of {len(order)} plan at ({x * 100:.1f}, {y * 100:.1f}); order {got}; limit {getattr(t, 'last_refusal', None)}")
            built = t.build(got + [k for k in order if k not in got], (x, y, yaw), tower)
            top = top_of(t.belief[built[-1]]) - TABLE_Z if built else 0
            D.status(f"tallest tower done: {len(built)} blocks ({' / '.join(built)}), top {top * 1000:.0f} mm over the table")
        except BaseException as e:  # noqa: BLE001
            D.status(f"tallest stopped: {type(e).__name__}: {e}")
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            finish(t, clean)
        return
    if cmd == "scatter":  # every standing stack the records know, top-down, each block to a spread-out reachable spot
        if not any(bot(b) > TABLE_Z + 0.005 for b in world_from_records().values()):
            D.status("scatter: no stack standing 2 high or more: skipped")
            return
        D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        D.status(f"scatter session {session.name}: token taken")
        D.rec("/start")
        clean = True
        try:
            t.look_all("look-scatter")
            do_scatter_setup(t)
            stacked, base = t.scatter_list
            D.status(f"scatter: tower top-down {stacked + base}")
            for n_, k in enumerate(stacked + base):
                wait_no_hands(t)
                try:
                    px, py, pyaw = t.pick_spot(k, [0])
                    t.block(k, (px, py, pyaw, 0, None), 1, n_ + 1, "unstack")
                except (D.NoPlan, D.NoSpot) as e:
                    D.status(f"scatter: {k} skipped ({str(e)[:200]})")
                    t.op.st["target"] = None
            D.status(f"scatter done: {len(t.block_times)} blocks moved")
        except BaseException as e:  # noqa: BLE001
            D.status(f"scatter stopped: {type(e).__name__}: {e}")
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            finish(t, clean)
        return
    if cmd == "loop":  # loop [--avoid X,Y[,TOP]] ...: stack 2-3 high at random spots, unstack to random spots, again
        avoid = []
        for k_, a in enumerate(sys.argv):
            if a == "--avoid":
                v = [float(u) for u in sys.argv[k_ + 1].split(",")]
                avoid.append((v[0], v[1], v[2] if len(v) > 2 else 0.11))
        D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        D.status(f"stack loop session {session.name}: token taken")
        D.rec("/start")
        clean = True
        try:
            t.stack_loop(avoid)
        except BaseException as e:  # noqa: BLE001
            D.status(f"stack loop stopped: {type(e).__name__}: {e}")
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            finish(t, clean)
        return
    if cmd == "recover":  # every piled or touching block apart on the table (a tumbled tower), recorded as recover episodes
        D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        D.rec("/start")
        clean = True
        try:
            t.recover()
        except BaseException as e:  # noqa: BLE001
            D.status(f"recover stopped: {type(e).__name__}: {e}")
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            finish(t, clean)
        return
    if cmd == "pile":  # pile SESSION TOWER_IDS... -- X Y COLOUR:SHAPE:LEVEL ...: unstack a pile at (X, Y) onto the tower
        i = sys.argv.index("--")
        prev, tower_args = TALL / sys.argv[2], sys.argv[3:i]
        px, py = float(sys.argv[i + 1]), float(sys.argv[i + 2])
        j = sys.argv.index("--then") if "--then" in sys.argv else len(sys.argv)
        then_args = sys.argv[j + 1:] if j < len(sys.argv) else []
        pile = [a.split(":") for a in sys.argv[i + 3:j] if not a.startswith("--")]
        lying_ids = {a.split(":")[0].split("@")[0] for a in tower_args + then_args if a.endswith(":lying")}
        parked = [a.split(":")[0] for a in then_args]
        tower = [a.split(":")[0].split("@")[0] for a in tower_args]
        fits = {}
        for f in sorted(TALL.glob("*/calib_records.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["kind"] == "place" and r.get("fit") and r["block"] in tower + parked:
                    fits[r["block"]] = r["fit"]
        for a in tower_args:  # id@x,y,yaw gives a tower block's pose directly (sim runs)
            if "@" in a:
                k, xyz = a.split(":")[0].split("@")
                x_, y_, w_ = (float(v) for v in xyz.split(","))
                fits[k] = {"x": x_, "y": y_, "yaw": w_, "iou": 0.9, "edge": False}
        dry = "--dry" in sys.argv
        if "--sim" in sys.argv:
            sys.argv = [a for a in sys.argv if a not in ("--sim",)]
        if not dry:
            D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        clean = True
        try:
            if not dry:
                D.status(f"tall tower session {session.name}: unstacking the pile at ({px * 100:.0f}, {py * 100:.0f}) onto {' / '.join(tower)}; token taken")
                D.rec("/start")
            t.belief = {k: {"x": f["x"], "y": f["y"], "yaw": f["yaw"], "level": n, "iou": f["iou"], "edge": f["edge"]} for n, (k, f) in enumerate((k, fits[k]) for k in tower)}
            z = TABLE_Z
            for k in tower:
                t.belief[k]["z0"], t.belief[k]["lying"] = z, k in lying_ids
                z = top_of(t.belief[k])
            for k in parked:
                f = fits[k]
                t.belief[k] = {"x": f["x"], "y": f["y"], "yaw": f["yaw"], "level": 0, "iou": f["iou"], "edge": f["edge"], "z0": TABLE_Z, "lying": k in lying_ids}
            tb = {k: dict(v) for k, v in t.belief.items()}
            wait_no_hands(t)
            blocks, rejected, path = t.look_all("look-pile", move=not dry)
            img = cv2.resize(cv2.imread(str(t.last_look_img)), (640, 360), interpolation=cv2.INTER_AREA)
            for k in tower:  # the tower from this look, each at its own bottom and shape
                b = tb[k]
                f = t.op.view.find(img, k, t.q_look, 0.08, shape_of(b), (b["x"], b["y"], b["yaw"]))
                if f is not None and f["iou"] >= 0.8 and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) < 0.02:
                    b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"])
            names = []
            gx, gy = px, py
            for colour, shape, lvl in pile:
                bid = f"{colour}#pile"
                b = {"x": gx, "y": gy, "yaw": 0.0, "level": int(lvl), "z0": TABLE_Z + int(lvl) * H, "lying": shape == "lying", "iou": 0.0, "edge": False}
                f = t.op.view.find(img, bid, t.q_look, 0.08, shape_of(b), (gx, gy, 0.0))
                D.status(f"pile {bid}: {shape} at level {lvl}: fit {None if f is None else {kk: round(v, 3) for kk, v in f.items() if isinstance(v, float)}}")
                if f is None or f["iou"] < 0.55 or math.hypot(f["x"] - gx, f["y"] - gy) > 0.08:
                    if not names:
                        raise D.Fatal(f"pile top {bid} not found as {shape} at level {lvl}")
                    # hidden under the blocks above it: placed under them, re-looked once they are off
                    D.status(f"pile {bid}: hidden for now; re-looked when its turn comes")
                    tb[bid] = b
                    names.append(bid)
                    continue
                b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"], edge=f["edge"])
                gx, gy = f["x"], f["y"]  # the blocks under it sit right below
                tb[bid] = b
                names.append(bid)
            t.belief = tb
            # the known blocks' own side faces (a tower's lower blocks fail the flat fit) are not obstacles
            for b in tb.values():
                if len(t.obstacles):
                    t.obstacles = t.obstacles[np.hypot(t.obstacles[:, 0] - b["x"], t.obstacles[:, 1] - b["y"]) > 0.045]
            t.put_blocks()
            t.dr = {"carry_z": 0.15, "over_dz": 0.01}
            # dry run: each pile block in turn onto the tower top, as high as reach plans
            b0 = t.belief[tower[0]]
            if "--move" in sys.argv:  # move the tower near x 18-22 cm, where the reach goes highest, then add the pile
                t.move_tower(tower, names, parked)
                return
            got = t.plan_tower((b0["x"], b0["y"], b0["yaw"]), names[:1], tower)
            # the pile comes off top first; the hidden ones plan for real once they are seen
            order = names if got else []
            D.status(f"pile plan: {order} onto the tower (planned {got}); limit: {getattr(t, 'last_refusal', None)}")
            if dry:
                return
            built = t.build_seq(order, tower)
            D.status(f"tall tower done: {len(built)} blocks ({' / '.join(built)}), top at {(top_of(t.belief[built[-1]]) - TABLE_Z) * 1000:.0f} mm over the table")
        except BaseException as e:  # noqa: BLE001
            D.status(f"tall tower stopped: {type(e).__name__}: {e}")
            if dry:
                raise
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            if not dry:
                finish(t, clean)
        return
    if cmd == "run":
        D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        D.status(f"tall tower session {session.name}: token taken")
        D.rec("/start")
        clean = True
        try:
            blocks, rejected, path = t.look_all("look")
            # stacks standing from earlier runs: their blocks from the records (level, shape), not the table-level look
            do_scatter_setup(t)
            stacked, sbase = t.scatter_list
            blocks = dict(t.belief)
            if stacked:
                D.status(f"tall tower: standing stacks from the records: {stacked + sbase}")
            if len([b for b in blocks.values() if b["x"] >= 0.15]) < 2:
                D.status(f"tall tower: {len(blocks)} free block(s): scattering the standing stacks instead")
                do_scatter_setup(t)
                stacked, base = t.scatter_list
                for n_, k in enumerate(stacked + base):
                    wait_no_hands(t)
                    try:
                        px, py, pyaw = t.pick_spot(k, [0])
                        t.block(k, (px, py, pyaw, 0, None), 1, n_ + 1, "unstack")
                    except (D.NoPlan, D.NoSpot) as e:
                        D.status(f"scatter: {k} skipped ({str(e)[:200]})")
                        t.op.st["target"] = None
                raise D.CycleFail("scattered instead of building")
            D.status(f"tall tower look: {len(blocks)} blocks " + ", ".join(f"{k} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) iou {b['iou']:.2f}"
                                                                          for k, b in blocks.items()) + f"; overlay {path}")
            # candidates: blocks fitted well (0.8 or better) that the arm can reach; a poor fit (a tilted pile, a hidden
            # block) stays in the belief as something to keep clear of, never picked
            # untangle first: a pile or a poorly seen piece (fit 0.55-0.8, or within 6 cm of another block) is picked and
            # set down alone at a spread-out spot, then the scene is looked at again
            pile = [k for k, b in blocks.items() if b.get("iou", 0) >= 0.55 and b["x"] >= 0.15 and not b.get("placed") and
                    (b.get("iou", 0) < 0.8 or any(math.hypot(b["x"] - o["x"], b["y"] - o["y"]) < 0.06 for j, o in blocks.items() if j != k))]
            n_untangled = 0
            skipped = set()
            while pile and n_untangled < 3:
                area = D.AREA
                D.AREA = ((0.22, 0.34), (-0.10, 0.15))
                k = pile[0]
                try:
                    px, py, pyaw = t.pick_spot(k, [0])
                    t.block(k, (px, py, pyaw, 0, None), 1, n_untangled + 1, "untangle")
                    n_untangled += 1
                except D.CycleFail as e:
                    D.status(f"untangle: {k} skipped ({str(e)[:200]})")
                    skipped.add(k)
                    t.op.st["target"] = None
                    st = arm_state()
                    if st["grip"] < 0.75 and st["goal_grip"] < 0.75:
                        raise D.Fatal(f"holding after an untangle failure: {e}")
                finally:
                    D.AREA = area
                blocks, rejected, path = t.look_all("look")  # the pile again (it changes when a piece comes off)
                do_scatter_setup(t)
                blocks = dict(t.belief)
                pile = [kk for kk, b in blocks.items() if kk not in skipped and b.get("iou", 0) >= 0.55 and b["x"] >= 0.15 and not b.get("placed") and
                        (b.get("iou", 0) < 0.8 or any(math.hypot(b["x"] - o["x"], b["y"] - o["y"]) < 0.06 for j, o in blocks.items() if j != kk))]
            cand = {k: b for k, b in blocks.items() if b.get("iou", 0) >= 0.8 and b["x"] >= 0.15}
            # flat blocks low (50 x 50 footprint, the steadiest), blocks on their side last: a side block adds 50 mm at
            # the top for the same reach; within each, the most isolated first
            order = sorted(cand, key=lambda k: (bool(blocks[k].get("lying")),
                                                -min([math.hypot(blocks[k]["x"] - o["x"], blocks[k]["y"] - o["y"]) for j, o in blocks.items() if j != k] or [1])))
            if len(order) < 2:
                raise D.CycleFail(f"fewer than 2 well-fitted free blocks ({order}); the next step scatters")
            t.dr = {"carry_z": 0.15, "over_dz": 0.01}
            res = t.tower_spot(order, len(order))
            if res is None:
                raise D.Fatal(f"no tower spot: {dict(t.why)}")
            x, y, yaw, got = res
            D.status(f"tall tower plan: {len(got)} of {len(order)} levels plan at ({x * 100:.1f}, {y * 100:.1f}) yaw {math.degrees(yaw):.0f}; "
                     f"order {got}; limit: {getattr(t, 'last_refusal', None) if len(got) < len(order) else 'none'}")
            built = t.build(got, (x, y, yaw))
            D.status(f"tall tower done: {len(built)} blocks, top at {(TABLE_Z + len(built) * H + 0.008) * 1000:.0f} mm over the table")
        except BaseException as e:  # noqa: BLE001
            D.status(f"tall tower stopped: {type(e).__name__}: {e}")
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            finish(t, clean)


def world_from_records():
    """The blocks the arm has placed and not picked up since, from every tall session's records in time order (a
    grasp removes whatever was placed where it gripped; ids are per session, so entries are keyed by place)."""
    rows = []
    for f in sorted(TALL.glob("*/calib_records.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines()]
    rows.sort(key=lambda r: r.get("t", ""))
    ov = TALL / "world_override.json"
    since = json.loads(ov.read_text()).get("since", "") if ov.exists() else ""
    rows = [r for r in rows if r.get("t", "") >= since]  # a reset: what the records say before it is no longer true
    world = {}
    for r in rows:
        if r["kind"] == "grasp" and r.get("wrist_look"):
            w = r["wrist_look"]
            z = bot(w) if ("level" in w or "z0" in w) else TABLE_Z
            for k in [k for k, b in world.items() if math.hypot(b["x"] - w["x"], b["y"] - w["y"]) < 0.03 and abs(bot(b) - z) < 0.02]:
                del world[k]
        if r["kind"] == "place" and r.get("fit"):
            f = r["fit"]
            k = r["block"]
            while k in world:
                k += "'"
            world[k] = {"x": f["x"], "y": f["y"], "yaw": f["yaw"], "level": f.get("level", 0), "iou": f["iou"], "edge": f["edge"],
                        "z0": f.get("z0"), "lying": f.get("lying", False), "t_seen": 0, "placed": True}
    # corrections seen by a person or another camera: a stack at (x, y) bottom first, each block flat or on its side;
    # bottoms recomputed from the shapes (runs/station/tall/world_override.json, applied when newer than the records)
    ov = TALL / "world_override.json"
    if ov.exists():
        for xy in json.loads(ov.read_text()).get("remove", []):  # stacks a person saw fall or cleared away
            for k in [k for k, b in world.items() if math.hypot(b["x"] - xy[0], b["y"] - xy[1]) < 0.03]:
                del world[k]
        for st in json.loads(ov.read_text()).get("stacks", []):
            members = sorted([k for k, b in world.items() if math.hypot(b["x"] - st["x"], b["y"] - st["y"]) < 0.03], key=lambda k: bot(world[k]))
            if len(members) != len(st["shapes"]):
                continue
            z = TABLE_Z
            for k, shape in zip(members, st["shapes"]):
                world[k]["lying"], world[k]["z0"], world[k]["confirmed"] = shape == "side", z, True
                z = top_of(world[k])
    return world


def do_scatter_setup(t):
    """The belief for a scatter: the standing stacks from the records, the rest from the look just taken."""
    placed = world_from_records()
    # a stack the look no longer shows (knocked over, moved by hand) is dropped: its top block's colour must be seen
    # within 6 cm of it
    for k in [k for k, b in placed.items() if bot(b) > TABLE_Z + 0.005]:
        if k not in placed:
            continue
        b = placed[k]
        under = [o for o, ob in placed.items() if math.hypot(ob["x"] - b["x"], ob["y"] - b["y"]) < 0.03]
        top = max(under, key=lambda o: bot(placed[o]))
        # a raised block fitted at table level lands up to 10 cm out from its real place; a stack a person confirmed
        # (world_override.json) is kept as is
        seen = True or placed[top].get("confirmed") or any(colour_of(i) == colour_of(top.rstrip("'")) and math.hypot(d["x"] - placed[top]["x"], d["y"] - placed[top]["y"]) < 0.12
                                                   for i, d in t.belief.items())
        if not seen:
            D.status(f"scatter: the stack at ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) is not in the look (top {top}): dropped")
            for o in under:
                placed.pop(o, None)
    stacked = sorted([k for k, b in placed.items() if bot(b) > TABLE_Z + 0.005], key=lambda k: -bot(placed[k]))
    base = [k for k, b in placed.items() if bot(b) <= TABLE_Z + 0.005 and any(math.hypot(b["x"] - placed[o]["x"], b["y"] - placed[o]["y"]) < 0.03 for o in stacked)]
    for k in list(t.belief):
        if any(math.hypot(t.belief[k]["x"] - placed[o]["x"], t.belief[k]["y"] - placed[o]["y"]) < 0.04 for o in stacked + base):
            del t.belief[k]
    t.belief.update({k: placed[k] for k in stacked + base})
    for k, b in list(t.belief.items()):
        # a stack's raised blocks, cast onto the table plane, land up to 10 cm out: those points are the stack itself
        r = 0.10 if k in stacked + base else 0.045
        if len(t.obstacles):
            t.obstacles = t.obstacles[np.hypot(t.obstacles[:, 0] - b["x"], t.obstacles[:, 1] - b["y"]) > r]
    t.put_blocks()
    t.scatter_list = (stacked, base)


def finish(t, clean):
    try:
        t.op.st["target"] = None
        if clean:
            t.to_pose(LOOK_TIP, "end")
            time.sleep(0.4)
            img, q, grip, stem = t.op.snap("final")
            D.status(f"tall tower final photos: {t.op.run / (stem + '-wrist.jpg')} and {t.op.run / (stem + '-overhead.jpg')}")
    except BaseException as e:  # noqa: BLE001
        clean = False
        D.status(f"end pose failed: {e}")
    t.saves.join()
    if os.environ.get("SIM_EVAL_OUT"):  # calib/sim_eval.py: the loop's own counts for this run
        pathlib.Path(os.environ["SIM_EVAL_OUT"]).write_text(json.dumps(D.jdump({
            "counts": t.counts, "block_times": t.block_times, "clean": clean,
            "belief": {k: {kk: b.get(kk) for kk in ("x", "y", "yaw", "level", "z0", "lying", "iou")} for k, b in t.belief.items()}})))
    D.rec("/stop")
    if clean:
        D.release_token()
        D.status("tall tower: token released")
    else:
        (D.TOKEN / "who.txt").write_text(f"data-loop HELD {time.strftime('%H:%M:%S')}: tall tower stopped unclean\n")
    t.sync()


if __name__ == "__main__":
    main()
