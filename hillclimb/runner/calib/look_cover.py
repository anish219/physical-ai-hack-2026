"""Wrist look poses that together see the whole reachable table, from the calibrated wrist camera model
(calib/tower_plan.WristView: the mount, f, k1, k2, the jaws' CAD drawn through it).

A block at (x, y) counts as seen from a pose when, at yaw 0 and 45 deg, flat on the table:
  - all 8 corners of its box project inside the frame, 10 px from every edge, and inside the lens model's good zone
    (normalised radius under 1.35; past about 1.6 the k2 term folds the image back on itself);
  - its silhouette is at least 600 px (at 640x360) and at least 85 % of it is clear of the jaws.
The region is x 15-50 cm, y -30..+30 cm (arm frame), on a 1 cm grid. Candidate poses are fingers-down looks (the only
kind the planner and the camera rule allow): fingertips x 18-40 cm, y -15..+25 cm, the highest legal height of
19-23 cm, the tool frame's jaw yaw 0-345 deg in 15 deg steps (the frame held exactly: the camera sits on one side of
the jaws), each solved by IK and passed by TowerSafety.check_pose. A greedy set cover gives three looks; the best pair
among the 150 widest single looks, refined in 1-2 cm and 5 deg steps, covers it all with two (tall_tower.LOOKS, Nov 8:
100 %, and 99.9 % with either look 1 cm or 3 deg off).

    python -m calib.look_cover            # the search (about 10 min): the chosen poses, each one's coverage, the map
    python -m calib.look_cover --new      # tall_tower.LOOKS
    python -m calib.look_cover --old      # the three looks tall_tower used before (standard, +-10 cm in y)
"""
from __future__ import annotations

import math
import sys

import numpy as np

from hillclimb.runner.calib.opstack import LOOK_SEED, LOOK_TIP, TABLE_Z
from hillclimb.runner.calib.tower_plan import FACE, H, Kin, Shape, WristView, frame_down
from hillclimb.runner.calib.tower_safety import TowerSafety, outward_ok

REGION = ((0.15, 0.50), (-0.30, 0.30))
STEP = 0.01
MARGIN_PX, R_MAX, MIN_PX, CLEAR = 10, 1.35, 600, 0.85
FLAT = Shape((FACE, FACE, H), TABLE_Z)


def in_view(view, q, blocks, opening=0.08, jaws=None, margin=MARGIN_PX):
    """For each block dict (x, y, yaw, z0 optional): True when its whole box is in the frame from model joints q by
    the rule above (corners inside the margin and the lens's good zone, the box centre and 85 % of its corners clear of
    the jaws). A cheap per-block form of Cover.seen for the run: which look sees a block, which blocks a look saw."""
    cam = view.pose(q, opening)
    Rc, pc = cam
    jaws = view.jaws_mask(q, opening) if jaws is None else jaws
    out = []
    for b in blocks:
        z0 = b.get("z0", TABLE_Z)
        P = Shape((FACE, FACE, H), z0).corners(b["x"], b["y"], b.get("yaw", 0.0))
        Pc = (P - pc) @ Rc
        if (Pc[:, 2] < 0.02).any() or (np.hypot(Pc[:, 0] / Pc[:, 2], Pc[:, 1] / Pc[:, 2]) > R_MAX).any():
            out.append(False)
            continue
        uv, _ = view.project(cam, np.vstack([P, P.mean(0)]))
        if (uv[:, 0] < margin).any() or (uv[:, 0] > view.W - margin).any() or (uv[:, 1] < margin).any() or (uv[:, 1] > view.H - margin).any():
            out.append(False)
            continue
        hit = [bool(jaws[min(int(v), view.H - 1), min(int(u), view.W - 1)]) for u, v in uv]
        out.append(not hit[-1] and sum(hit[:-1]) <= 1)
    return out


class Cover:
    def __init__(self):
        self.kin = Kin()
        self.view = WristView(self.kin)
        self.safe = TowerSafety()
        self.xs = np.arange(REGION[0][0], REGION[0][1] + 1e-6, STEP)
        self.ys = np.arange(REGION[1][0], REGION[1][1] + 1e-6, STEP)
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.grid = np.stack([gx.ravel(), gy.ravel()], 1)

    def tip(self, q):
        pos, R = self.kin.fk(q)
        return pos + self.kin.tip * R[:, 2]

    def solve(self, xyz, yaw_deg, seed=LOOK_SEED):
        """Model joints for fingers-down fingertips at xyz in exactly the frame frame_down(yaw) (the jaw axis along
        yaw; the camera sits on the tool's +y side, so yaw and yaw + 180 see different things), or None when that
        frame breaks the camera rule, has no IK or fails TowerSafety."""
        best = None
        xyz = np.asarray(xyz, float)
        for k in range(1):
            R = frame_down(math.radians(yaw_deg) + k * math.pi / 2)
            if not outward_ok(R, xyz[:2]):
                continue
            q, ok = self.kin.ik_one(xyz + [0, 0, self.kin.tip], R, seed)
            if not ok:
                q, _ = self.kin.ik(xyz + [0, 0, self.kin.tip], [R], seed)
                if q is None:
                    continue
            if np.linalg.norm(self.tip(q) - xyz) > 0.003 or not self.safe.check_pose(q, 0.08)[0]:
                continue
            c = float(np.abs(q - seed).sum())
            if best is None or c < best[0]:
                best = (c, q)
        return None if best is None else best[1]

    def seen(self, q, opening=0.08):
        """Boolean per grid point: a flat block there is fully in view (see the module doc)."""
        import cv2
        cam = self.view.pose(q, opening)
        Rc, pc = cam
        jaws = self.view.jaws_mask(q, opening) > 0
        ok = np.ones(len(self.grid), bool)
        for yaw in (0.0, math.pi / 4):
            c, s = math.cos(yaw), math.sin(yaw)
            loc = np.array([[a * FACE / 2, b * FACE / 2] for a in (-1, 1) for b in (-1, 1)])
            off = loc @ np.array([[c, s], [-s, c]])
            P = np.concatenate([np.concatenate([self.grid[:, None, :] + off[None], np.full((len(self.grid), 4, 1), z)], 2)
                                for z in (TABLE_Z, TABLE_Z + H)], 1)  # (n, 8, 3)
            Pc = (P.reshape(-1, 3) - pc) @ Rc
            front = Pc[:, 2] > 0.02
            zc = np.maximum(Pc[:, 2], 1e-6)
            r = np.hypot(Pc[:, 0] / zc, Pc[:, 1] / zc)
            uv, _ = self.view.project(cam, P.reshape(-1, 3))
            inside = front & (r < R_MAX) & (uv[:, 0] >= MARGIN_PX) & (uv[:, 0] <= self.view.W - MARGIN_PX) \
                & (uv[:, 1] >= MARGIN_PX) & (uv[:, 1] <= self.view.H - MARGIN_PX)
            inside = inside.reshape(-1, 8).all(1)
            uv = uv.reshape(-1, 8, 2)
            for i in np.nonzero(inside & ok)[0]:
                hull = cv2.convexHull(np.round(uv[i]).astype(np.int32))
                x0, y0, w, h = cv2.boundingRect(hull)
                mm = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(mm, hull - [x0, y0], 1)
                a = int(mm.sum())
                vis = int((mm.astype(bool) & ~jaws[y0:y0 + h, x0:x0 + w]).sum())
                if vis < MIN_PX or vis < CLEAR * a:
                    inside[i] = False
            ok &= inside
        return ok

    def candidates(self):
        out = []
        for x in np.arange(0.18, 0.401, 0.03):
            for y in np.arange(-0.15, 0.251, 0.05):
                for yaw in range(0, 360, 15):
                    for z in (0.23, 0.215, 0.20, 0.19):  # 8 mm under the reach limit (REACH_Z 238 mm) at most
                        q = self.solve([x, y, z], yaw)
                        if q is not None:
                            out.append(((round(float(x), 3), round(float(y), 3), z, float(yaw)), q))
                            break
        return out

    def choose(self, cands, maps, need):
        """Greedy set cover of `need`; ties go to the pose nearest the standard look."""
        left = need.copy()
        chosen = []
        while left.any():
            gain = [(int((m & left).sum()), -math.hypot(p[0] - LOOK_TIP[0], p[1] - LOOK_TIP[1])) for (p, _), m in zip(cands, maps)]
            i = max(range(len(cands)), key=lambda k: gain[k])
            if gain[i][0] == 0:
                break
            chosen.append(i)
            left &= ~maps[i]
        return chosen

    def show(self, M, title):
        M = M.reshape(len(self.xs), len(self.ys))
        print(title)
        for i in range(len(self.xs) - 1, -1, -1):
            print(f"  x {self.xs[i] * 100:4.0f} " + "".join("#" if M[i, j] else "." for j in range(len(self.ys))))
        print("         y -30 " + " " * (len(self.ys) - 12) + "+30")


    def refine_pair(self, pa, pb):
        """The best pair near (pa, pb): x 1 cm, y 2 cm, z down to 19 cm, yaw 5 deg steps within 10 deg; ties go to the
        pair that loses least with either pose 1 cm or 3 deg off (the real arm's error)."""
        def around(p):
            out = []
            for dx in (-0.01, 0.0, 0.01):
                for dy in (-0.02, 0.0, 0.02):
                    for z in sorted({p[2], 0.19}):
                        for dyaw in (-10, -5, 0, 5, 10):
                            pp = (round(p[0] + dx, 3), round(p[1] + dy, 3), z, p[3] + dyaw)
                            q = self.solve(pp[:3], pp[3])
                            if q is not None:
                                out.append((pp, self.seen(q)))
            return out
        A, B = around(pa), around(pb)
        full = sorted(((ma | mb).mean(), a, b, ma, mb) for a, ma in A for b, mb in B)[-20:]

        def worst(a, b, ma, mb):
            w = 1.0
            for p, other in ((a, mb), (b, ma)):
                for dx, dy, dyaw in ((0.01, 0, 0), (-0.01, 0, 0), (0, 0.01, 0), (0, -0.01, 0), (0, 0, 3), (0, 0, -3)):
                    q = self.solve([p[0] + dx, p[1] + dy, p[2]], p[3] + dyaw)
                    if q is not None:
                        w = min(w, (self.seen(q) | other).mean())
            return w
        best = max(full, key=lambda f: (round(f[0], 4), worst(*f[1:])))
        return best[1], best[2]


def main():
    cv = Cover()
    if "--old" in sys.argv:
        poses = [(LOOK_TIP[0], LOOK_TIP[1] + dy, LOOK_TIP[2], LOOK_TIP[3]) for dy in (0.0, 0.10, -0.10)]
    elif "--new" in sys.argv:
        from hillclimb.runner.calib.tall_tower import LOOKS
        poses = LOOKS
    else:
        cands = cv.candidates()
        print(f"{len(cands)} legal candidate poses", flush=True)
        maps = [cv.seen(q) for _, q in cands]
        union = np.any(maps, 0)
        print(f"every candidate together sees {union.mean() * 100:.1f} % of the region")
        chosen = cv.choose(cands, maps, union)
        poses = [cands[i][0] for i in chosen]
        print(f"greedy: {len(poses)} looks {poses}")
        # fewer: the best pair among the 150 widest single looks, refined around
        top = sorted(range(len(cands)), key=lambda i: -maps[i].mean())[:150]
        i, j = max(((i, j) for n, i in enumerate(top) for j in top[n + 1:]), key=lambda ij: (maps[ij[0]] | maps[ij[1]]).mean())
        print(f"best pair {cands[i][0]} {cands[j][0]}: {(maps[i] | maps[j]).mean() * 100:.2f} %", flush=True)
        pa, pb = cv.refine_pair(cands[i][0], cands[j][0])
        qa, qb = cv.solve(pa[:3], pa[3]), cv.solve(pb[:3], pb[3])
        if (cv.seen(qa) | cv.seen(qb)).mean() >= np.any([maps[k] for k in chosen], 0).mean() - 1e-9:
            poses = [pa, pb]
    U = np.zeros(len(cv.grid), bool)
    for p in poses:
        q = cv.solve(p[:3], p[3])
        if q is None:
            print("no legal pose", p)
            continue
        m = cv.seen(q)
        U |= m
        print(f"pose tip ({p[0] * 100:.1f}, {p[1] * 100:.1f}, {p[2] * 100:.1f}) cm yaw {p[3]:.0f}: {m.mean() * 100:.1f} % alone, union {U.mean() * 100:.1f} %")
    cv.show(U, f"union of {len(poses)} looks: {U.mean() * 100:.1f} % of x 15-50, y -30..30 cm")
    print("POSES", [tuple(round(v, 3) for v in p) for p in poses])


if __name__ == "__main__":
    main()
