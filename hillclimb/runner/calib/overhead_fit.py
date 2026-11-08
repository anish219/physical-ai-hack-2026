"""A block fitted in the overhead frame at a given height: the second camera of the standing check (calib/data_loop.py).

The camera: on the real station calib/cam_mid.json (vision/locate.Camera, 1280 x 720, the lens k1 included); against
calib/station_sim.py the sim's own free view (lookat 0.32, 0.05, 0; distance 0.95 m; azimuth 180; elevation -62;
fovy 45 deg; 640 x 480), the frame /overhead.jpg serves. Which one follows opstack.URL: the real arm server's URL is
the real camera, anything else the sim's.

fit(img, colour, shape, guess) works as WristView._fit: the colour blob nearest the guess (projected at the shape's
mid height), the box's projected hull matched to it (IoU) over x, y and yaw. Returns dict(x, y, yaw, iou, px, blob)
or None when no blob of that colour is within `near` px of where the guess projects.
"""
from __future__ import annotations

import json
import math
import pathlib

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL_URL = "http://127.0.0.1:8691"
SIM_VIEW = dict(lookat=(0.32, 0.05, 0.0), distance=0.95, azimuth=180.0, elevation=-62.0, fovy=45.0, W=640, H=480)


class SimOverhead:
    def __init__(self, v=SIM_VIEW):
        az, el = math.radians(v["azimuth"]), math.radians(v["elevation"])
        self.fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        self.pos = np.asarray(v["lookat"], float) - v["distance"] * self.fwd
        self.right = np.cross(self.fwd, [0, 0, 1.0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        self.W, self.H = v["W"], v["H"]
        self.f = 0.5 * self.H / math.tan(math.radians(v["fovy"]) / 2)

    def project(self, P):
        d = np.asarray(P, float).reshape(-1, 3) - self.pos
        z = d @ self.fwd
        zz = np.maximum(z, 1e-6)
        u = self.W / 2 + self.f * (d @ self.right) / zz
        v = self.H / 2 - self.f * (d @ self.up) / zz
        return np.stack([u, v], 1), z > 0.05


class RealOverhead:
    def __init__(self):
        from hillclimb.runner.vision.locate import Camera
        import os
        c = json.loads((ROOT / "calib" / os.environ.get("OVERHEAD_CAM", "cam_mid_nov8.json")).read_text())
        if "fovy_deg" not in c:  # the Nov 8 refit gives f (pixels) instead
            c["fovy_deg"] = 2 * math.degrees(math.atan(c["height"] / 2 / c["f"]))
        self.cam = Camera.from_calib(c)
        self.W, self.H = c["width"], c["height"]

    def project(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        return np.asarray(self.cam.project(P), float).reshape(-1, 2), np.ones(len(P), bool)


_CAMS = {}


def camera(url=None):
    if url is None:
        from hillclimb.runner.calib import opstack as O
        url = O.URL
    kind = "real" if url == REAL_URL else "sim"
    if kind not in _CAMS:
        _CAMS[kind] = RealOverhead() if kind == "real" else SimOverhead()
    return _CAMS[kind]


def colour_mask(bgr, colour):
    from hillclimb.runner.calib.tower_plan import WristView
    return WristView.colour_mask(bgr, colour, np.zeros(bgr.shape[:2], np.uint8))


def hull(cam, shape, x, y, yaw):
    uv, ok = cam.project(shape.corners(x, y, yaw))
    if not ok.all() or not np.all(np.abs(uv) < 1e5):
        return None
    return cv2.convexHull(np.round(uv).astype(np.int32))


def fit(bgr, colour, shape, guess, cam=None, near=80, free=True, mask=None):
    """free: x, y, yaw fitted (Nelder-Mead from the guess); else the IoU of the guess as it is."""
    cam = cam or camera()
    if bgr.shape[1] != cam.W:
        bgr = cv2.resize(bgr, (cam.W, cam.H), interpolation=cv2.INTER_AREA)
    m = colour_mask(bgr, colour) if mask is None else mask
    n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 30]
    g_uv, ok = cam.project([[guess[0], guess[1], shape.z0 + shape.dims[2] / 2]])
    if not comps or not ok.all():
        return None
    pick = min(comps, key=lambda i: np.linalg.norm(cen[i] - g_uv[0]))
    if np.linalg.norm(cen[pick] - g_uv[0]) > near * cam.W / 640:
        return None
    blob = (lab == pick).astype(np.uint8)
    cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(blob, cs, -1, 1, -1)
    blob = blob > 0
    x0, y0, w0, h0 = stats[pick, :4]
    pad = int(60 * cam.W / 640)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(cam.W, x0 + w0 + pad), min(cam.H, y0 + h0 + pad)
    target = blob[Y0:Y1, X0:X1]

    def cost(p):
        h = hull(cam, shape, *p)
        if h is None:
            return 1.0
        mm = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        cv2.fillConvexPoly(mm, h - [X0, Y0], 1)
        mm = mm > 0
        return 1 - np.sum(mm & target) / max(np.sum(mm | target), 1)

    if not free:
        return dict(x=float(guess[0]), y=float(guess[1]), yaw=float(guess[2]), iou=float(1 - cost(guess)), px=int(blob.sum()), blob=int(pick))
    from scipy.optimize import minimize
    best = None
    for yaw in (guess[2], guess[2] + shape.period / 2):
        x_init = np.array([guess[0], guess[1], yaw])
        simplex = np.array([x_init, x_init + [0.01, 0, 0], x_init + [0, 0.01, 0], x_init + [0, 0, 0.25]])
        r = minimize(cost, x_init, method="Nelder-Mead", options={"xatol": 3e-4, "fatol": 1e-4, "initial_simplex": simplex, "maxiter": 250})
        if best is None or r.fun < best.fun:
            best = r
    x, y, yaw = best.x
    return dict(x=float(x), y=float(y), yaw=float(shape.fold(yaw)), iou=float(1 - best.fun), px=int(blob.sum()), blob=int(pick))
