"""Where the runner thinks every block is, drawn on the overhead frame (Isaiah, Nov 8 11:43: "show me where it thinks
all cubes are every time it places"): each believed block as a 3D box (its footprint and height, at its believed
level) projected through the overhead camera (calib/cam_mid.json via overhead_fit.camera), outlined in its colour and
labelled with its id and level; standing stacks drawn thicker. Saved beside the session's frames and copied to
runs/station/<FT_ROOT>/belief-latest.jpg so the newest is always in one place.

    draw(belief, bgr, out_path, title="")
"""
from __future__ import annotations

import math
import os
import pathlib
import shutil

import cv2
import numpy as np

COLOURS = {"orange": (40, 140, 255), "green": (60, 200, 60), "blue": (230, 140, 40), "yellow": (40, 220, 240),
           "purple": (200, 90, 220), "pink": (200, 90, 220), "red": (40, 40, 230)}
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _box(x, y, yaw, z0, hx, hy, h):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for z in (z0, z0 + h):
        for dx, dy in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)):
            pts.append([x + c * dx - s * dy, y + s * dx + c * dy, z])
    return np.array(pts)


def draw(belief, bgr, out_path, title=""):
    from hillclimb.runner.calib import data_loop as D
    from hillclimb.runner.calib import overhead_fit as OF
    cam = OF.camera()
    img = bgr.copy()
    sx, sy = img.shape[1] / cam.W, img.shape[0] / cam.H
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for k, b in sorted(belief.items(), key=lambda kv: D.bot(kv[1])):
        colour = k.split("#")[0].rstrip("'")
        col = COLOURS.get(colour, (255, 255, 255))
        hx, hy = D.half_of(b)
        P = _box(b["x"], b["y"], b.get("yaw", 0.0), D.bot(b), hx, hy, D.hgt(b))
        uv, ok = cam.project(P)
        if not np.all(np.isfinite(uv)) or not np.all(np.abs(uv) < 1e5):
            continue
        uv = np.round(uv * [sx, sy]).astype(int)
        raised = D.bot(b) > D.TABLE_Z + 0.005
        for a, c in edges:
            cv2.line(img, tuple(uv[a]), tuple(uv[c]), col, 3 if raised else 2, cv2.LINE_AA)
        lvl = int(round((D.bot(b) - D.TABLE_Z) / D.H))
        top = uv[4:].mean(0).astype(int)
        label = f"{k} L{lvl}" + (f" {b.get('iou', 0):.2f}" if b.get("iou") is not None else "")
        cv2.putText(img, label, (int(top[0]) - 40, int(top[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (int(top[0]) - 40, int(top[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    if title:
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    latest = ROOT / "runs" / "station" / os.environ.get("FT_ROOT", "night_nov8") / "belief-latest.jpg"
    latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_path, latest)
    return out_path
