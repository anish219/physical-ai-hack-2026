"""Images for the gate's questions: the reference-versus-now composite, zooms, and finding the power strip.

The composite matches the recorded tests exactly: top row the strip crop (reference left, now right) at 550 x 425 each,
bottom row the full overhead view at 550 x 310 each. Boxes are normalized (x0, y0, x1, y1) so any frame size works.
"""
from __future__ import annotations

import cv2
import numpy as np


def decode(jpeg: bytes) -> np.ndarray | None:
    if not jpeg:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def encode(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def clamp_box(box) -> list[float]:
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in box)
    if x1 - x0 < 0.05:
        x1 = min(1.0, x0 + 0.2)
        x0 = max(0.0, x1 - 0.2)
    if y1 - y0 < 0.05:
        y1 = min(1.0, y0 + 0.2)
        y0 = max(0.0, y1 - 0.2)
    return [x0, y0, x1, y1]


def crop(img: np.ndarray, box) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = clamp_box(box)
    return img[int(y0 * h):max(int(y0 * h) + 2, int(y1 * h)), int(x0 * w):max(int(x0 * w) + 2, int(x1 * w))]


def composite(ref_jpeg: bytes, now_jpeg: bytes, strip_box) -> bytes | None:
    ref, now = decode(ref_jpeg), decode(now_jpeg)
    if ref is None or now is None:
        return None
    zr = cv2.resize(crop(ref, strip_box), (550, 425), interpolation=cv2.INTER_AREA)
    zn = cv2.resize(crop(now, strip_box), (550, 425), interpolation=cv2.INTER_AREA)
    fr = cv2.resize(ref, (550, 310), interpolation=cv2.INTER_AREA)
    fn = cv2.resize(now, (550, 310), interpolation=cv2.INTER_AREA)
    return encode(np.vstack([np.hstack([zr, zn]), np.hstack([fr, fn])]))


def zoom(jpeg: bytes, box, width: int = 640) -> bytes | None:
    img = decode(jpeg)
    if img is None:
        return None
    c = crop(img, box)
    h = max(2, int(round(c.shape[0] * width / max(1, c.shape[1]))))
    return encode(cv2.resize(c, (width, h), interpolation=cv2.INTER_AREA))


def scaled(jpeg: bytes, width: int = 640) -> bytes | None:
    return zoom(jpeg, [0, 0, 1, 1], width)


def find_strip(jpeg: bytes) -> dict:
    """Finds the black power strip on the pale table and returns a crop box around it with room for the cables.

    Returns {"ok": bool, "box": [x0, y0, x1, y1], "strip": [...], "why": str}. Never raises.
    """
    try:
        img = decode(jpeg)
        if img is None:
            return {"ok": False, "why": "no frame"}
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dark = ((hsv[..., 2] < 70) & (hsv[..., 1] < 90)).astype(np.uint8) * 255
        table = (hsv[..., 2] > 150) & (hsv[..., 1] < 60)  # the pale table top
        rows = np.where(table.mean(axis=1) > 0.25)[0]
        top = int(rows.min()) if rows.size else int(0.3 * h)  # ignore the room above the table
        dark[:top] = 0
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        best, best_score = None, 0.0
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            frac = area / float(w * h)
            if not 0.001 <= frac <= 0.05:
                continue
            long_side, short_side = max(bw, bh), max(1, min(bw, bh))
            elong = long_side / short_side
            fill = area / float(bw * bh)
            if elong < 1.8 or fill < 0.35:
                continue
            # surrounded by table: the strip lies on the table, unlike arm parts at the frame edge
            pad = 6
            ring = table[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
            around = float(ring.mean()) if ring.size else 0.0
            score = around * min(elong, 6.0) * fill
            if score > best_score:
                best, best_score = (x, y, bw, bh), score
        if best is None:
            return {"ok": False, "why": "no strip-shaped dark blob on the table"}
        x, y, bw, bh = best
        cx, cy = x + bw / 2.0, y + bh / 2.0
        cw, ch = max(2.6 * bw, 0.30 * w), max(1.9 * bh, 0.42 * h)
        box = clamp_box([(cx - cw / 2) / w, (cy - ch / 2) / h, (cx + cw / 2) / w, (cy + ch / 2) / h])
        strip = [x / w, y / h, (x + bw) / w, (y + bh) / h]
        return {"ok": True, "box": [round(v, 4) for v in box], "strip": [round(v, 4) for v in strip], "score": round(best_score, 3)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": f"error: {type(e).__name__}: {e}"}
