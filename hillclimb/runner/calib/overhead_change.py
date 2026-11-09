"""Did a block that was not moved change between two overhead frames (before and after one block move)?

The knock-over guard of the Nov 8 night run (a tower knocked over by a place spoiled fine-tune #1's move 24): every
block colour's mask in both frames, and the pixels that changed between them, except the ones the move itself explains:
  - the arm: dark pixels in either frame (grown 9 px), and pixels at least 25 % darker in one frame (its shadow);
  - the moved block: pixels that are its colour in either frame (it left them or now covers them; a taller stack also
    hides what is behind it from this angle);
  - of the moved block's own colour (two orange blocks: one big, one small), the largest area that went and the
    largest that came (where it was and where it is);
  - pixels within 20 px of the moved block's own colour (where it was, where it is): its shaded faces read as another
    colour (a yellow block's front face as orange, Nov 8 01:43); a knocked neighbour moves further than that.
Anything else of a block colour that went or came, over MIN_PX at 640 x 360 (a far small block's top face is about
400 px), is a change. Only the left arm's part of the table counts (TABLE_ROI): no background, no right side (the
basket, the headset, the right arm); and orange starts at saturation 90 (the foam reads 93-120, skin 73-98). It is not
a person check (Isaiah, Nov 8: the table is kept clear; nothing may stop the run for someone in the background).

    python -m calib.overhead_change BEFORE.jpg AFTER.jpg COLOUR [OUT.jpg]
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

COLOURS = ("yellow", "blue", "orange", "purple", "green")
# OpenCV HSV hue bins and saturation floors (calib/tall_tower.HUES, measured on the station's light)
HUES = {"yellow": ([(22, 38)], 35), "orange": ([(5, 20)], 90), "blue": ([(88, 125)], 70),
        "purple": ([(135, 176)], 55), "green": ([(40, 88)], 55)}
VAL_MIN = 60
W, H = 640, 360
MIN_PX = 120
ARM_V = 70  # the black arm and its cables
SHADE = 0.75
TABLE_ROI = (0, 160, 460, 360)  # x0, y0, x1, y1 at 640 x 360: the table's near-left part, under the overhead's table edge


def mask(hsv, colour):
    bins, sat = HUES[colour]
    m = np.zeros(hsv.shape[:2], bool)
    for lo, hi in bins:
        m |= (hsv[..., 0] >= lo) & (hsv[..., 0] < hi)
    m = (m & (hsv[..., 1] >= sat) & (hsv[..., 2] >= VAL_MIN)).astype(np.uint8)
    x0, y0, x1, y1 = TABLE_ROI
    roi = np.zeros_like(m)
    roi[y0:y1, x0:x1] = 1
    m &= roi
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def big_parts(m, keep_largest=False, merge=0):
    """Connected areas of m over MIN_PX (after a 5 px opening), largest first; keep_largest=False drops the largest.
    merge: pieces this close (px) are one area (a block cut in two by the arm's shadow edge)."""
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    grown = cv2.dilate(m, np.ones((2 * merge + 1, 2 * merge + 1), np.uint8)) if merge else m
    n, lab, _, _ = cv2.connectedComponentsWithStats(grown)
    parts = []
    for i in range(1, n):
        own = (lab == i) & (m > 0)
        a = int(own.sum())
        if a >= MIN_PX:
            ys, xs = np.nonzero(own)
            parts.append((a, (int(xs.mean()), int(ys.mean())), own))
    parts.sort(key=lambda p: -p[0])
    return parts if keep_largest else parts[1:]


def changed(before, after, moved, out=None):
    """(changed, list of (colour, went/came, px, centre)) between two overhead BGR frames; moved: the moved block's
    colour. out: a path for a picture of the two frames with the unexplained changes outlined."""
    A = cv2.resize(before, (W, H), interpolation=cv2.INTER_AREA)
    B = cv2.resize(after, (W, H), interpolation=cv2.INTER_AREA)
    ha, hb = cv2.cvtColor(A, cv2.COLOR_BGR2HSV), cv2.cvtColor(B, cv2.COLOR_BGR2HSV)
    va, vb = ha[..., 2].astype(float), hb[..., 2].astype(float)
    grow = np.ones((19, 19), np.uint8)
    arm_a = cv2.dilate(((va < ARM_V) | (va < SHADE * vb)).astype(np.uint8), grow) > 0
    arm_b = cv2.dilate(((vb < ARM_V) | (vb < SHADE * va)).astype(np.uint8), grow) > 0
    ma = {c: mask(ha, c) for c in COLOURS}
    mb = {c: mask(hb, c) for c in COLOURS}
    # where the moved block was (A) and is (B): its own colour mask, and 20 px around it (its shaded faces, any hue)
    near_a = cv2.dilate(ma[moved].astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    near_b = cv2.dilate(mb[moved].astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    seen_a, seen_b = arm_a | ma[moved] | near_a, arm_b | mb[moved] | near_b
    found = []
    for c in COLOURS:
        went = ma[c] & ~mb[c]
        came = mb[c] & ~ma[c]
        if c == moved:
            parts = [("went", p) for p in big_parts(went & ~arm_b, merge=20)] + [("came", p) for p in big_parts(came & ~arm_a, merge=20)]
        else:
            parts = [("went", p) for p in big_parts(went & ~seen_b, True)] + [("came", p) for p in big_parts(came & ~seen_a, True)]
        found += [(c, how, a, ctr, m) for how, (a, ctr, m) in parts]
    if out is not None:
        pic = np.hstack([A, B]).copy()
        for c, how, a, ctr, m in found:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for dx in (0, W):
                cv2.drawContours(pic[:, dx:dx + W], cs, -1, (0, 0, 255), 2)
            cv2.putText(pic, f"{c} {how} {a}px", (W + ctr[0] - 30, ctr[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(pic, "before", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pic, f"after (moved: {moved}){'  CHANGED' if found else ''}", (W + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if found else (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out), pic, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bool(found), [(c, how, a, ctr) for c, how, a, ctr, _ in found]


if __name__ == "__main__":
    a, b = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])
    print(changed(a, b, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None))
