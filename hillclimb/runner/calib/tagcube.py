"""Isaiah's tag cubes BIG70 (70 mm, ids 30-35) and B39 (39.2 mm, ids 6-11), ArUco DICT_4X4_50: marker corners in the cube's frame, detection, and
drawing helpers.

Corners come from robot-studio-touch/robots/maker-arm/calibration/cubes.json (cube_models.py: the corner order matches
what the detector reports).
"""

import json
import pathlib

import cv2
import numpy as np

CUBES = pathlib.Path.home() / "Documents/BallahTech/robot-studio-touch/robots/maker-arm/calibration/cubes.json"
CUBE_IDS = {"BIG70": range(30, 36), "B39": range(6, 12)}  # Isaiah's printed cubes on the station's table
SIZE = {"BIG70": 0.070, "B39": 0.0392}
CORNERS = {int(i): np.array(c, float) for i, c in json.load(open(CUBES)).items() if any(int(i) in r for r in CUBE_IDS.values())}
CUBE_OF = {i: name for name, r in CUBE_IDS.items() for i in r}
_DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())


def _raw(g, scale):
    big = g if scale == 1 else cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cs, ids, _ = _DET.detectMarkers(big)
    return [] if ids is None else [(int(i), c.reshape(4, 2) / scale) for c, i in zip(cs, ids.ravel())]


def detect(img):
    """{id: 4x2 corner pixels} for the cubes' markers in a BGR or grey image (also read at 2x for small tags)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = {}
    for i, c in _raw(g, 1) + _raw(g, 2):
        if i in CORNERS and i not in found:
            found[i] = c
    out = {}
    if found:
        for i, c in found.items():
            if True:
                c = cv2.cornerSubPix(g, c.reshape(4, 1, 2).astype(np.float32), (3, 3), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
                out[int(i)] = c.reshape(4, 2).astype(float)
    return out


def edges(size=0.070):
    """A cube's 12 edges in its frame (for drawing its outline)."""
    h = size / 2
    v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
    return [(v[a], v[b]) for a in range(8) for b in range(a + 1, 8) if np.sum(v[a] != v[b]) == 1]
