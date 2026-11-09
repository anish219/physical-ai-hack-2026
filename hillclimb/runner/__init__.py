"""The scripted tower runner (see README.md). make_backend() is what hillclimb.left builds it with."""
from __future__ import annotations

import os
from urllib.parse import urlparse

REAL_LEFT_PORT = 8691


def make_backend(settings: dict, arm=None):
    """A RunnerBackend for the left controller. settings: the controller's settings dict; optional keys
    sim_port, helper_spots ([[x, y], ...] m) and tilt_theta_deg / tilt_up_mm / tilt_sag_mm. arm: the controller's
    ArmClient; when its base is a local port other than the real left arm server's, that port is taken as the sim's.
    The runner keeps its own URLs (env LEFT_URL, TEAM_STATE_URL, DREAM_STATE_URL, REC_URL) on the real station."""
    from hillclimb.runner.backend import RunnerBackend
    settings = dict(settings or {})
    sim_port = settings.get("sim_port") or os.environ.get("RUNNER_SIM_PORT")
    base = getattr(arm, "base", "") or ""
    if not sim_port and base:
        u = urlparse(base)
        if u.hostname in ("127.0.0.1", "localhost") and u.port and u.port != REAL_LEFT_PORT:
            sim_port = u.port
    spots = settings.get("helper_spots") or []
    if not spots and os.environ.get("RUNNER_HELPER_SPOTS"):  # "x,y;x,y" in cm
        spots = [[float(v) / 100 for v in sp.split(",")] for sp in os.environ["RUNNER_HELPER_SPOTS"].split(";") if sp]
    log = None
    try:
        from hillclimb.common.log import JsonlLog
        log = JsonlLog("runner")
    except Exception:  # noqa: BLE001
        pass
    return RunnerBackend(sim_port=int(sim_port) if sim_port else None, helper_spots=[tuple(s) for s in spots],
                         tilt_theta_deg=float(settings.get("tilt_theta_deg", 40.0)),
                         tilt_up_mm=float(settings.get("tilt_up_mm", 25.0)),
                         tilt_sag_mm=float(settings.get("tilt_sag_mm", 25.0)), log=log)
