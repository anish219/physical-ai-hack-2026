"""Start a station sim (calib.station_sim from the pah-synth copy on the station) on a port, with the settings the
spikes found right: EGL rendering, the Nov 8 overhead camera, half render scale.

    ~/dream-env/bin/python -m hillclimb.sim.launch --port 9131 [--layout table --seed 5 --speed 1]

Prints the sim's PID; stop it with kill. Point the harness at it with LEFT_URL=http://127.0.0.1:<port>.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

SIM_HOME = pathlib.Path(os.environ.get("SIM_HOME", "~/pah-synth")).expanduser()
PY = os.environ.get("SIM_PYTHON", str(pathlib.Path("~/dream-env/bin/python").expanduser()))


def start(port: int, layout: str = "table", seed: int = 5, speed: float = 1.0, photoreal: float = 4.0, log: str | None = None) -> subprocess.Popen:
    env = dict(os.environ, MUJOCO_GL="egl", SIM_DR="plain", SIM_OVERHEAD="nov8", SIM_RENDER_SCALE="0.5")
    out = open(log or f"/tmp/hillclimb_sim_{port}.log", "w")
    p = subprocess.Popen([PY, "-m", "calib.station_sim", "--port", str(port), "--speed", str(speed), "--mode", "real",
                          "--layout", layout, "--seed", str(seed), "--photoreal", str(photoreal)],
                         cwd=str(SIM_HOME), env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    return p


def wait_ready(port: int, timeout: float = 90) -> bool:
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=2).read()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--layout", default="table")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    p = start(a.port, a.layout, a.seed, a.speed)
    ok = wait_ready(a.port)
    print(f"sim pid {p.pid} port {a.port} ready={ok}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
