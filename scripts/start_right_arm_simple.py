"""Launch the headless right-arm worker.

From a shell, at the checkout root:

    PYTHONPATH=src .venv/bin/python scripts/start_right_arm_simple.py \
        --config configs/right_arm_controller.json \
        --run-id UNIQUE_MATCH_ID --go-time SHARED_UNIX_SECONDS \
        --run-dir /tmp/relay/UNIQUE_MATCH_ID

The station Script field `scripts/start_right_arm_simple.py` calls `start` on START and
`stop` when START is pressed again. `stop` signals the worker so it cancels
without homing or changing arm power.
"""

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_lock = threading.Lock()
_worker = None


def launch_args(run_id, go_time, run_dir):
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_right_arm_simple.py"),
        "--config",
        "configs/right_arm_controller.json",
        "--run-id",
        run_id,
        "--go-time",
        str(go_time),
        "--run-dir",
        str(run_dir),
    ]


def start(context=None):
    global _worker
    stop()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = Path("/tmp/relay") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH"))))
    proc = subprocess.Popen(launch_args(run_id, time.time(), run_dir), cwd=ROOT, env=env, start_new_session=True)
    with _lock:
        _worker = proc
    return proc


def stop(context=None):
    global _worker
    with _lock:
        proc = _worker
    if proc is None or proc.poll() is not None:
        return None
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        return proc
    return proc


def main():
    import run_right_arm_simple

    run_right_arm_simple.main()


if __name__ == "__main__":
    main()
