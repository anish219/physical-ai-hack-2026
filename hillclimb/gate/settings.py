"""Gate settings: environment first, then hillclimb/config.json ("gate" section or top level), then defaults.

Names follow hillclimb/INTERFACES.md. The API key is read from KEYS_FILE on demand and never logged.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent  # hillclimb/

DEFAULTS = {
    "LEFT_URL": "http://127.0.0.1:8691",
    "RIGHT_URL": "http://127.0.0.1:8692",
    "OVERHEAD_FROM": "left",  # which arm server serves /overhead.jpg: "left", "right" or a full URL
    "KEYS_FILE": "~/.config/hillclimb/keys.env",
    "LOG_DIR": "~/hillclimb_logs",
    "VLM_MODEL": "gpt-6.1-terra",
    # the strip crop, normalized (x0, y0, x1, y1); this one is the box the recorded tests used (280,120,220,170 of 640x360)
    "STRIP_BOX": [0.4375, 0.3333, 0.78125, 0.8056],
    "FAST_PERIOD_S": 2.0,  # one fast read every this many seconds
    "FAST_TIMEOUT_S": 5.0,  # a fast read slower than this counts as "no" (measured max about 5 s)
    "AGENTIC_BUDGET_S": 15.0,  # an agentic check slower than this counts as "no"
    "AGENTIC_MAX_TOOLS": 4,
    "AGENTIC_MIN_GAP_S": 1.5,  # the second agentic yes must look at a "now" frame at least this much newer
    "GATE_FRAME_MAX_AGE_S": 2.0,  # older overhead frames are stale (when the station clock is shared)
    "GATE_FRAME_POLL_S": 0.25,
    "GATE_BUFFER_S": 8.0,  # how much overhead history the agentic tools can ask for
    # right jaw, normalized grip from the arm server's /state (0 closed, 1 open); 0.04 to 0.06 is a USB plug in the jaw
    "JAW_HOLD_LO": 0.03,
    "JAW_HOLD_HI": 0.085,
    "JAW_HOLD_MIN_S": 0.5,
    "JAW_OPEN_MIN": 0.10,
    "JAW_OPEN_MIN_S": 0.3,
    "JAW_POLL_S": 0.2,
    "GATE_STATE_MAX_AGE_S": 2.0,  # jaw state older than this makes the gate unhealthy
    "SOCKET_VERIFIER_DIR": "~/usb-geometry-verifier-20261107/usb-geometry-verifier-v1/scripts",
    "SOCKET_PORTS": [0, 1, 2, 3],  # sockets occupied at go (the verifier's port ids)
    "SOCKET_REQUIRED": False,  # True: the gate waits for the verifier's proof; False: "unknown" does not block
    "SOCKET_BASELINE_S": 6.0,  # how long /gate/arm waits for the verifier's pre-go baseline
    "GATE_HTTP_TIMEOUT_S": 2.0,
}

_lock = threading.Lock()
_overrides: dict = {}


def _file_values() -> dict:
    path = ROOT / "config.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if isinstance(data.get("gate"), dict):
        return {**{k: v for k, v in data.items() if not isinstance(v, dict)}, **data["gate"]}
    return data


def get(name: str):
    with _lock:
        if name in _overrides:
            return _overrides[name]
    raw = os.environ.get(name)
    if raw is not None:
        default = DEFAULTS.get(name)
        try:
            if isinstance(default, bool):
                return raw.strip().lower() in ("1", "true", "yes")
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                return type(default)(raw)
            if isinstance(default, list):
                return json.loads(raw)
        except ValueError:
            return default
        return raw
    file_values = _file_values()
    if name in file_values:
        return file_values[name]
    return DEFAULTS.get(name)


def set_override(name: str, value) -> None:
    """Runtime changes (calibration, tests); never written back to config.json."""
    with _lock:
        _overrides[name] = value


def clear_overrides() -> None:
    with _lock:
        _overrides.clear()


def overhead_url() -> str:
    src = str(get("OVERHEAD_FROM"))
    base = get("LEFT_URL") if src == "left" else get("RIGHT_URL") if src == "right" else src
    return base.rstrip("/") + "/overhead.jpg"


def api_key() -> str | None:
    """OPENAI_API_KEY from the environment, else from KEYS_FILE. Never printed or logged."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    path = pathlib.Path(str(get("KEYS_FILE"))).expanduser()
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    except OSError:
        return None
    return None


def log_dir() -> pathlib.Path:
    path = pathlib.Path(str(get("LOG_DIR"))).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path
