"""Settings for every hillclimb part: defaults, then hillclimb/config.json, then the environment."""
from __future__ import annotations

import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULTS = {
    "LEFT_URL": "http://127.0.0.1:8691",
    "RIGHT_URL": "http://127.0.0.1:8692",
    "OVERHEAD_FROM": "left",  # which arm server serves the overhead camera
    "HILLCLIMB_PORT": 8770,
    "KEYS_FILE": "~/.config/hillclimb/keys.env",
    "LOG_DIR": "~/hillclimb_logs",
    "STATE_FILE": "~/hillclimb_logs/left_state.json",
    "GATE_URL": "",  # only when the gate runs as its own process; empty means the in-process gate module
    "VLM_MODEL": "gpt-6.1-terra",
    "VLM_TIMEOUT_S": 4.0,
    "ARM_TIMEOUT_S": 1.5,
    "FRAME_MAX_AGE_S": 0.5,
}
_INT = {"HILLCLIMB_PORT"}
_FLOAT = {"VLM_TIMEOUT_S", "ARM_TIMEOUT_S", "FRAME_MAX_AGE_S"}


def load() -> dict:
    cfg = dict(DEFAULTS)
    f = ROOT / "config.json"
    try:
        if f.exists():
            cfg.update(json.loads(f.read_text()))
    except Exception:  # noqa: BLE001  a broken config file falls back to defaults
        pass
    for k in DEFAULTS:
        if os.environ.get(k) not in (None, ""):
            cfg[k] = os.environ[k]
    for k in _INT:
        cfg[k] = int(cfg[k])
    for k in _FLOAT:
        cfg[k] = float(cfg[k])
    for k in ("KEYS_FILE", "LOG_DIR", "STATE_FILE"):
        cfg[k] = str(pathlib.Path(str(cfg[k])).expanduser())
    return cfg


CFG = load()
