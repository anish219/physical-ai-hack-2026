"""The gate's event log: hillclimb.common.log's JsonlLog("gate") when it loads, else a local JSON-lines file.

A write that fails is dropped, never raised.
"""
from __future__ import annotations

import json
import threading
import time

from . import settings

_lock = threading.Lock()
_shared = None
try:
    from hillclimb.common.log import JsonlLog

    _shared = JsonlLog("gate", str(settings.log_dir()))
except Exception:  # noqa: BLE001  (the gate also runs on its own, without the common package)
    _shared = None


def event(name: str, **fields) -> None:
    if _shared is not None:
        _shared.write(name, **fields)
        return
    row = {"t": round(time.time(), 3), "event": name, **fields}
    try:
        line = json.dumps(row, default=str)
        with _lock, open(settings.log_dir() / "gate.jsonl", "a") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass
