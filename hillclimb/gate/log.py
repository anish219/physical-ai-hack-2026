"""JSON-lines event log for the gate. A write that fails is dropped, never raised."""
from __future__ import annotations

import json
import threading
import time

from . import settings

_lock = threading.Lock()


def event(kind: str, **fields) -> None:
    row = {"t": round(time.time(), 3), "kind": kind, **fields}
    try:
        line = json.dumps(row, default=str)
        with _lock, open(settings.log_dir() / "gate.jsonl", "a") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001  (the log must never stop the gate)
        pass
