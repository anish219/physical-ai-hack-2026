"""JSON lines logs, one file per part. Writing never raises."""
from __future__ import annotations

import json
import pathlib
import threading
import time

from hillclimb.common.config import CFG


class JsonlLog:
    def __init__(self, name: str, log_dir: str | None = None):
        self.path = pathlib.Path(log_dir or CFG["LOG_DIR"]) / f"{name}.jsonl"
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass

    def write(self, kind: str, **fields) -> None:
        row = {"t": round(time.time(), 3), "kind": kind, **fields}
        try:
            line = json.dumps(row, default=str)
            with self._lock, self.path.open("a") as f:
                f.write(line + "\n")
        except Exception:  # noqa: BLE001  logging must never take the controller down
            pass
