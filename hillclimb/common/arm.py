"""Client for one arm server (the team's armserver or the station sim): every call has a timeout and returns None
or False on failure instead of raising. Motion calls first ask an optional guard, so a locked controller can never
send a command by mistake."""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Callable

from hillclimb.common.config import CFG


class ArmClient:
    def __init__(self, base: str, timeout: float | None = None, guard: Callable[[], bool] | None = None):
        self.base = base.rstrip("/")
        self.timeout = timeout or CFG["ARM_TIMEOUT_S"]
        self.guard = guard  # returns True when motion is allowed
        self.last_error: str | None = None

    def _req(self, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read(), dict(r.headers)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{path}: {type(e).__name__}: {e}"[:200]
            return None

    def state(self) -> dict | None:
        got = self._req("/state")
        if got is None:
            return None
        try:
            return json.loads(got[0])
        except Exception:  # noqa: BLE001
            self.last_error = "/state: not JSON"
            return None

    def frame(self, name: str) -> tuple[bytes, float] | None:
        """(jpeg bytes, age in seconds) for 'overhead' or 'wrist'. Age uses the server's X-Stamp when it sends one."""
        got = self._req(f"/{name}.jpg")
        if got is None:
            return None
        body, headers = got
        stamp = headers.get("X-Stamp") or headers.get("x-stamp")
        try:
            age = max(0.0, time.time() - float(stamp)) if stamp else 0.0
        except ValueError:
            age = 0.0
        return body, age

    def fresh_frame(self, name: str, max_age: float | None = None) -> bytes | None:
        f = self.frame(name)
        if f is None or f[1] > (max_age if max_age is not None else CFG["FRAME_MAX_AGE_S"]):
            return None
        return f[0]

    def _allowed(self) -> bool:
        try:
            return self.guard is None or bool(self.guard())
        except Exception:  # noqa: BLE001
            return False

    def target(self, q: list[float], grip: float) -> bool:
        if not self._allowed():
            self.last_error = "refused: motion locked"
            return False
        return self._req("/target", {"q": list(q), "grip": grip}) is not None

    def move(self, q: list[float], grip: float, rate: float = 0.3) -> bool:
        if not self._allowed():
            self.last_error = "refused: motion locked"
            return False
        return self._req("/move", {"q": list(q), "grip": grip, "rate": rate}) is not None

    def stop(self) -> bool:
        """Hold where it is. Always allowed: stopping is never locked."""
        return self._req("/stop", {}) is not None
