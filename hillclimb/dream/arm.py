"""A small client for one arm server (the team's 8691 left server, or a station sim on its own port).

Only GET /state, GET /overhead.jpg, GET /wrist.jpg, POST /target, POST /move and POST /stop are used. Every call has a
timeout; callers get an exception they can catch, never a hang.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

REAL_RIGHT_PORTS = (8692,)
FORBIDDEN_PORTS = (8081, 8084, 8697, 18081, 18084)


class ArmClient:
    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        port = int(self.url.rsplit(":", 1)[-1].split("/")[0]) if ":" in self.url.split("//", 1)[-1] else 80
        if port in REAL_RIGHT_PORTS + FORBIDDEN_PORTS:
            raise ValueError(f"refusing to drive {self.url}: that port is not the left arm")

    def _req(self, path: str, body: dict | None = None, timeout: float | None = None) -> bytes:
        rq = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={} if body is None else {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(rq, timeout=timeout or self.timeout) as r:
            return r.read()

    def state(self) -> dict:
        return json.loads(self._req("/state"))

    def frame(self, name: str) -> bytes:
        return self._req(f"/{name}.jpg")

    def target(self, q, grip: float) -> None:
        """Direct target, no speed walk (what Dream's rows use)."""
        self._req("/target", {"q": [float(v) for v in q], "grip": float(grip)})

    def move(self, q, grip: float, rate: float = 0.6) -> None:
        """A walked move at `rate` (rad/s cap), for the retreat and start-pose moves."""
        self._req("/move", {"q": [float(v) for v in q], "grip": float(grip), "rate": float(rate)})

    def stop(self) -> None:
        """Hold where the arm is."""
        try:
            self._req("/stop", {})
        except (urllib.error.URLError, OSError, ValueError):
            pass
