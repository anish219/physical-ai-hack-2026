"""Right-jaw release evidence from the right arm server's /state (read-only).

A release is a hold (grip inside the cable band for JAW_HOLD_MIN_S) followed by an opening (grip over JAW_OPEN_MIN for
JAW_OPEN_MIN_S). It is remembered from the moment it happens, because the right arm keeps working: by the time the
camera confirms the cable on the table, the jaw may already be closing on the next one.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

from . import log, settings


class JawTracker:
    """Pure release logic over (t, grip) samples; no I/O, so it is easy to test."""

    def __init__(self):
        self.reset(0.0)

    def reset(self, since: float) -> None:
        self.since = since
        self.hold_start: float | None = None
        self.held_long = False
        self.open_start: float | None = None
        self.releases: list[float] = []
        self.last_grip: float | None = None
        self.last_t: float | None = None

    def add(self, t: float, grip: float | None) -> None:
        if grip is None or t < self.since:
            return
        lo, hi = float(settings.get("JAW_HOLD_LO")), float(settings.get("JAW_HOLD_HI"))
        open_min = float(settings.get("JAW_OPEN_MIN"))
        self.last_grip, self.last_t = grip, t
        if lo <= grip <= hi:
            if self.hold_start is None:
                self.hold_start = t
            if t - self.hold_start >= float(settings.get("JAW_HOLD_MIN_S")):
                self.held_long = True
            self.open_start = None
        elif grip >= open_min:
            self.hold_start = None
            if self.held_long:
                if self.open_start is None:
                    self.open_start = t
                if t - self.open_start >= float(settings.get("JAW_OPEN_MIN_S")):
                    self.releases.append(self.open_start)
                    self.held_long = False
                    self.open_start = None
        elif grip < lo:  # closed on nothing: the plug slipped out, so a later opening is not a release onto the table
            self.hold_start = None
            self.open_start = None
            self.held_long = False
        else:  # between the hold band and the open threshold: in motion, wait
            self.hold_start = None
            self.open_start = None

    def released_since(self, t: float) -> bool:
        return any(r >= t for r in self.releases)

    def is_open(self) -> bool:
        return self.last_grip is not None and self.last_grip >= float(settings.get("JAW_OPEN_MIN"))


class JawWatch:
    """Polls RIGHT_URL/state into a JawTracker. Never posts to the right arm."""

    def __init__(self, url: str | None = None):
        self.url = (url or str(settings.get("RIGHT_URL"))).rstrip("/") + "/state"
        self.tracker = JawTracker()
        self._lock = threading.Lock()
        self._last_ok = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gate-jaw", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        errors = 0
        while not self._stop.is_set():
            started = time.time()
            try:
                with urllib.request.urlopen(self.url, timeout=float(settings.get("HTTP_TIMEOUT_S"))) as r:
                    st = json.loads(r.read())
                grip = st.get("grip")
                with self._lock:
                    self.tracker.add(started, float(grip) if grip is not None else None)
                    self._last_ok = time.time()
                errors = 0
            except Exception:  # noqa: BLE001
                errors += 1
                if errors in (1, 10, 100):
                    log.event("jaw_state_failed", url=self.url, count=errors)
            self._stop.wait(max(0.0, float(settings.get("JAW_POLL_S")) - (time.time() - started)))

    def reset(self, since: float) -> None:
        with self._lock:
            self.tracker.reset(since)

    def fresh(self) -> bool:
        return time.time() - self._last_ok <= float(settings.get("STATE_MAX_AGE_S"))

    def released_since(self, t: float) -> bool:
        with self._lock:
            return self.tracker.released_since(t)

    def is_open(self) -> bool:
        with self._lock:
            return self.tracker.is_open()

    def close(self) -> None:
        self._stop.set()
