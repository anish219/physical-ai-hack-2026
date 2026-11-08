"""Overhead frame sources for the gate.

HttpFrameSource polls an arm server's /overhead.jpg into a short ring buffer, so the agentic check can ask for frames
from a few seconds back, or wait up to 2 s for a newer one. A frame counts only if its capture stamp moved on (a frozen
camera is stale); when the station clock is shared it must also be younger than FRAME_MAX_AGE_S.
"""
from __future__ import annotations

import bisect
import threading
import time
import urllib.request

from . import log, settings


class Frame:
    __slots__ = ("t", "jpeg", "stamp")

    def __init__(self, t: float, jpeg: bytes, stamp: float | None):
        self.t, self.jpeg, self.stamp = t, jpeg, stamp


class FrameSource:
    """Interface: latest(), at(t, wait_s) and fresh()."""

    def latest(self) -> Frame | None:
        raise NotImplementedError

    def at(self, t: float, wait_s: float = 0.0) -> Frame | None:
        raise NotImplementedError

    def fresh(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass


class HttpFrameSource(FrameSource):
    def __init__(self, url: str | None = None, poll_s: float | None = None, buffer_s: float | None = None):
        self.url = url or settings.overhead_url()
        self.poll_s = float(poll_s or settings.get("FRAME_POLL_S"))
        self.buffer_s = float(buffer_s or settings.get("BUFFER_S"))
        self._frames: list[Frame] = []
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._last_stamp: float | None = None
        self._last_ok = 0.0
        self._clock_shared: bool | None = None
        self._thread = threading.Thread(target=self._run, name="gate-overhead", daemon=True)
        self._thread.start()

    def _fetch(self) -> tuple[bytes, float | None] | None:
        try:
            with urllib.request.urlopen(self.url, timeout=float(settings.get("HTTP_TIMEOUT_S"))) as r:
                data = r.read()
                stamp = r.headers.get("X-Stamp")
            if not data or data[:2] != b"\xff\xd8":
                return None
            return data, (float(stamp) if stamp else None)
        except Exception:  # noqa: BLE001  (a failed poll is just a missing frame)
            return None

    def _run(self) -> None:
        errors = 0
        while not self._stop.is_set():
            started = time.time()
            got = self._fetch()
            if got is None:
                errors += 1
                if errors in (1, 10, 100):
                    log.event("overhead_fetch_failed", url=self.url, count=errors)
            else:
                errors = 0
                data, stamp = got
                if stamp is not None and self._clock_shared is None:
                    self._clock_shared = abs(started - stamp) < 30.0  # same machine clock, or close enough
                if stamp is None or self._last_stamp is None or stamp > self._last_stamp:
                    t = stamp if (stamp is not None and self._clock_shared) else started
                    with self._lock:
                        self._frames.append(Frame(t, data, stamp))
                        cutoff = t - self.buffer_s
                        while self._frames and self._frames[0].t < cutoff:
                            self._frames.pop(0)
                        self._lock.notify_all()
                    self._last_stamp = stamp
                    self._last_ok = time.time()
            self._stop.wait(max(0.0, self.poll_s - (time.time() - started)))

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def fresh(self) -> bool:
        f = self.latest()
        if f is None:
            return False
        max_age = float(settings.get("FRAME_MAX_AGE_S"))
        if time.time() - self._last_ok > max_age:  # no new frame arrived lately: frozen camera or dead server
            return False
        if self._clock_shared and f.stamp is not None:
            return time.time() - f.stamp <= max_age
        return True

    def at(self, t: float, wait_s: float = 0.0) -> Frame | None:
        """The buffered frame nearest to time t; for a future t, waits up to wait_s for it to arrive."""
        deadline = time.time() + max(0.0, wait_s)
        with self._lock:
            while (not self._frames or self._frames[-1].t < t - 0.15) and time.time() < deadline:
                self._lock.wait(timeout=max(0.01, min(0.2, deadline - time.time())))
            if not self._frames:
                return None
            times = [f.t for f in self._frames]
            i = bisect.bisect_left(times, t)
            cands = [self._frames[j] for j in (i - 1, i) if 0 <= j < len(self._frames)]
            best = min(cands, key=lambda f: abs(f.t - t))
            return best if abs(best.t - t) <= 1.0 else None

    def close(self) -> None:
        self._stop.set()


class ListFrameSource(FrameSource):
    """Frames supplied by a test or replay: a callable (t) -> jpeg bytes or None, plus a fixed 'now'."""

    def __init__(self, get_jpeg, now_t: float, start_t: float = 0.0, end_t: float = 1e9):
        self.get_jpeg, self.now_t, self.start_t, self.end_t = get_jpeg, now_t, start_t, end_t

    def latest(self) -> Frame | None:
        data = self.get_jpeg(self.now_t)
        return Frame(self.now_t, data, self.now_t) if data else None

    def at(self, t: float, wait_s: float = 0.0) -> Frame | None:
        t = max(self.start_t, min(self.end_t, t))
        data = self.get_jpeg(t)
        return Frame(t, data, t) if data else None

    def fresh(self) -> bool:
        return self.latest() is not None
