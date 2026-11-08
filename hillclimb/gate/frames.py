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
        self.urls = [url] if url else settings.overhead_urls()  # fallbacks: the other arm server's copy of the same camera
        self._stamps: dict[str, float] = {}
        self._hashes: dict[str, int] = {}
        self.poll_s = float(poll_s or settings.get("GATE_FRAME_POLL_S"))
        self.buffer_s = float(buffer_s or settings.get("GATE_BUFFER_S"))
        self._frames: list[Frame] = []
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._last_stamp: float | None = None
        self._last_ok = 0.0
        self._clock_shared: bool | None = None
        self._hist: list[Frame] = []  # slow history (one per HISTORY_EVERY_S) for a late-start reference
        self._hist_dir = settings.log_dir() / "overhead_history"
        self._load_history()
        self._thread = threading.Thread(target=self._run, name="gate-overhead", daemon=True)
        self._thread.start()

    def _fetch_one(self, url: str) -> tuple[bytes, float | None] | None:
        try:
            with urllib.request.urlopen(url, timeout=float(settings.get("GATE_HTTP_TIMEOUT_S"))) as r:
                data = r.read()
                stamp = r.headers.get("X-Stamp")
            if not data or data[:2] != b"\xff\xd8":
                return None
            return data, (float(stamp) if stamp else None)
        except Exception:  # noqa: BLE001  (a failed poll is just a missing frame)
            return None

    def _fetch(self) -> tuple[bytes, float | None] | None:
        """The first source with a frame newer than that source's own last one; a frozen source falls through.
        Stamps are tracked per source and the clock check is never reset, so switching sources can't make a stale
        frame look fresh."""
        for u in self.urls:
            got = self._fetch_one(u)
            if got is None:
                continue
            stamp = got[1]
            prev = self._stamps.get(u)
            if stamp is not None and prev is not None and stamp <= prev:
                continue  # frozen here: try the next server
            if stamp is not None:
                self._stamps[u] = stamp
            else:  # no X-Stamp: a byte-identical JPEG is a frozen camera
                h = hash(got[0])
                if self._hashes.get(u) == h:
                    continue
                self._hashes[u] = h
            if u != self.url:
                self.url = u
                log.event("overhead_source", url=u)
            return got
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
                if stamp is not None and abs(started - stamp) < 30.0:
                    self._clock_shared = True  # same machine clock; once seen, a stale stamp always fails the age check
                elif stamp is not None and self._clock_shared is None:
                    self._clock_shared = False
                if True:  # _fetch already refused a stamp that did not move on for its source
                    t = stamp if (stamp is not None and self._clock_shared) else started
                    with self._lock:
                        self._frames.append(Frame(t, data, stamp))
                        cutoff = t - self.buffer_s
                        while self._frames and self._frames[0].t < cutoff:
                            self._frames.pop(0)
                        self._lock.notify_all()
                    self._last_stamp = stamp
                    self._last_ok = time.time()
                    self._keep_history(t, data, stamp)
            self._stop.wait(max(0.0, self.poll_s - (time.time() - started)))

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    # ------------------------------------------------------------------ slow history (survives a server restart)
    def _load_history(self) -> None:
        try:
            self._hist_dir.mkdir(parents=True, exist_ok=True)
            keep = time.time() - float(settings.get("HISTORY_S"))
            for f in sorted(self._hist_dir.glob("*.jpg")):
                try:
                    t = float(f.stem)
                except ValueError:
                    continue
                if t < keep:
                    f.unlink(missing_ok=True)
                else:
                    self._hist.append(Frame(t, b"", t))  # bytes read lazily from disk
        except OSError:
            pass

    def _keep_history(self, t: float, data: bytes, stamp: float | None) -> None:
        if self._hist and t - self._hist[-1].t < float(settings.get("HISTORY_EVERY_S")):
            return
        try:
            (self._hist_dir / f"{t:.2f}.jpg").write_bytes(data)
        except OSError:
            pass
        with self._lock:
            self._hist.append(Frame(t, b"", t))
            keep = t - float(settings.get("HISTORY_S"))
            while self._hist and self._hist[0].t < keep:
                old = self._hist.pop(0)
                try:
                    (self._hist_dir / f"{old.t:.2f}.jpg").unlink(missing_ok=True)
                except OSError:
                    pass

    def before(self, t: float, max_gap_s: float = 10.0) -> Frame | None:
        """The newest history frame taken at or before time t, and no more than max_gap_s before it; None if the
        history does not reach back that far (the server started later)."""
        with self._lock:
            cands = [f for f in self._hist if f.t <= t]
        if not cands or t - cands[-1].t > max_gap_s:
            return None
        f = cands[-1]
        try:
            data = (self._hist_dir / f"{f.t:.2f}.jpg").read_bytes()
        except OSError:
            return None
        return Frame(f.t, data, f.stamp) if data[:2] == b"\xff\xd8" else None

    def fresh(self) -> bool:
        f = self.latest()
        if f is None:
            return False
        max_age = float(settings.get("GATE_FRAME_MAX_AGE_S"))
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
