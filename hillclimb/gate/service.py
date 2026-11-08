"""The gate service: a background checker and the /gate/* routes.

Mount: the hillclimb server calls `gate.service.handle(method, path, body)` for paths starting with /gate/ and sends
back (code, json). Standalone: `python -m hillclimb.gate.serve --port 8771`.

Loop (after /gate/arm): every FAST_PERIOD_S, one fast overhead read against the reference. On a yes, an agentic
overhead check. After one agentic yes, the next check is agentic again on a frame at least AGENTIC_MIN_GAP_S newer.
The gate opens when GateCore says so, and stays open (latched) until the next /gate/arm.
"""
from __future__ import annotations

import threading
import time
import traceback

from . import imaging, log, settings, vlm
from .core import GateCore
from .frames import FrameSource, HttpFrameSource
from .jaw import JawWatch
from .socket_check import SocketCheck


class Gate:
    def __init__(self, source: FrameSource | None = None, jaw: JawWatch | None = None, socket: SocketCheck | None = None):
        self.source = source if source is not None else HttpFrameSource()
        self.jaw = jaw if jaw is not None else JawWatch()
        self.socket = socket if socket is not None else SocketCheck()
        self.core = GateCore(min_gap_s=float(settings.get("AGENTIC_MIN_GAP_S")), socket_required=bool(settings.get("SOCKET_REQUIRED")))
        self.ref_jpeg: bytes | None = None
        self.ref_t: float | None = None
        self.strip_box = list(settings.get("STRIP_BOX"))
        self.last_reads: list[dict] = []
        self.last_fast_answer: dict | None = None
        self.last_fast_t = 0.0
        self.last_agentic: dict | None = None
        self.errors = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ control
    def arm(self, body: dict | None = None) -> dict:
        body = body or {}
        if body.get("strip_box"):
            self.strip_box = imaging.clamp_box(body["strip_box"])
        f = self.source.latest()
        if f is None or not self.source.fresh():
            return {"ok": False, "why": "no fresh overhead frame for the reference; gate stays shut"}
        with self._lock:
            self.ref_jpeg, self.ref_t = f.jpeg, f.t
            self.core.arm(time.time())
            self.last_reads, self.last_fast_answer, self.last_agentic = [], None, None
        self.jaw.reset(time.time())
        self.socket.arm()
        log.event("gate_armed", ref_t=f.t, strip_box=self.strip_box)
        self.start()
        return {"ok": True, "ref_t": f.t, "strip_box": self.strip_box}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gate-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.socket.stop()

    def calibrate(self, body: dict | None = None) -> dict:
        body = body or {}
        if body.get("box"):
            self.strip_box = imaging.clamp_box(body["box"])
            settings.set_override("STRIP_BOX", self.strip_box)
            log.event("gate_calibrated", mode="manual", box=self.strip_box)
            return {"ok": True, "mode": "manual", "box": self.strip_box}
        f = self.source.latest()
        if f is None:
            return {"ok": False, "why": "no overhead frame"}
        found = imaging.find_strip(f.jpeg)
        if found.get("ok"):
            self.strip_box = found["box"]
            settings.set_override("STRIP_BOX", self.strip_box)
        log.event("gate_calibrated", mode="auto", result=found)
        return {"mode": "auto", **found, "box_in_use": self.strip_box}

    # ------------------------------------------------------------------ loop
    def _healthy(self) -> bool:
        return self.source.fresh() and self.jaw.fresh()

    def _remember(self, row: dict) -> None:
        with self._lock:
            self.last_reads.append(row)
            del self.last_reads[:-10]
        log.event("gate_read", **row)

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.time()
            try:
                self._step()
            except Exception:  # noqa: BLE001  (the loop survives anything; the gate just stays shut)
                self.errors += 1
                log.event("gate_loop_error", trace=traceback.format_exc()[-600:])
            if self.core.open:
                self._stop.wait(1.0)
                continue
            self._stop.wait(max(0.05, float(settings.get("FAST_PERIOD_S")) - (time.time() - started)))

    def _step(self) -> None:
        if not self.core.armed or self.core.open or self.ref_jpeg is None:
            return
        healthy = self._healthy()
        now = self.source.latest()
        if now is None or not healthy:
            self.core.fast_read(time.time(), yes=False, ok=False)
            self._remember({"kind": "fast", "yes": False, "error": "stale or missing input", "ms": 0})
            return
        run_agentic = self.core.streak >= 1  # a first agentic yes: confirm with a second agentic check, no fast read
        if not run_agentic:
            comp = imaging.composite(self.ref_jpeg, now.jpeg, self.strip_box)
            res = vlm.fast_check(comp)
            ok = res["error"] is None
            with self._lock:
                self.last_fast_answer, self.last_fast_t = res.get("answer"), time.time()
            run_agentic = self.core.fast_read(time.time(), yes=res["yes"], ok=ok)
            self._remember({"kind": "fast", "yes": res["yes"], "ms": res["ms"], "error": res["error"], "answer": res["answer"], "now_t": now.t})
            if not run_agentic:
                return
        else:
            # the confirming check must look at a frame newer than the first yes by AGENTIC_MIN_GAP_S
            need = (self.core.last_yes_now_t or 0.0) + float(settings.get("AGENTIC_MIN_GAP_S"))
            f = self.source.at(need, wait_s=float(settings.get("AGENTIC_MIN_GAP_S")) + 1.0)
            if f is None or f.t < need - 0.05:
                return  # no newer frame yet; try again next tick
            now = f
        res = vlm.agentic_check(self.source, self.ref_jpeg, now.t, self.strip_box)
        with self._lock:
            self.last_agentic = res
        self.core.agentic_read(time.time(), now.t, yes=res["yes"], ok=res["error"] is None)
        self._remember({"kind": "agentic", "yes": res["yes"], "ms": res["ms"], "error": res["error"], "tools": res["tools"],
                        "asked": res["asked"], "evidence": res["evidence"], "now_t": now.t})
        opened = self.core.try_open(time.time(), self.socket.state(), self.jaw.released_since(self.core.armed_at or 0.0), self._healthy())
        if opened:
            log.event("gate_open", checks=self.core.checks(self.socket.state(), True, True), socket=self.socket.info())

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        with self._lock:
            ans = self.last_fast_answer or {}
            fast_known = bool(ans) and time.time() - self.last_fast_t < 10.0
            plugged_now = ans.get("plugged_now") if isinstance(ans.get("plugged_now"), int) else None
            healthy = self._healthy()
            jaw_released = self.jaw.released_since(self.core.armed_at or 0.0) if self.core.armed else False
            checks = self.core.checks(self.socket.state(), jaw_released, healthy)
            return {
                "known": bool(fast_known and healthy) or self.core.open,
                "open": self.core.open,
                "armed": self.core.armed,
                "latched_at": self.core.latched_at,
                # the main controller's verify_usb fields
                "unplugged_count": max(0, 4 - plugged_now) if plugged_now is not None else 0,
                "plugged_count": plugged_now if plugged_now is not None else 0,
                "gripper_released": self.jaw.is_open(),
                "cable_on_table": bool(self.core.open or (self.last_agentic or {}).get("yes")),
                "fully_inserted": False,
                "checks": checks,
                "socket": self.socket.info(),
                "strip_box": self.strip_box,
                "last_reads": list(self.last_reads),
                "loop_errors": self.errors,
            }


_gate: Gate | None = None
_gate_lock = threading.Lock()


def get_gate() -> Gate:
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = Gate()
        return _gate


def handle(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Routes /gate/*. Never raises: any failure is a 500 with the gate's state unchanged (still shut if it was)."""
    try:
        p = path.split("?", 1)[0].rstrip("/")
        g = get_gate()
        if method == "GET" and p == "/gate/state":
            return 200, g.state()
        if method == "POST" and p == "/gate/arm":
            r = g.arm(body or {})
            return (200 if r.get("ok") else 409), r
        if method == "POST" and p == "/gate/calibrate":
            return 200, g.calibrate(body or {})
        if method == "POST" and p == "/gate/stop":
            g.stop()
            return 200, {"ok": True}
        if method == "GET" and p == "/gate/health":
            return 200, {"overhead_fresh": g.source.fresh(), "jaw_fresh": g.jaw.fresh(), "socket": g.socket.info(), "loop_errors": g.errors,
                         "key_present": settings.api_key() is not None}
        return 404, {"error": f"no route {method} {p}"}
    except Exception as e:  # noqa: BLE001
        log.event("gate_handler_error", path=path, error=f"{type(e).__name__}: {e}")
        return 500, {"error": f"{type(e).__name__}: {str(e)[:200]}", "open": bool(_gate and _gate.core.open)}
