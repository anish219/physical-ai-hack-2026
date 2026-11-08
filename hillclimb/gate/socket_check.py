"""Optional adapter for the team's geometry socket verifier (~/usb-geometry-verifier-20261107, used read-only).

The verifier needs same-host loopback camera URLs, a pre-go baseline of occupied sockets and the right wrist camera.
If it cannot be imported, cannot see its calibrated geometry, or never gets a baseline, this check reports "unknown".
By default "unknown" does not block the gate (SOCKET_REQUIRED=False); "yes" is the verifier's latched proof that an
initially occupied socket is now empty. The verifier never reports a plain "no", so this check cannot veto alone.
"""
from __future__ import annotations

import pathlib
import sys
import threading
import time
import uuid

from . import log, settings


class SocketCheck:
    def __init__(self):
        self.available = False
        self.status = "unknown"
        self.detail: dict = {}
        self._verifier = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        try:
            d = pathlib.Path(str(settings.get("SOCKET_VERIFIER_DIR"))).expanduser()
            if not d.is_dir():
                raise ImportError(f"{d} missing")
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            import usb_camera_shadow  # noqa: F401
            import usb_verifier_runtime  # noqa: F401
            self._shadow = usb_camera_shadow
            self._runtime = usb_verifier_runtime
            self.available = True
        except Exception as e:  # noqa: BLE001
            self.detail = {"why": f"verifier not loaded: {type(e).__name__}: {str(e)[:120]}"}

    def arm(self) -> None:
        """Start a fresh verifier for this round: collect the pre-go baseline, then arm at go."""
        self.stop()
        with self._lock:
            self.status, self.detail = "unknown", {}
        if not self.available:
            self.detail = {"why": "verifier not available"}
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gate-socket", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        run_id = f"hillclimb-{uuid.uuid4().hex[:8]}"
        scene_url = settings.overhead_url()
        wrist_url = str(settings.get("RIGHT_URL")).rstrip("/") + "/wrist.jpg"
        try:
            verifier = self._runtime.UsbGeometryVerifier(run_id, set(settings.get("SOCKET_PORTS")))
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self.detail = {"why": f"verifier init failed: {type(e).__name__}: {str(e)[:120]}"}
            return
        armed = False
        t_start = time.monotonic()
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                images, stamps, ages, errors = self._shadow.fetch_pair(scene_url, wrist_url, run_id)
                res = verifier.observe(rgb_scene=images.get("scene"), rgb_wrist=images.get("wrist"), stamps=stamps, now=time.monotonic())
                if not armed:
                    if verifier.gate.baseline_ready(time.monotonic()):
                        verifier.gate.arm(time.monotonic())
                        armed = True
                        log.event("socket_armed", run_id=run_id)
                    elif time.monotonic() - t_start > float(settings.get("SOCKET_BASELINE_S")):
                        with self._lock:
                            self.detail = {"why": "no pre-go baseline (camera identity or geometry may not match the calibration)",
                                           "reason": res.get("reason"), "camera_errors": errors}
                with self._lock:
                    if res.get("stage_passed"):
                        if self.status != "yes":
                            log.event("socket_passed", proof=res.get("proof"))
                        self.status = "yes"
                    self.detail.update({"reason": res.get("reason"), "armed": armed, "ports": (res.get("detection") or {}).get("ports")})
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self.detail["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            self._stop.wait(max(0.0, 0.2 - (time.monotonic() - started)))

    def state(self) -> str:
        with self._lock:
            return self.status

    def info(self) -> dict:
        with self._lock:
            return {"available": self.available, "status": self.status, **self.detail}

    def stop(self) -> None:
        self._stop.set()
