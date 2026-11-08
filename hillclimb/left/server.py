"""The hillclimb HTTP server (stdlib, threaded, default port 8770).

Routes
  GET  /                       status page
  GET  /api/state              left controller + gate + health, one JSON
  GET  /cam/<overhead|left_wrist|right_wrist>.jpg   camera proxy (read-only)
  POST /left/arm               settings (JSON body), refused while a match round runs
  POST /left/go | /left/stop | /left/estop
  GET  /left/state             INTERFACES.md left state
  POST /left/stack_next        main-controller binding: one placement (drive must be "external")
  POST /left/rebuild           main-controller binding: recover and restack
  GET  /left/verify_tower      main-controller binding: {"known","block_count","collapsed","standing_free"}
  *    /gate/*                 the USB gate module's own routes (hillclimb.gate.http_handler), or 503 when absent

    cd <repo> && ~/dream-env/bin/python -m hillclimb.left.server [--port 8770]
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hillclimb.common.arm import ArmClient
from hillclimb.common.config import CFG
from hillclimb.left.controller import LeftController

HERE = pathlib.Path(__file__).resolve().parent
_gate_mod = None
_gate_err = ""


def gate_module():
    """hillclimb.gate, imported once and lazily; None when it is missing or broken."""
    global _gate_mod, _gate_err
    if _gate_mod is None and not _gate_err:
        try:
            _gate_mod = importlib.import_module("hillclimb.gate")
            if not hasattr(_gate_mod, "http_handler"):
                _gate_err, _gate_mod = "hillclimb.gate has no http_handler", None
        except Exception as e:  # noqa: BLE001
            _gate_err = f"{type(e).__name__}: {e}"[:200]
    return _gate_mod


def make_handler(ctl: LeftController, cams: dict[str, tuple[ArmClient, str]]):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass

        def _send(self, code: int, obj=None, ctype="application/json", raw: bytes | None = None):
            body = raw if raw is not None else json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            try:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:  # noqa: BLE001
                return {}

        def _gate(self, method: str, body):
            g = gate_module()
            if g is None:
                return self._send(503, {"known": False, "open": False, "error": f"gate unavailable: {_gate_err}"})
            try:
                code, obj = g.http_handler(method, self.path.split("?")[0], body)
                return self._send(code, obj)
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"known": False, "open": False, "error": repr(e)[:200]})

        def do_GET(self):  # noqa: N802
            p = self.path.split("?")[0]
            try:
                if p == "/":
                    return self._send(200, raw=(HERE / "status.html").read_bytes(), ctype="text/html; charset=utf-8")
                if p == "/api/state":
                    snap = ctl.snapshot()
                    return self._send(200, {"left": snap, "gate": ctl.gate_cache, "gate_module": _gate_err or "ok", "t": time.time()})
                if p.startswith("/cam/") and p.endswith(".jpg"):
                    name = p[5:-4]
                    if name not in cams:
                        return self._send(404, {"error": "no such camera"})
                    arm, which = cams[name]
                    f = arm.frame(which)
                    if f is None:
                        return self._send(503, {"error": "camera unavailable"})
                    return self._send(200, raw=f[0], ctype="image/jpeg")
                if p == "/left/state":
                    return self._send(200, ctl.snapshot())
                if p == "/left/verify_tower":
                    return self._send(200, ctl.verify_tower())
                if p.startswith("/gate/"):
                    return self._gate("GET", None)
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001  a bad request never takes the server down
                return self._send(500, {"error": repr(e)[:200]})

        def do_POST(self):  # noqa: N802
            p = self.path.split("?")[0]
            body = self._body()
            try:
                if p == "/left/arm":
                    ok, why = ctl.arm_round(body)
                    return self._send(200 if ok else 409, {"ok": ok, "why": why})
                if p == "/left/go":
                    ok, why = ctl.go()
                    return self._send(200 if ok else 409, {"ok": ok, "why": why})
                if p == "/left/stop":
                    ctl.stop_round()
                    return self._send(200, {"ok": True})
                if p == "/left/estop":
                    ctl.emergency_stop()
                    return self._send(200, {"ok": True})
                if p == "/left/stack_next":
                    return self._send(200, ctl.stack_next())
                if p == "/left/rebuild":
                    return self._send(200, ctl.rebuild())
                if p.startswith("/gate/"):
                    return self._gate("POST", body)
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"ok": False, "error": repr(e)[:200]})

    return H


def build(port: int | None = None, ctl: LeftController | None = None) -> tuple[ThreadingHTTPServer, LeftController]:
    ctl = ctl or LeftController()
    right = ArmClient(CFG["RIGHT_URL"])
    over = ctl.overhead
    cams = {"overhead": (over, "overhead"), "left_wrist": (ctl.arm, "wrist"), "right_wrist": (right, "wrist")}
    srv = ThreadingHTTPServer(("0.0.0.0", port or CFG["HILLCLIMB_PORT"]), make_handler(ctl, cams))
    srv.daemon_threads = True
    return srv, ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=CFG["HILLCLIMB_PORT"])
    a = ap.parse_args()
    srv, ctl = build(a.port)
    ctl.start()
    g = gate_module()
    if g is not None and hasattr(g, "start"):
        try:
            g.start()
        except Exception as e:  # noqa: BLE001
            print("gate start failed:", repr(e)[:200])
    print(f"hillclimb on :{a.port}  left={CFG['LEFT_URL']} right={CFG['RIGHT_URL']} gate={'ok' if g else _gate_err}", flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        ctl.shutdown()
        srv.shutdown()


if __name__ == "__main__":
    main()
