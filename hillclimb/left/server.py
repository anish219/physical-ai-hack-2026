"""The hillclimb HTTP server (stdlib, threaded, default port 8770).

Routes
  GET  /                       status page
  GET  /api/state              left controller + gate + health, one JSON
  GET  /cam/<overhead|left_wrist|right_wrist>.jpg   camera proxy (read-only)
  POST /left/arm               settings (JSON body), refused while a match round runs
  POST /left/go | /left/stop | /left/estop | /left/restart (stop, arm with {"settings": {...}} or the current ones, go)
  GET  /api/right              read-only right arm view (newest /tmp/relay run, right arm server state)
  GET  /api/timeline           last key events from the left and gate logs, newest first
  GET  /left/state             INTERFACES.md left state
  POST /left/stack_next        main-controller binding: one placement (drive must be "external")
  POST /left/rebuild           main-controller binding: recover and restack
  GET  /left/verify_tower      main-controller binding: {"known","block_count","collapsed","standing_free"}
  *    /gate/*                 the USB gate module's own routes (hillclimb.gate.http_handler), or 503 when absent

    cd <repo> && ~/dream-env/bin/python -m hillclimb.left.server [--port 8770]
"""
from __future__ import annotations

import argparse
import dataclasses
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


def _tail(path: pathlib.Path, n: int) -> list[dict]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200000))
            lines = f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def timeline(n: int = 15) -> list[dict]:
    """The last key events from the left and gate logs, newest first, in plain words."""
    log_dir = pathlib.Path(str(CFG.get("LOG_DIR", "."))).expanduser()
    rows = []
    for r in _tail(log_dir / "left.jsonl", 300):
        ev = r.get("event")
        if ev == "state":
            rows.append((r.get("t", 0), f"left: {r.get('frm')} to {r.get('to')} ({r.get('why', '')})"))
        elif ev == "step":
            what = r.get("what") or r.get("kind") or "step"
            rows.append((r.get("t", 0), f"left {what}: {'ok' if r.get('ok') else 'failed'}"
                         f"{', placed ' + str(r.get('placed_id')) if r.get('placed_id') else ''}"
                         f"{', tower ' + str(r.get('tower_count')) if r.get('tower_count') is not None else ''}"
                         f"{': ' + str(r.get('why'))[:120] if r.get('why') else ''}"))
        elif ev in ("arm", "resume", "tick_error", "stage2_passed"):
            rows.append((r.get("t", 0), {"arm": "left: round armed", "resume": "left: server restarted, round resumed",
                                         "tick_error": f"left error: {str(r.get('error'))[:120]}",
                                         "stage2_passed": "left: stage 2 passed"}[ev]))
    last_yes = None
    for r in _tail(log_dir / "gate.jsonl", 600):
        ev = r.get("event")
        if ev == "gate_read":
            yes = bool(r.get("yes"))
            if yes != last_yes:
                a = r.get("answer") or {}
                rows.append((r.get("t", 0), f"USB {r.get('kind')} read now says {'YES' if yes else 'no'}"
                             f" (plugged {a.get('plugged_ref')} to {a.get('plugged_now')}, gripper holding {a.get('gripper_holding_cable_now')})"))
                last_yes = yes
        elif ev in ("gate_armed", "gate_open", "gate_opened", "gate_latched", "gate_auto_disarmed", "socket_passed"):
            rows.append((r.get("t", 0), {"gate_armed": "USB gate armed: reference photo taken",
                                         "gate_auto_disarmed": "USB gate stopped reading (round over)",
                                         "socket_passed": "socket check: a hub port emptied"}.get(ev, "USB GATE OPEN: cable unplugged and on the table")))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [{"t": t, "text": text} for t, text in rows[:n]]


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

        def _go(self, body):
            ok, why = ctl.go()
            gate = None
            if ok and not (body or {}).get("keep_gate"):  # one Go: the gate takes its reference and resets its latch
                g = gate_module()
                try:
                    gate = g.http_handler("POST", "/gate/arm", {})[1] if g is not None else {"error": "gate unavailable"}
                except Exception as e:  # noqa: BLE001
                    gate = {"error": repr(e)[:200]}
            return ok, why, gate

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
                if p == "/api/right":
                    from hillclimb.left import right_status
                    return self._send(200, right_status.status())
                if p == "/api/timeline":
                    return self._send(200, {"events": timeline()})
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
                    ok, why, gate = self._go(body)
                    return self._send(200 if ok else 409, {"ok": ok, "why": why, "gate": gate})
                if p == "/left/restart":  # stop, then arm with these (or the current) settings, then go: one call
                    ctl.stop_round("restart")
                    t_end = time.time() + 10
                    while ctl.state not in ("END", "IDLE") and time.time() < t_end:
                        time.sleep(0.1)
                    cfg = (body or {}).get("settings") or dataclasses.asdict(ctl.settings)
                    ok_a, why_a = ctl.arm_round(cfg)
                    if not ok_a:
                        return self._send(409, {"ok": False, "stop": "done", "arm": why_a})
                    ok, why, gate = self._go(body)
                    return self._send(200 if ok else 409, {"ok": ok, "stop": "done", "arm": why_a, "go": why, "gate": gate})
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
