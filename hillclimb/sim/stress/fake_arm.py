"""A fake arm server with switchable faults (stdlib HTTP, numpy/cv2 only for the JPEGs). Never imports hillclimb.

GET /state, /overhead.jpg, /wrist.jpg; POST /move, /target, /stop. Every request is counted in .calls.
mode keys:
  state:    ok | q_null | disabled | http500 | notjson | slow
  move:     ok | http400 | slow
  overhead: ok | frozen | frozen_nostamp | missing | garbage | slow | flaky (every other request 404)
  grip:     float, the jaw value /state reports (for the gate's JawWatch on the right arm)
  slow_s:   how long "slow" sleeps
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


def jpeg(i: int) -> bytes:
    img = np.full((360, 640, 3), 200, np.uint8)
    cv2.putText(img, f"frame {i}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class FakeArm:
    def __init__(self, name: str = "arm"):
        self.name = name
        self.mode = {"state": "ok", "move": "ok", "overhead": "ok", "grip": 0.8, "slow_s": 3.0}
        self.calls: collections.Counter = collections.Counter()
        self.log: list[tuple[float, str]] = []
        self.i = 0
        self._frozen: tuple[bytes, float] | None = None
        self._lock = threading.Lock()
        arm = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def _send(self, code, body: bytes, ctype="application/json", headers=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                p = self.path.split("?")[0]
                arm._count("GET " + p)
                m = arm.mode
                if p == "/state":
                    st = m["state"]
                    if st == "slow":
                        time.sleep(m["slow_s"])
                    if st == "http500":
                        return self._send(500, b'{"error":"boom"}')
                    if st == "notjson":
                        return self._send(200, b"<html>not json")
                    q = None if st == "q_null" else [0.0, -1.84, 1.76, -1.48, 0.0, 0.0]
                    body = {"q": q, "goal": q, "grip": m["grip"], "goal_grip": m["grip"], "enabled": st != "disabled",
                            "t": time.time()}
                    return self._send(200, json.dumps(body).encode())
                if p in ("/overhead.jpg", "/wrist.jpg"):
                    ov = m["overhead"] if p == "/overhead.jpg" else "ok"
                    if ov == "slow":
                        time.sleep(m["slow_s"])
                    if ov == "missing" or (ov == "flaky" and arm.calls["GET " + p] % 2 == 0):
                        return self._send(404, b'{"error":"no camera"}')
                    if ov == "garbage":
                        return self._send(200, b"\xff\xd8" + os.urandom(3000), "image/jpeg", {"X-Stamp": f"{time.time():.3f}"})
                    if ov in ("frozen", "frozen_nostamp"):
                        with arm._lock:
                            if arm._frozen is None:
                                arm._frozen = (jpeg(arm.i), time.time() - float(m.get("frozen_age_s", 0.0)))
                        data, stamp = arm._frozen
                        hdr = {"X-Stamp": f"{stamp:.3f}"} if ov == "frozen" else {}
                        return self._send(200, data, "image/jpeg", hdr)
                    with arm._lock:
                        arm._frozen = None
                        arm.i += 1
                        i = arm.i
                    return self._send(200, jpeg(i), "image/jpeg", {"X-Stamp": f"{time.time():.3f}"})
                return self._send(404, b"{}")

            def do_POST(self):  # noqa: N802
                p = self.path.split("?")[0]
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                arm._count("POST " + p)
                if p in ("/move", "/target"):
                    if arm.mode["move"] == "slow":
                        time.sleep(arm.mode["slow_s"])
                    if arm.mode["move"] == "http400":
                        return self._send(400, b'{"error":"bad target"}')
                return self._send(200, b'{"ok":true}')

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def _count(self, what: str) -> None:
        with self._lock:
            self.calls[what] += 1
            self.log.append((time.time(), what))

    def motion_calls(self, since: float = 0.0) -> int:
        with self._lock:
            return sum(1 for t, w in self.log if t >= since and w in ("POST /move", "POST /target"))

    def count(self, what: str, since: float = 0.0) -> int:
        with self._lock:
            return sum(1 for t, w in self.log if t >= since and w == what)
