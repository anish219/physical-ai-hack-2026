"""Standalone gate server for testing: python -m hillclimb.gate.serve --port 8771

Routes: GET /gate/state, GET /gate/health, POST /gate/arm, POST /gate/calibrate, POST /gate/stop.
In the full harness the left controller's server mounts the same routes through gate.service.handle.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import service, settings


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _reply(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self._reply(*service.handle("GET", self.path))

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            body = {}
        self._reply(*service.handle("POST", self.path, body if isinstance(body, dict) else {}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--left-url")
    ap.add_argument("--right-url")
    ap.add_argument("--overhead-from")
    a = ap.parse_args(argv)
    for name, value in (("LEFT_URL", a.left_url), ("RIGHT_URL", a.right_url), ("OVERHEAD_FROM", a.overhead_from)):
        if value:
            settings.set_override(name, value)
    service.get_gate()  # start the frame and jaw pollers
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"gate on http://{a.host}:{a.port}  overhead {settings.overhead_url()}  key {'found' if settings.api_key() else 'MISSING'}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
