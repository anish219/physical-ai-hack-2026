import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from dream_hackathon.sequences import SequenceManager
from hillclimb.left import match


def test_script_start_arms_and_goes_and_stop_posts_stop():
    events = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def _send(self, payload):
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0] == "/api/state":
                self._send({"ok": True})
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            events.append((self.path.split("?")[0], self._body()))
            self._send({"ok": True, "why": "ok"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous = match.CFG["HILLCLIMB_PORT"]
    match.CFG["HILLCLIMB_PORT"] = server.server_address[1]
    try:
        assert match.serves(match.base_url())
        assert not match.serves("http://127.0.0.1:8090/start")
        match.start({"selection": "left"})
        match.stop({"event": "stop"})
    finally:
        match.CFG["HILLCLIMB_PORT"] = previous
        server.shutdown()
        thread.join(timeout=2)
    assert [path for path, _body in events] == ["/left/arm", "/left/go", "/left/stop"]
    assert events[0][1]["backend"] == "scripted"
    assert events[0][1]["mode"] == "match"
    assert events[0][1]["drive"] == "auto"


def test_http_start_for_the_hillclimb_port_arms_and_goes():
    events = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def _send(self, payload):
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            events.append((self.path.split("?")[0], None))
            self._send({"ok": True})

        def do_POST(self):  # noqa: N802
            events.append((self.path.split("?")[0], self._body()))
            self._send({"ok": True, "why": "ok"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous = match.CFG["HILLCLIMB_PORT"]
    match.CFG["HILLCLIMB_PORT"] = server.server_address[1]
    nodes = {
        side: SimpleNamespace(snapshot=lambda: {"arm_open": False}, cancel_home=lambda: None)
        for side in ("left", "right")
    }
    callers = {side: SimpleNamespace(sides=(side,)) for side in ("left", "right", "both")}
    sequences = SequenceManager(callers, nodes, lambda _event: None)
    sequences.update("left", {"mode": "http", "httpUrl": match.base_url(), "phases": []})
    sequences.update("right", {"phases": []})
    try:
        assert sequences.start_match() == ["left"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            posts = [path for path, _body in events if path.startswith("/left")]
            if posts == ["/left/arm", "/left/go"]:
                break
            time.sleep(0.02)
        sequences.stop_sides(["left"])
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and sequences.runs["left"].snapshot()["busy"]:
            time.sleep(0.02)
    finally:
        sequences.close()
        match.CFG["HILLCLIMB_PORT"] = previous
        server.shutdown()
        thread.join(timeout=2)
    posts = [(path, body) for path, body in events if path.startswith("/left")]
    assert [path for path, _body in posts] == ["/left/arm", "/left/go", "/left/stop"]
    assert "event" not in posts[0][1]
