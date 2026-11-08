"""Independent LiveKit screen feed + Dimensional frame/event recording.

No project imports, robot connections, camera handles, or controller requests.
"""

import argparse
import base64
import hashlib
import hmac
import json
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
sys.path.insert(0, str(RUNTIME / "dimos-0.0.15"))

from dimos.memory.blobstore.sqlite import SqliteBlobStore  # noqa: E402

ROOM = "hillclimb-bonus"


def token(role):
    def encode(value):
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

    now = int(time.time())
    claims = {
        "iss": "devkey",
        "sub": f"{role}-{uuid.uuid4().hex[:8]}",
        "nbf": now - 5,
        "exp": now + 86400,
        "video": {
            "roomJoin": True,
            "room": ROOM,
            "canSubscribe": True,
            "canPublish": role == "publisher",
            "canPublishData": False,
        },
    }
    payload = ".".join(encode(json.dumps(x).encode()) for x in ({"alg": "HS256", "typ": "JWT"}, claims))
    signature = hmac.new(b"secret", payload.encode(), hashlib.sha256).digest()
    return f"{payload}.{encode(signature)}"


class Recorder:
    """Use Dimensional's real SQLite blob store for every saved payload."""

    def __init__(self, path):
        self.path = Path(path)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.store = SqliteBlobStore(conn=self.connection)
        self.store.start()
        self.lock = threading.Lock()
        self.counts = {"events": 0, "frames": 0}
        # Microseconds fit exactly in browser JSON numbers (nanoseconds don't).
        self.next_id = time.time_ns() // 1000
        self.latest_frame_id = None

    def save(self, stream, payload):
        with self.lock:
            self.next_id += 1
            key = self.next_id
            self.store.put(stream, key, payload)
            self.connection.commit()
            self.counts[stream] += 1
            if stream == "frames":
                self.latest_frame_id = key
            return key

    def event(self, data):
        return self.save("events", json.dumps({"recorded_at": time.time(), **data}).encode())

    def summary(self):
        with self.lock:
            return {
                "recorder": "dimos.memory.blobstore.sqlite.SqliteBlobStore",
                "database": str(self.path),
                **self.counts,
                "latest_frame_id": self.latest_frame_id,
            }

    def latest(self):
        with self.lock:
            return self.store.get("frames", self.latest_frame_id) if self.latest_frame_id else None

    def close(self):
        self.store.stop()
        self.connection.close()


def handler_for(recorder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, body, content_type="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlsplit(self.path)
            if url.path == "/":
                self.reply((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/app.js":
                self.reply((ROOT / "app.js").read_bytes(), "text/javascript")
            elif url.path == "/livekit-client.js":
                self.reply((RUNTIME / "livekit-client.umd.js").read_bytes(), "text/javascript")
            elif url.path == "/api/token":
                role = parse_qs(url.query).get("role", ["viewer"])[0]
                self.reply(json.dumps({"token": token(role), "url": "ws://127.0.0.1:7880", "room": ROOM}).encode())
            elif url.path == "/api/status":
                self.reply(json.dumps(recorder.summary()).encode())
            elif url.path == "/api/latest.jpg":
                data = recorder.latest()
                self.reply(
                    data or b"No frames recorded yet", "image/jpeg" if data else "text/plain", 200 if data else 404
                )
            else:
                self.reply(b"Not found", "text/plain", 404)

        def do_POST(self):  # noqa: N802
            path = urlsplit(self.path).path
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if path == "/api/event":
                key = recorder.event(json.loads(body))
            elif path == "/api/frame":
                key = recorder.save("frames", body)
            else:
                self.reply(b"Not found", "text/plain", 404)
                return
            self.reply(json.dumps({"id": key}).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    output = ROOT / "recordings" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    recorder = Recorder(output / "bonus.db")
    http = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(recorder))
    server_log = (output / "livekit.log").open("w", encoding="utf-8")
    livekit = subprocess.Popen(
        [str(RUNTIME / "livekit" / "livekit-server.exe"), "--dev", "--bind", "127.0.0.1", "--node-ip", "127.0.0.1"],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    recorder.event({"type": "demo_started", "room": ROOM})
    print(f"Publisher: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Viewer:    http://127.0.0.1:{args.port}/?viewer=1", flush=True)
    print(f"Recording: {recorder.path}", flush=True)
    print("Ctrl+C stops this demo and its own LiveKit server.", flush=True)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http.server_close()
        livekit.terminate()
        livekit.wait()
        recorder.event({"type": "demo_stopped"})
        recorder.close()
        server_log.close()


if __name__ == "__main__":
    main()
