"""Exercise real Dimensional persistence and the standalone HTTP endpoints."""

import base64
import hashlib
import hmac
import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from server import RUNTIME, Recorder, handler_for, token


class DemoTest(unittest.TestCase):
    def test_dimensional_payloads_survive_reopen(self):
        with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
            path = Path(directory) / "bonus.db"
            recorder = Recorder(path)
            frame = b"\xff\xd8jpeg-payload\xff\xd9"
            frame_id = recorder.save("frames", frame)
            event_id = recorder.event({"type": "video_received"})
            self.assertEqual(recorder.latest(), frame)
            recorder.close()
            reopened = sqlite3.connect(path)
            from dimos.memory.blobstore.sqlite import SqliteBlobStore

            store = SqliteBlobStore(conn=reopened)
            self.assertEqual(store.get("frames", frame_id), frame)
            self.assertEqual(json.loads(store.get("events", event_id))["type"], "video_received")
            store.stop()
            reopened.close()

    def test_http_records_and_reads_back_frame(self):
        with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
            recorder = Recorder(Path(directory) / "bonus.db")
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(recorder))
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                image = b"\xff\xd8http-round-trip\xff\xd9"
                request = urllib.request.Request(base + "/api/frame", data=image, method="POST")
                with urllib.request.urlopen(request) as response:
                    self.assertIn("id", json.load(response))
                with urllib.request.urlopen(base + "/api/latest.jpg") as response:
                    self.assertEqual(response.read(), image)
                with urllib.request.urlopen(base + "/api/status") as response:
                    status = json.load(response)
                self.assertEqual(status["frames"], 1)
                self.assertEqual(status["recorder"], "dimos.memory.blobstore.sqlite.SqliteBlobStore")
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
                recorder.close()

    def test_livekit_tokens_have_valid_signature_and_roles(self):
        for role in ("publisher", "viewer"):
            encoded = token(role)
            header, payload, signature = encoded.split(".")
            expected = hmac.new(b"secret", f"{header}.{payload}".encode(), hashlib.sha256).digest()
            self.assertEqual(base64.urlsafe_b64decode(signature + "=="), expected)
            claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
            self.assertEqual(claims["video"]["canPublish"], role == "publisher")
            self.assertTrue(claims["video"]["roomJoin"])
            self.assertEqual(claims["video"]["room"], "hillclimb-bonus")


if __name__ == "__main__":
    unittest.main()
