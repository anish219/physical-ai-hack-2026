import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from recorder import Recorder, RunStore, compact_state, make_handler, read_run


def eventually(predicate, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition timed out")


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_id = "20261108T120000-1234abcd"

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_dimos_roundtrip_and_wal_read_during_recording(self):
        store = RunStore(self.root / self.run_id, "Real SDK", "test", 10000)
        jpeg = b"\xff\xd8actual-cached-jpeg\xff\xd9"
        store.append("overhead", time.time(), jpeg, {"captured_at": 10})
        store.append("events", time.time(), b'{"stage":"alignment"}', {"title": "alignment"})
        store.commit()
        info = read_run(self.root, self.run_id)
        self.assertEqual(info["counts"], {"overhead": 1, "events": 1})
        stream, payload, _ = read_run(self.root, self.run_id, 1)
        self.assertEqual((stream, payload), ("overhead", jpeg))
        store.close("test", {"dropped": 3})
        self.assertEqual(read_run(self.root, self.run_id)["metrics"]["dropped"], 3)

    def test_storage_limit_preserves_prior_data(self):
        store = RunStore(self.root / self.run_id, "Limit", "test", 5)
        store.append("state", time.time(), b"{}", {})
        with self.assertRaises(OverflowError):
            store.append("overhead", time.time(), b"123456", {})
        store.close("storage_limit", {})
        self.assertEqual(len(read_run(self.root, self.run_id)["observations"]), 1)

    def test_bad_paths_and_streams_are_rejected(self):
        with self.assertRaises(ValueError):
            read_run(self.root, "../secret")
        store = RunStore(self.root / self.run_id, "Invalid", "test", 500)
        with self.assertRaises(ValueError):
            store.append("state;DROP", time.time(), b"{}", {})
        store.close("test", {})

    def test_telemetry_omits_drafts_and_credentials(self):
        data = compact_state(
            {
                "api_drafts": {"api_key": "secret"},
                "credentials": "secret",
                "dream": {"right": {"busy": True, "api_key": "secret"}},
                "arms": {"left": {"arm_open": False}},
            }
        )
        self.assertNotIn("secret", json.dumps(data))
        self.assertTrue(data["dream"]["right"]["busy"])


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rec = Recorder(self.tmp.name, min_free=0, auto=False)
        # Writer-only tests: no real network endpoints are used.
        self.writer = threading.Thread(target=self.rec.writer)
        self.writer.start()
        self.rec.writer_ready.wait()

    def tearDown(self):
        self.rec.quit.set()
        self.writer.join(3)
        self.tmp.cleanup()

    def test_start_stop_flush_and_no_cross_run_contamination(self):
        first = self.rec.request("start", "One")["id"]
        self.rec.enqueue("state", b'{"arm":"left"}')
        self.rec.request("stop")
        self.assertEqual(read_run(self.rec.root, first)["counts"]["state"], 1)
        second = self.rec.request("start", "Two")["id"]
        self.rec.enqueue("state", b'{"old":true}', expected_run=first)
        self.rec.enqueue("state", b'{"new":true}', expected_run=second)
        self.rec.request("stop")
        info = read_run(self.rec.root, second)
        self.assertEqual(info["counts"]["state"], 1)
        self.assertEqual(read_run(self.rec.root, second, 1)[1], b'{"new":true}')

    def test_full_queue_drops_without_blocking_producer(self):
        self.rec.request("start")
        # Hold writer in a deliberately slow disk write; producers still never wait.
        entered = threading.Event()
        original = RunStore.append

        def slow(store, *args):
            entered.set()
            time.sleep(0.25)
            return original(store, *args)

        with patch.object(RunStore, "append", slow):
            self.rec.enqueue("state", b"{}")
            self.assertTrue(entered.wait(1))
            began = time.monotonic()
            for _ in range(1000):
                self.rec.enqueue("state", b"{}")
            self.assertLess(time.monotonic() - began, 0.1)
            self.assertLessEqual(self.rec.pending.qsize(), 24)
            self.assertGreater(self.rec.status()["metrics"]["dropped"], 0)
        self.rec.request("stop")

    def test_disk_limit_stops_recording_not_control(self):
        self.rec.max_bytes = 5
        run_id = self.rec.request("start")["id"]
        self.rec.enqueue("state", b"123456")
        eventually(lambda: self.rec.active is None)
        self.assertEqual(read_run(self.rec.root, run_id)["end_reason"], "storage_limit")

    def test_duration_limit_and_low_disk_refusal(self):
        self.rec.max_seconds = 0.01
        run_id = self.rec.request("start")["id"]
        eventually(lambda: self.rec.active is None)
        self.assertEqual(read_run(self.rec.root, run_id)["end_reason"], "duration_limit")
        self.rec.min_free = 2**80
        with self.assertRaisesRegex(ValueError, "free"):
            self.rec.request("start")
        self.assertIsNone(self.rec.active)

    def test_live_preview_works_without_recording_and_expires(self):
        fetched = []

        def fake_fetch(url, image=False):
            fetched.append(url)
            return b"\xff\xd8preview\xff\xd9", {"X-Stamp": str(time.time())}

        self.rec.fetch = fake_fetch
        self.rec.preview_until = time.monotonic() + 1.2
        sampler = threading.Thread(target=self.rec.poll_frames)
        sampler.start()
        try:
            eventually(lambda: "overhead" in self.rec.live_frames)
            self.assertIsNone(self.rec.active)
            self.assertEqual(self.rec.metrics["written"], 0)
            self.assertEqual(list(self.rec.root.iterdir()), [])
            time.sleep(2.2)
            count = len(fetched)
            time.sleep(1.1)
            self.assertEqual(len(fetched), count)
            self.assertLessEqual(count, 2)
            self.assertTrue(all(url.endswith("/overhead.jpg") for url in fetched))
        finally:
            self.rec.quit.set()
            sampler.join(2)

    def test_service_stop_drains_queue(self):
        run_id = self.rec.request("start")["id"]
        for _ in range(10):
            self.rec.enqueue("state", b"{}")
        self.rec.sampling.clear()
        self.rec.quit.set()
        self.writer.join(3)
        info = read_run(self.rec.root, run_id)
        self.assertEqual(info["counts"]["state"], 10)
        self.assertEqual(info["end_reason"], "service_shutdown")

    def test_local_control_and_remote_read_only_and_csrf(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.rec))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:

            def post(headers):
                return urllib.request.urlopen(
                    urllib.request.Request(
                        base + "/api/start",
                        data=b'{"label":"HTTP test"}',
                        headers=headers,
                    ),
                    timeout=2,
                )

            with self.assertRaises(urllib.error.HTTPError) as rejected:
                post({"Origin": "http://evil.example"})
            self.assertEqual(rejected.exception.code, 403)
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                post({"Host": "10.103.0.132:8791"})
            self.assertEqual(rejected.exception.code, 403)
            with post({"Content-Type": "application/json"}) as response:
                run_id = json.load(response)["id"]
            self.rec.request("stop")
            with urllib.request.urlopen(base + "/api/runs/" + run_id) as response:
                self.assertEqual(json.load(response)["label"], "HTTP test")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class FeedTests(unittest.TestCase):
    def test_stale_jpegs_not_saved_and_only_gets_used(self):
        requests = []

        class Feed(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):  # noqa: N802
                requests.append(self.path)
                body = b"\xff\xd8fake\xff\xd9"
                self.send_response(200)
                self.send_header("X-Stamp", str(time.time() - 100))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Feed)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(tmp, auto=False, min_free=0)
            rec.cameras = {"overhead": (f"http://127.0.0.1:{server.server_port}/overhead.jpg",)}
            writer = threading.Thread(target=rec.writer)
            writer.start()
            run_id = rec.request("start")["id"]
            sampler = threading.Thread(target=rec.poll_frames)
            sampler.start()
            eventually(lambda: rec.status()["metrics"]["skipped_stale"] > 0)
            rec.request("stop")
            rec.quit.set()
            sampler.join(2)
            writer.join(2)
            self.assertNotIn("overhead", read_run(rec.root, run_id)["counts"])
            self.assertEqual(requests, ["/overhead.jpg"])
        server.shutdown()
        server.server_close()


class CombinedIntegrationTests(unittest.TestCase):
    def test_auto_match_capture_events_images_and_finish(self):
        match = {"started": None, "stopped": None}

        class Feed(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):  # noqa: N802
                now = time.time()
                if self.path == "/api/state":
                    data = {
                        "match_started_at": match["started"],
                        "match_stopped_at": match["stopped"],
                        "arms": {"left": {"arm_open": True}},
                    }
                    body = json.dumps(data).encode()
                elif self.path == "/api/status":
                    body = json.dumps(
                        {
                            "events": [
                                {
                                    "at": now,
                                    "side": "right",
                                    "stage": "alignment",
                                    "state": "running",
                                    "detail": "fixture only",
                                }
                            ]
                        }
                    ).encode()
                else:
                    body = b"\xff\xd8test-jpeg\xff\xd9"
                self.send_response(200)
                self.send_header("X-Stamp", str(now))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Feed)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rec = Recorder(tmp, controller=base, auto=True, min_free=0)
                rec.cameras = {name: (base + "/cached.jpg",) for name in ("overhead", "left_wrist", "right_wrist")}
                rec.launch()
                try:
                    eventually(lambda: rec.last_seen is not None)
                    match["started"] = time.time()
                    eventually(lambda: rec.active is not None, timeout=4)
                    run_id = rec.active["id"]
                    eventually(lambda: rec.metrics["written"] >= 5, timeout=4)
                    match["stopped"] = time.time()
                    eventually(lambda: rec.active is None, timeout=4)
                    info = read_run(rec.root, run_id)
                    for stream in (
                        "events",
                        "state",
                        "overhead",
                        "left_wrist",
                        "right_wrist",
                    ):
                        self.assertGreater(info["counts"][stream], 0)
                    self.assertEqual(info["end_reason"], "match_finished")
                    self.assertIn("overhead", rec.live_frames)
                finally:
                    rec.shutdown()
        finally:
            server.shutdown()
            server.server_close()

    def test_viewer_token_has_no_publish_grant_and_publisher_requires_localhost(self):
        import base64
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "livekit-secret.json").write_text(json.dumps({"key": "test", "secret": "test-secret"}))
            rec = Recorder(root / "runs", min_free=0)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(rec))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with patch("recorder.ROOT", root):
                    req = urllib.request.Request(
                        base + "/api/livekit/token",
                        headers={"Host": "10.103.0.132:8791"},
                    )
                    with urllib.request.urlopen(req) as response:
                        config = json.load(response)
                    body = config["token"].split(".")[1]
                    claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
                    self.assertFalse(claims["video"]["canPublish"])
                    req = urllib.request.Request(
                        base + "/api/livekit/token?role=publisher",
                        headers={"Host": "10.103.0.132:8791"},
                    )
                    with self.assertRaises(urllib.error.HTTPError) as rejected:
                        urllib.request.urlopen(req)
                    self.assertEqual(rejected.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
