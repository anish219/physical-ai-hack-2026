"""Independent, bounded, read-only observer. Never imports robot control code."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import hmac
import json
import os
import queue
import re
import shutil
import signal
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".runtime" / "dimos-0.0.15"))

from dimos.memory.blobstore.sqlite import SqliteBlobStore  # noqa: E402

MAX_BODY = 1024 * 1024
STREAMS = ("state", "events", "overhead", "left_wrist", "right_wrist")
RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[a-f0-9]{8}$")


def compact_state(state):
    """Allowlist telemetry; drafts, credentials, prompts and script bodies stay out."""
    result = {k: state.get(k) for k in ("inputs", "combined_input", "match_started_at", "match_stopped_at")}
    for group, keys in {
        "arms": ("arm_open", "q", "grip", "error", "state", "pose", "cameras"),
        "dream": ("state", "busy", "running", "error", "checkpoint", "model"),
        "sequences": (
            "state",
            "busy",
            "running",
            "phase_title",
            "step_label",
            "error",
            "elapsedSeconds",
        ),
        "flows": ("state", "busy", "running", "error"),
    }.items():
        result[group] = {
            side: {key: data[key] for key in keys if key in data}
            for side, data in state.get(group, {}).items()
            if isinstance(data, dict)
        }
    return result


class RunStore:
    """One SQLite connection/writer per run. Blobs are written AND read using dimOS."""

    def __init__(self, directory, label, source, max_bytes):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.path = directory / "run.db"
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, at REAL, stream TEXT, meta TEXT)")
        self.conn.execute("CREATE INDEX timeline ON observations(at, id)")
        self.conn.execute("CREATE TABLE run_info (id INTEGER PRIMARY KEY, data TEXT)")
        self.store = SqliteBlobStore(conn=self.conn)
        self.store.start()
        self.max_bytes = max_bytes
        self.info = {
            "id": directory.name,
            "label": label,
            "source": source,
            "started_at": time.time(),
            "ended_at": None,
            "status": "recording",
            "counts": {},
            "payload_bytes": 0,
            "recorder": "dimos.memory.blobstore.sqlite.SqliteBlobStore",
            "dimos_version": "0.0.15",
        }
        self.save_info()
        self.commit()

    def save_info(self):
        self.conn.execute("INSERT OR REPLACE INTO run_info VALUES (1, ?)", (json.dumps(self.info),))

    def append(self, stream, at, payload, meta):
        if stream not in STREAMS:
            raise ValueError("Unknown stream")
        if self.info["payload_bytes"] + len(payload) > self.max_bytes:
            raise OverflowError("Per-run storage limit reached")
        row = self.conn.execute(
            "INSERT INTO observations(at,stream,meta) VALUES(?,?,?)",
            (at, stream, json.dumps(meta)),
        )
        self.store.put(stream, row.lastrowid, payload)
        self.info["counts"][stream] = self.info["counts"].get(stream, 0) + 1
        self.info["payload_bytes"] += len(payload)

    def commit(self):
        self.save_info()
        self.conn.commit()

    def close(self, reason, metrics):
        self.info.update(ended_at=time.time(), status="finished", end_reason=reason, metrics=metrics)
        self.commit()
        self.store.stop()
        self.conn.close()


def read_run(root, run_id, observation=None, timeline=True):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid run id")
    path = root / run_id / "run.db"
    if not path.exists():
        raise FileNotFoundError(run_id)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.2)
    try:
        info = json.loads(conn.execute("SELECT data FROM run_info WHERE id=1").fetchone()[0])
        if observation is not None:
            row = conn.execute("SELECT stream,meta FROM observations WHERE id=?", (observation,)).fetchone()
            if not row or row[0] not in STREAMS:
                raise FileNotFoundError("Observation")
            store = SqliteBlobStore(conn=conn)
            store.start()
            try:
                payload = store.get(row[0], observation)
            finally:
                store.stop()
            return row[0], payload, json.loads(row[1])
        if not timeline:
            return info
        rows = conn.execute("SELECT id,at,stream,meta FROM observations ORDER BY at,id LIMIT 12000").fetchall()
        info["observations"] = [dict(id=r[0], at=r[1], stream=r[2], **json.loads(r[3])) for r in rows]
        return info
    finally:
        conn.close()


class Recorder:
    def __init__(
        self,
        root,
        controller="http://127.0.0.1:8081",
        queue_size=24,
        frame_interval=1.0,
        state_interval=2.0,
        max_seconds=900,
        max_bytes=512 * 1024**2,
        min_free=2 * 1024**3,
        auto=True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.controller = controller.rstrip("/")
        self.frame_interval = max(1.0, frame_interval)
        self.state_interval = max(2.0, state_interval)
        self.max_seconds, self.max_bytes, self.min_free, self.auto = (
            max_seconds,
            max_bytes,
            min_free,
            auto,
        )
        self.pending = queue.Queue(maxsize=queue_size)
        self.commands = queue.Queue(maxsize=4)
        self.quit = threading.Event()
        self.lock = threading.Lock()
        self.active = None
        self.sampling = threading.Event()
        self.last_state = None
        self.last_seen = None
        self.error = ""
        self.metrics = {
            "dropped": 0,
            "fetch_errors": 0,
            "skipped_stale": 0,
            "written": 0,
            "bytes": 0,
            "slow_fetches": 0,
            "queue_peak": 0,
        }
        self.live_frames = {}
        self.preview_until = 0.0
        self.livekit_host = "10.103.0.132"
        self.cameras = {
            "overhead": (
                "http://127.0.0.1:8691/overhead.jpg",
                "http://127.0.0.1:8692/overhead.jpg",
            ),
            "left_wrist": ("http://127.0.0.1:8691/wrist.jpg",),
            "right_wrist": ("http://127.0.0.1:8692/wrist.jpg",),
        }
        self.threads = []
        self.writer_ready = threading.Event()

    def launch(self):
        for target in (
            self.writer,
            self.poll_state,
            self.poll_events,
            self.poll_frames,
        ):
            thread = threading.Thread(target=target, daemon=True, name=target.__name__)
            self.threads.append(thread)
            thread.start()
        self.writer_ready.wait(2)

    def bump(self, key, amount=1):
        with self.lock:
            self.metrics[key] += amount

    def fetch(self, url, image=False):
        start = time.monotonic()
        try:
            # No ambient proxies: these are loopback requests to existing servers.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=0.25) as response:
                payload = response.read(MAX_BODY + 1)
                if len(payload) > MAX_BODY:
                    raise ValueError("Feed response exceeded 1 MiB")
                headers = dict(response.headers)
                if image and not payload.startswith(b"\xff\xd8"):
                    raise ValueError("Not a JPEG")
                return (payload, headers) if image else json.loads(payload)
        except (OSError, ValueError, TimeoutError):
            self.bump("fetch_errors")
            return None
        finally:
            if time.monotonic() - start > 0.15:
                self.bump("slow_fetches")

    def enqueue(self, stream, payload, meta=None, at=None, expected_run=None):
        with self.lock:
            run_id = self.active["id"] if self.active and self.sampling.is_set() else None
        if run_id is None or (expected_run is not None and expected_run != run_id):
            return
        try:
            self.pending.put_nowait((run_id, stream, at or time.time(), payload, meta or {}))
            with self.lock:
                self.metrics["queue_peak"] = max(self.metrics["queue_peak"], self.pending.qsize())
        except queue.Full:
            self.bump("dropped")

    def request(self, action, label="Rehearsal", source="manual"):
        if action not in ("start", "stop"):
            raise ValueError("Unknown action")
        if action == "start" and len(label) > 120:
            raise ValueError("Label is too long")
        done, result = threading.Event(), {}
        self.commands.put_nowait((action, label.strip() or "Rehearsal", source, done, result))
        if not done.wait(2):
            raise TimeoutError("Recorder is busy; check status before retrying")
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def status(self):
        with self.lock:
            active = dict(self.active) if self.active else None
            metrics = dict(self.metrics)
            last_seen = self.last_seen
            error = self.error
        return {
            "active": active,
            "metrics": metrics,
            "last_controller_seen": last_seen,
            "error": error,
            "auto_record_matches": self.auto,
            "queue_depth": self.pending.qsize(),
            "limits": {
                "frame_hz": 1 / self.frame_interval,
                "state_hz": 1 / self.state_interval,
                "max_seconds": self.max_seconds,
                "max_mib": self.max_bytes // 1024**2,
                "queue_items": self.pending.maxsize,
            },
            "recorder": "dimos.memory.blobstore.sqlite.SqliteBlobStore",
            "version": "0.0.15",
        }

    def writer(self):
        store = None
        commit_at = time.monotonic()
        self.writer_ready.set()

        def finish(reason):
            nonlocal store
            self.sampling.clear()
            if store:
                store.close(reason, self.status()["metrics"])
                store = None
            with self.lock:
                self.active = None

        try:
            while not self.quit.is_set() or not self.pending.empty():
                try:
                    action, label, source, done, result = self.commands.get_nowait()
                except queue.Empty:
                    pass
                else:
                    try:
                        if action == "start":
                            if store:
                                raise ValueError("A recording is already active")
                            if shutil.disk_usage(self.root).free < self.min_free:
                                raise ValueError("Less than 2 GiB free; recording disabled")
                            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:8]
                            store = RunStore(self.root / run_id, label, source, self.max_bytes)
                            with self.lock:
                                self.active = {
                                    "id": run_id,
                                    "label": label,
                                    "started_at": store.info["started_at"],
                                    "source": source,
                                }
                                self.metrics = dict.fromkeys(self.metrics, 0)
                                self.error = ""
                            self.sampling.set()
                            result["id"] = run_id
                        else:
                            # Stop sampling first; drain all already queued observations before closing.
                            self.sampling.clear()
                            while store:
                                try:
                                    item = self.pending.get_nowait()
                                except queue.Empty:
                                    break
                                if item[0] == store.info["id"]:
                                    store.append(*item[1:])
                                    self.bump("written")
                                    self.bump("bytes", len(item[3]))
                            finish("operator" if source == "manual" else source)
                            result["stopped"] = True
                    except Exception as exc:
                        result["error"] = str(exc)
                        if action == "stop":
                            finish("storage_error")
                    finally:
                        done.set()
                try:
                    item = self.pending.get(timeout=0.1)
                except queue.Empty:
                    item = None
                if item and store and item[0] == store.info["id"]:
                    try:
                        store.append(*item[1:])
                        self.bump("written")
                        self.bump("bytes", len(item[3]))
                    except OverflowError:
                        finish("storage_limit")
                now = time.monotonic()
                if store and now - commit_at >= 1:
                    store.commit()
                    commit_at = now
                    if time.time() - store.info["started_at"] >= self.max_seconds:
                        finish("duration_limit")
                    elif shutil.disk_usage(self.root).free < self.min_free:
                        finish("disk_space_limit")
        except Exception as exc:
            with self.lock:
                self.error = f"Recorder stopped: {exc}"
            self.sampling.clear()
        finally:
            try:
                finish("service_shutdown")
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)

    def poll_state(self):
        marker = object()
        initialized = False
        while not self.quit.is_set():
            state = self.fetch(self.controller + "/api/state")
            if isinstance(state, dict):
                with self.lock:
                    self.last_state = compact_state(state)
                    self.last_seen = time.time()
                    active = dict(self.active) if self.active else None
                started, ended = (
                    state.get("match_started_at"),
                    state.get("match_stopped_at"),
                )
                if self.auto and initialized and started and started != marker and not ended and not active:
                    with contextlib.suppress(ValueError, TimeoutError, queue.Full):
                        self.request(
                            "start",
                            "Match " + datetime.now().strftime("%H:%M:%S"),
                            "match",
                        )
                marker, initialized = started, True
                if (
                    active
                    and active["source"] == "match"
                    and ended
                    and ended >= active["started_at"] - self.state_interval
                ):
                    with contextlib.suppress(ValueError, TimeoutError, queue.Full):
                        self.request("stop", source="match_finished")
                self.enqueue(
                    "state",
                    json.dumps(compact_state(state)).encode(),
                    {"title": "Robot state"},
                )
            self.quit.wait(self.state_interval)

    def poll_events(self):
        run_id, seen = None, set()
        while not self.quit.is_set():
            with self.lock:
                active = dict(self.active) if self.active else None
            if self.sampling.is_set() and active:
                if run_id != active["id"]:
                    run_id, seen = active["id"], set()
                feed = self.fetch(self.controller + "/api/status")
                if isinstance(feed, dict):
                    for event in feed.get("events", []):
                        if not isinstance(event, dict) or event.get("at", 0) < active["started_at"]:
                            continue
                        encoded = json.dumps(event, sort_keys=True).encode()
                        key = hashlib.sha256(encoded).digest()
                        if key in seen:
                            continue
                        # The feed is bounded, so bound this dedup set as well.
                        if len(seen) > 4096:
                            seen.clear()
                        seen.add(key)
                        meta = {k: event.get(k) for k in ("side", "stage", "state", "detail")}
                        meta["title"] = f"{event.get('side', '')} · {event.get('stage', '')}"
                        self.enqueue("events", encoded, meta, event["at"], expected_run=run_id)
            self.quit.wait(self.state_interval)

    def poll_frames(self):
        stamps, run_id = {}, None
        while not self.quit.is_set():
            began = time.monotonic()
            with self.lock:
                active = dict(self.active) if self.active else None
                preview = time.monotonic() < self.preview_until
            recording = self.sampling.is_set() and active is not None
            if recording or preview:
                current_run = active["id"] if recording else None
                if run_id != current_run:
                    stamps, run_id = {}, current_run
                for stream, urls in self.cameras.items():
                    if not recording and stream != "overhead":
                        continue
                    if recording and not self.sampling.is_set():
                        break
                    frame = None
                    for url in urls:
                        candidate = self.fetch(url, image=True)
                        if not candidate:
                            continue
                        data, headers = candidate
                        stamp = headers.get("X-Stamp")
                        try:
                            captured_at = float(stamp)
                        except (TypeError, ValueError):
                            captured_at = None
                        if captured_at is None or not -1 <= time.time() - captured_at <= 5:
                            self.bump("skipped_stale")
                            continue
                        frame = candidate
                        break
                    if not frame:
                        continue
                    if stamps.get(stream) == stamp:
                        self.bump("skipped_stale")
                        continue
                    with self.lock:
                        self.live_frames[stream] = (data, captured_at)
                    stamps[stream] = stamp
                    meta = {
                        "title": stream.replace("_", " "),
                        "captured_at": captured_at,
                        "joints": headers.get("X-Joints"),
                        "grip": headers.get("X-Grip"),
                    }
                    if recording:
                        self.enqueue(stream, data, meta, captured_at, expected_run=run_id)
            self.quit.wait(max(0.1, self.frame_interval - (time.monotonic() - began)))

    def shutdown(self):
        self.sampling.clear()
        self.quit.set()
        for thread in self.threads:
            thread.join(timeout=3)


def make_handler(recorder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, value, status=200, content_type="application/json"):
            payload = json.dumps(value).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def local_operator(self):
            host = urllib.parse.urlsplit("http://" + self.headers.get("Host", "")).hostname
            return self.client_address[0] in ("127.0.0.1", "::1") and host in (
                "127.0.0.1",
                "localhost",
                "::1",
            )

        def do_GET(self):  # noqa: N802
            url = urllib.parse.urlsplit(self.path)
            try:
                if url.path == "/":
                    return self.reply(
                        (ROOT / "index.html").read_bytes(),
                        content_type="text/html; charset=utf-8",
                    )
                if url.path == "/livekit-client.js":
                    return self.reply(
                        (ROOT / ".runtime/livekit-client.js").read_bytes(),
                        content_type="text/javascript; charset=utf-8",
                    )
                if url.path == "/api/livekit/token":
                    role = urllib.parse.parse_qs(url.query).get("role", ["viewer"])[0]
                    if role == "publisher" and not self.local_operator():
                        return self.reply({"error": "Publisher requires local operator access"}, 403)
                    credentials = json.loads((ROOT / "livekit-secret.json").read_text())

                    def encode(data):
                        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

                    now = int(time.time())
                    claims = {
                        "iss": credentials["key"],
                        "sub": role + "-" + uuid.uuid4().hex[:8],
                        "nbf": now - 5,
                        "exp": now + 3600,
                        "video": {
                            "roomJoin": True,
                            "room": "leonard-runs",
                            "canSubscribe": True,
                            "canPublish": role == "publisher",
                            "canPublishData": False,
                        },
                    }
                    body = ".".join(encode(json.dumps(x).encode()) for x in ({"alg": "HS256", "typ": "JWT"}, claims))
                    signature = hmac.new(credentials["secret"].encode(), body.encode(), hashlib.sha256).digest()
                    return self.reply(
                        {
                            "token": body + "." + encode(signature),
                            "url": "ws://" + recorder.livekit_host + ":7880",
                            "room": "leonard-runs",
                        }
                    )
                if url.path == "/api/live/overhead.jpg":
                    with recorder.lock:
                        recorder.preview_until = time.monotonic() + 3.0
                        frame = recorder.live_frames.get("overhead")
                    if not frame or time.time() - frame[1] > 5:
                        return self.reply(
                            {"error": "Scene preview is warming up or the camera feed is unavailable."},
                            503,
                        )
                    return self.reply(frame[0], content_type="image/jpeg")
                if url.path == "/api/status":
                    return self.reply({**recorder.status(), "can_control": self.local_operator()})
                if url.path == "/api/runs":
                    runs = []
                    for directory in sorted(recorder.root.iterdir(), reverse=True):
                        if not RUN_ID.fullmatch(directory.name):
                            continue
                        try:
                            data = read_run(recorder.root, directory.name, timeline=False)
                            data.pop("observations", None)
                            if data["status"] == "recording" and (recorder.active or {}).get("id") != directory.name:
                                data["status"] = "interrupted"
                            runs.append(data)
                        except (OSError, sqlite3.Error, TypeError):
                            continue
                        if len(runs) >= 100:
                            break
                    return self.reply(runs)
                parts = url.path.strip("/").split("/")
                if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                    return self.reply(read_run(recorder.root, parts[2]))
                if len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3] == "observations":
                    stream, payload, _meta = read_run(recorder.root, parts[2], int(parts[4]))
                    return self.reply(
                        payload,
                        content_type="image/jpeg"
                        if stream not in ("state", "events")
                        else "application/json; charset=utf-8",
                    )
                self.reply({"error": "Not found"}, 404)
            except (ValueError, FileNotFoundError, KeyError):
                self.reply({"error": "Not found"}, 404)
            except (sqlite3.Error, OSError):
                self.reply({"error": "Recording unavailable; retry shortly"}, 503)

        def do_POST(self):  # noqa: N802
            origin = self.headers.get("Origin")
            expected_origin = "http://" + self.headers.get("Host", "")
            if not self.local_operator() or (origin and origin != expected_origin):
                return self.reply(
                    {"error": "Read-only viewer. Use your localhost SSH tunnel to control recording."},
                    403,
                )
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 <= length <= 2048:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/start":
                    result = recorder.request("start", str(data.get("label", "Rehearsal")))
                elif self.path == "/api/stop":
                    result = recorder.request("stop")
                else:
                    return self.reply({"error": "Not found"}, 404)
                self.reply(result)
            except (ValueError, TimeoutError, queue.Full) as exc:
                self.reply({"error": str(exc)}, 409)

    return Handler


class LimitedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(2)
        return request, address

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--root", type=Path, default=ROOT / "runs")
    parser.add_argument("--controller", default="http://127.0.0.1:8081")
    parser.add_argument("--no-auto", action="store_true")
    parser.add_argument("--livekit-host", default="10.103.0.132")
    args = parser.parse_args()
    try:
        os.nice(15)
        if hasattr(os, "sched_setscheduler"):
            os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
        pass
    recorder = Recorder(args.root, controller=args.controller, auto=not args.no_auto)
    recorder.livekit_host = args.livekit_host
    server = LimitedHTTPServer((args.bind, args.port), make_handler(recorder))
    recorder.launch()

    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        f"Run recorder on {args.bind}:{args.port}; dimOS 0.0.15; no robot commands",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        recorder.shutdown()


if __name__ == "__main__":
    main()
