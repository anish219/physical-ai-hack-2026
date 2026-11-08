"""Local Dream-1 caller using only the selected independent arm HTTP server."""

import argparse
import base64
import copy
import json
import pathlib
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np

from dream_hackathon import layouts


class ArmEndpoint:
    """One fixed listener: never discovers, opens or sends to the opposite arm."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def request(self, path: str, body: dict | None = None) -> bytes:
        request = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={} if body is None else {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.read()

    def state(self) -> dict:
        return json.loads(self.request("/state"))

    def frame(self, name: str) -> bytes:
        return self.request(f"/{name}.jpg")

    def move(self, row: list[float], rate: float) -> None:
        self.request("/move", {"q": row[:6], "grip": row[6], "rate": rate})

    def stop(self) -> None:
        self.request("/stop", {})


def jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Could not encode camera frame")
    return encoded.tobytes()


def build_observation(
    side: str, state: dict, scene: bytes, wrist: bytes, blank: bytes, task: str
) -> tuple[dict, dict]:
    """Only supplied measured data enters the live side; the opposite side is constant dummy data.

    Dream's IK derives both tool poses from these joints. The dummy pose therefore
    comes from six zero joints, never a read of the opposite arm or cached context.
    Metal gripper drive is millimetres; the separate grip field is normalized.
    """
    other = "right" if side == "left" else "left"
    q = np.asarray(state["q"], dtype=float)
    if q.shape != (6,) or state["grip"] is None:
        raise ValueError(f"{side} arm server has no six-joint measurement yet")
    observation = {
        "prompt": task,
        f"state.{side}_arm": q.tolist(),
        f"state.{side}_gripper": [float(state["grip"]) * 100],
        f"state.{side}_grip": [float(state["grip"])],
        f"state.{other}_arm": [0.0] * 6,
        f"state.{other}_gripper": [0.0],
        f"state.{other}_grip": [0.0],
    }
    images = {"scene": scene, "wrist_l": blank, "wrist_r": blank}
    images["wrist_l" if side == "left" else "wrist_r"] = wrist
    frames = {f"video.{key}": base64.b64encode(value).decode("ascii") for key, value in images.items()}
    return frames, observation


def selected_rows(action: dict, side: str) -> list[list[float]]:
    """Discard all model output except this side's SDK-solved joint/grip chunk."""
    key = "q_l_chunk" if side == "left" else "q_r_chunk"
    rows = np.asarray(action[key], dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != 7:
        raise ValueError(f"Dream returned {key} with shape {rows.shape}; expected [rows, 7]")
    return rows.tolist()


def open_client(checkpoint: str, hz: float) -> Any:
    import dream

    client = dream.DreamClient()
    try:
        client.policy(
            {
                "checkpoint": checkpoint,
                "embodiment": layouts.embodiment("makermods"),
                "ik": {"arm": "metal", "head": "eef"},
                "action_hz": hz,
            }
        )
    except Exception:
        client.close()
        raise
    return client


def available_models() -> list[dict]:
    """List account checkpoints without opening a policy or invoking inference."""
    import dream

    with dream.DreamClient() as client:
        return [
            {"id": model.id, "owner": model.owner, "servable": model.servable}
            for model in client.get_models()
        ]


class ArmCaller:
    """Independent observation, inference and command workers for exactly one arm."""

    def __init__(
        self,
        side: str,
        endpoint: ArmEndpoint,
        *,
        width=320,
        height=180,
        hz=20.0,
        observation_hz=10.0,
        client_factory=open_client,
    ) -> None:
        self.side, self.endpoint = side, endpoint
        self.width, self.height, self.hz = width, height, hz
        self.observation_hz, self.client_factory = observation_hz, client_factory
        self.blank = jpeg(np.zeros((height, width, 3), dtype=np.uint8))
        self.lock = threading.RLock()
        self.command_lock = threading.Lock()
        self.shutdown = threading.Event()
        self.observer = None
        self.worker = None
        self.cancel = threading.Event()
        self.client = None
        self.images = {"scene": self.blank, "wrist": self.blank}
        self.sample = None
        self.state = dict(
            side=side,
            endpoint=endpoint.url,
            connected=False,
            running=False,
            busy=False,
            mode="idle",
            error="",
            observation_error="",
            task="",
            checkpoint="dreamlabs/dream-1-beta",
            measured=None,
            observation=None,
            rows=[],
            commands=0,
            calls=0,
            updated_at=None,
        )

    def snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.state)

    def connect(self) -> None:
        with self.lock:
            if self.observer is None:
                self.observer = threading.Thread(target=self.observe, daemon=True, name=f"observe:{self.side}")
                self.observer.start()
            self.state["connected"] = True

    def resize(self, encoded: bytes) -> bytes:
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Arm server returned an invalid JPEG")
        return jpeg(cv2.resize(image, (self.width, self.height)))

    def read_sample(self):
        return (
            self.endpoint.state(),
            self.resize(self.endpoint.frame("overhead")),
            self.resize(self.endpoint.frame("wrist")),
        )

    def observation(self, sample):
        return build_observation(self.side, *sample, self.blank, self.state["task"])

    def action_rows(self, action):
        return selected_rows(action, self.side)

    def send_row(self, row, rate):
        self.endpoint.move(row, rate)

    def observe(self) -> None:
        while not self.shutdown.is_set():
            started = time.monotonic()
            try:
                sample = self.read_sample()
                state, scene, wrist = sample
                with self.lock:
                    self.sample = sample
                    self.images = {"scene": scene, "wrist": wrist}
                    frames, observation = self.observation(sample)
                    if self.client is not None:
                        self.client.set_observation(frames, observation, timestamp=time.monotonic())
                    self.state.update(
                        measured=state, observation=observation, updated_at=time.time(), observation_error=""
                    )
            except Exception as error:
                with self.lock:
                    self.sample = None
                    self.state["observation_error"] = str(error)
            self.shutdown.wait(max(0, 1 / self.observation_hz - (time.monotonic() - started)))

    def start(self, values: dict, *, run: bool) -> None:
        task = str(values.get("task", "")).strip()
        checkpoint = str(values.get("checkpoint", "dreamlabs/dream-1-beta")).strip()
        rate = float(values.get("rate", 0.3))
        if not task or not checkpoint:
            raise ValueError("Enter a task and checkpoint")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Rate must be a positive finite number")
        with self.lock:
            if self.state["busy"]:
                raise ValueError("This arm already has an API call or run in progress")
            if self.sample is None:
                raise ValueError("Connect and wait for live arm data and both cameras")
            self.cancel = threading.Event()
            self.state.update(
                task=task,
                checkpoint=checkpoint,
                error="",
                busy=True,
                running=run,
                mode="running" if run else "calling",
            )
            self.worker = threading.Thread(
                target=self.infer, args=(checkpoint, rate, run, self.cancel), daemon=True, name=f"dream:{self.side}"
            )
            self.worker.start()

    def infer(self, checkpoint: str, rate: float, run: bool, cancel: threading.Event) -> None:
        client = None
        try:
            client = self.client_factory(checkpoint, self.hz)
            with self.lock:
                if cancel.is_set():
                    return
                self.client = client
                if self.sample is None:
                    raise ValueError(self.state["observation_error"] or "No observation available")
                frames, observation = self.observation(self.sample)
                client.set_observation(frames, observation, timestamp=time.monotonic())
            while not cancel.is_set():
                action = client.get_action()
                rows = self.action_rows(action)
                with self.lock:
                    self.state.update(rows=rows, calls=self.state["calls"] + 1)
                if not run:
                    break
                # Play this chunk at the configured action rate, then request the next.
                # Observation capture continues independently while Dream is in flight.
                for row in rows:
                    started = time.monotonic()
                    with self.command_lock:
                        if cancel.is_set():
                            break
                        self.send_row(row, rate)
                        with self.lock:
                            self.state["commands"] += 1
                    if cancel.wait(max(0, 1 / self.hz - (time.monotonic() - started))):
                        break
        except Exception as error:
            with self.lock:
                self.state["error"] = str(error)
        finally:
            with self.lock:
                self.client = None
            try:
                if client is not None:
                    client.close()
            finally:
                with self.lock:
                    self.state.update(busy=False, running=False, mode="idle")

    def stop(self) -> None:
        # Serialize the explicit stop with writes, including replies still in flight.
        self.cancel.set()
        with self.command_lock:
            self.endpoint.stop()
        with self.lock:
            self.state.update(running=False, mode="stopping" if self.state["busy"] else "idle")

    def cancel_playback(self):
        self.cancel.set()
        with self.command_lock:
            pass
        with self.lock:
            self.state.update(running=False, mode="stopping" if self.state["busy"] else "idle")

    def close(self) -> None:
        self.cancel.set()
        self.shutdown.set()
        if self.observer is not None:
            self.observer.join(timeout=16)
        if self.worker is not None:
            self.worker.join(timeout=20)


class StationCaller(ArmCaller):
    """Main-controller caller: one selected arm or a shared bimanual policy session."""

    def __init__(self, selection, endpoints, **kwargs):
        self.endpoints = endpoints
        self.sides = ("left", "right") if selection == "both" else (selection,)
        super().__init__(selection, endpoints[self.sides[0]], **kwargs)
        self.state["endpoints"] = {side: endpoints[side].url for side in self.sides}

    def read_sample(self):
        if self.side != "both":
            return super().read_sample()
        states = {side: self.endpoints[side].state() for side in self.sides}
        scene = self.resize(self.endpoints["left"].frame("overhead"))
        wrists = {side: self.resize(self.endpoints[side].frame("wrist")) for side in self.sides}
        return states, scene, wrists

    def observation(self, sample):
        if self.side != "both":
            return super().observation(sample)
        states, scene, wrists = sample
        frames, observation = build_observation(
            "left", states["left"], scene, wrists["left"], self.blank, self.state["task"]
        )
        right_frames, right_observation = build_observation(
            "right", states["right"], scene, wrists["right"], self.blank, self.state["task"]
        )
        frames["video.wrist_r"] = right_frames["video.wrist_r"]
        observation.update({key: value for key, value in right_observation.items() if key.startswith("state.right_")})
        return frames, observation

    def action_rows(self, action):
        if self.side != "both":
            return super().action_rows(action)
        left, right = selected_rows(action, "left"), selected_rows(action, "right")
        if len(left) != len(right):
            raise ValueError("Dream returned different left/right chunk lengths")
        return [lrow + rrow for lrow, rrow in zip(left, right, strict=True)]

    def send_row(self, row, rate):
        # SDK-solved targets use the same direct path as the leader controller.
        for index, side in enumerate(self.sides):
            values = row[index * 7 : (index + 1) * 7]
            self.endpoints[side].request("/target", {"q": values[:6], "grip": values[6]})

    def stop(self):
        self.cancel.set()
        with self.command_lock:
            for side in self.sides:
                self.endpoints[side].stop()
        with self.lock:
            self.state.update(running=False, mode="stopping" if self.state["busy"] else "idle")


def handler_for(callers: dict[str, ArmCaller]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, code, body, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                return self.reply(
                    200, pathlib.Path(__file__).with_name("dream_caller.html").read_bytes(), "text/html; charset=utf-8"
                )
            if path == "/api/state":
                return self.reply(200, json.dumps({s: c.snapshot() for s, c in callers.items()}).encode())
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "api" and parts[1] in callers and parts[2] == "state":
                return self.reply(200, json.dumps(callers[parts[1]].snapshot()).encode())
            if len(parts) == 3 and parts[0] == "camera" and parts[1] in callers:
                caller, name = callers[parts[1]], parts[2].removesuffix(".jpg")
                with caller.lock:
                    frame = caller.blank if name == "blank" else caller.images.get(name)
                if frame is not None:
                    return self.reply(200, frame, "image/jpeg")
            self.reply(404, b"{}")

        def do_POST(self):  # noqa: N802
            parts = self.path.strip("/").split("/")
            if len(parts) != 3 or parts[0] != "api" or parts[1] not in callers:
                return self.reply(404, b"{}")
            caller, action = callers[parts[1]], parts[2]
            try:
                values = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if action == "connect":
                    caller.connect()
                elif action in ("call", "run"):
                    caller.start(values, run=action == "run")
                elif action == "stop":
                    caller.stop()
                elif action == "cancel":
                    caller.cancel_playback()
                else:
                    return self.reply(404, b"{}")
                self.reply(200, json.dumps(caller.snapshot()).encode())
            except (ValueError, TypeError) as error:
                self.reply(400, json.dumps({"error": str(error)}).encode())
            except Exception as error:
                self.reply(502, json.dumps({"error": str(error)}).encode())

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true", help="Print account checkpoints and exit")
    parser.add_argument("--left-url", default="http://127.0.0.1:8691")
    parser.add_argument("--right-url", default="http://127.0.0.1:8692")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--action-hz", type=float, default=20)
    parser.add_argument("--observation-hz", type=float, default=10)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--main-controller", action="store_true", help="Include Both and use direct SDK targets")
    args = parser.parse_args(argv)
    if args.list_models:
        print(json.dumps({"models": available_models()}))
        return
    if min(args.action_hz, args.observation_hz, args.width, args.height) <= 0:
        parser.error("rates and image dimensions must be positive")
    callers = {
        side: ArmCaller(
            side,
            ArmEndpoint(url),
            width=args.width,
            height=args.height,
            hz=args.action_hz,
            observation_hz=args.observation_hz,
        )
        for side, url in (("left", args.left_url), ("right", args.right_url))
    }
    if args.main_controller:
        endpoints = {side: caller.endpoint for side, caller in callers.items()}
        callers = {
            selection: StationCaller(
                selection,
                endpoints,
                width=args.width,
                height=args.height,
                hz=args.action_hz,
                observation_hz=args.observation_hz,
            )
            for selection in ("left", "right", "both")
        }
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(callers)) as server:
        server.daemon_threads = True
        print(f"Dream-1 caller: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            for caller in callers.values():
                caller.close()


if __name__ == "__main__":
    main()
