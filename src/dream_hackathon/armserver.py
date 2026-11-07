"""Independent Metal arm HTTP controllers, compatible with pah/armserver.py.

Each selected arm has its own listener, motor loop, state and ownership. CameraHub
shares a physical camera only between listeners in this process. Vendor imports
and all hardware opening happen in main/run_arm, never when importing this module.
"""

import argparse
import dataclasses
import json
import pathlib
import subprocess
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np

TICK, MAX_RATE, ABORT_RAD, STALL_RAD = 1 / 30, 0.6, 0.3, 0.012
STATE_KEYS = ("q", "grip", "goal", "goal_grip", "rate", "moving", "stalled", "lag", "stamp", "cams", "error")


class ArmControl:
    """One arm's existing command semantics and motion algorithm, with no cross-arm state."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = {
            "q": None,
            "grip": None,
            "goal": None,
            "goal_grip": 0.0,
            "rate": 0.3,
            "moving": False,
            "stalled": False,
            "stall_watch": False,
            "lag": 0.0,
            "stamp": 0.0,
            "cams": {},
            "quit": False,
            "error": None,
        }
        self.frames: dict[str, tuple[bytes, Any, Any, float]] = {}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {k: dict(self.state[k]) if k == "cams" else self.state[k] for k in STATE_KEYS}

    def post(self, path: str, body: dict[str, Any]) -> None:
        with self.lock:
            st = self.state
            if path == "/move":
                if body.get("q") is not None:
                    st["goal"] = [float(v) for v in body["q"]]
                if body.get("grip") is not None:
                    st["goal_grip"] = float(np.clip(body["grip"], 0, 1))
                st["rate"] = float(body.get("rate", 0.3))
                st["stalled"] = False
                st["error"] = None
                st["stall_watch"] = bool(body.get("stop_on_stall", False))
            elif path == "/stop":
                st["goal"] = list(st["q"])
            elif path == "/quit":
                st["quit"] = True

    def camera_status(self, name: str, opened: bool) -> None:
        with self.lock:
            self.state["cams"][name] = opened

    def camera_frame(self, name: str, jpeg: bytes, stamp: float) -> None:
        with self.lock:
            self.frames[name] = (jpeg, self.state["q"], self.state["grip"], stamp)

    def control(
        self,
        arm: Any,
        command: Callable[..., Any],
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Preserves the original 30 Hz stepping, gripper slew, lag/stall handling and quit return."""
        cmd = np.array(arm.read().positions_rad, dtype=float)
        with self.lock:
            self.state["q"] = cmd.tolist()
            self.state["goal"] = cmd.tolist()
            grip_cmd = self.state["goal_grip"]
        lags, start = [], cmd.copy()
        while True:
            with self.lock:
                st = self.state
                goal, goal_grip, rate = np.array(st["goal"]), st["goal_grip"], min(st["rate"], MAX_RATE)
                quitting = st["quit"]
            if quitting:
                goal, rate = start, 0.3
            d = goal - cmd
            step = rate * TICK
            n = np.max(np.abs(d))
            cmd = goal.copy() if n <= step else cmd + d * (step / n)
            grip_cmd += float(np.clip(goal_grip - grip_cmd, -1.5 * TICK, 1.5 * TICK))
            # No HTTP or camera mutex is held across vendor I/O or sleep.
            arm.write(command(positions_rad=cmd, gripper=grip_cmd))
            sleep(TICK)
            reading = arm.read()
            meas = np.array(reading.positions_rad, dtype=float)
            lag = float(np.max(np.abs(cmd - meas)))
            with self.lock:
                st.update(
                    q=meas.tolist(),
                    grip=float(getattr(reading, "gripper", grip_cmd)),
                    lag=lag,
                    stamp=clock(),
                    moving=bool(np.max(np.abs(goal - cmd)) > 1e-4 or abs(goal_grip - grip_cmd) > 1e-3),
                )
                if st["quit"] and not st["moving"]:
                    return
                if lag > ABORT_RAD:
                    st["goal"] = meas.tolist()
                    st["error"] = f"joint lag {lag:.2f} rad: stopped where the arm is"
                    cmd = meas.copy()
                    continue
                if st["stall_watch"] and st["moving"]:
                    lags.append(float(np.max(np.abs(cmd[1:4] - meas[1:4]))))
                    if len(lags) > 10 and lags[-1] > max(lags[:10]) + STALL_RAD:
                        st["goal"] = meas.tolist()
                        st["stalled"] = True
                        st["stall_watch"] = False
                        cmd = meas.copy()
                else:
                    lags = []


def handler_for(control: ArmControl) -> type[BaseHTTPRequestHandler]:
    """The legacy API, bound to exactly one arm and camera metadata namespace."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def reply(self, code: int, body: bytes, ctype: str = "application/json", headers: tuple = ()) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            name = self.path.strip("/").split(".")[0]
            if name in ("wrist", "overhead"):
                with control.lock:
                    frame = control.frames.get(name)
                if frame is None:
                    return self.reply(503, b'{"error": "no frame"}')
                jpg, q, grip, stamp = frame
                return self.reply(
                    200,
                    jpg,
                    "image/jpeg",
                    (
                        ("X-Joints", json.dumps(q)),
                        ("X-Grip", json.dumps(grip)),
                        ("X-Stamp", f"{stamp:.3f}"),
                    ),
                )
            self.reply(200, json.dumps(control.snapshot()).encode())

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            control.post(self.path, body)
            self.reply(200, json.dumps({"ok": True}).encode())

    return Handler


class CameraStream:
    """One legacy camera capture loop; frames fan out with each arm's own measured pose."""

    def __init__(self, device: str, capture: Callable[..., Any]) -> None:
        self.device, self.capture = device, capture
        self.lock = threading.Lock()
        self.subscribers: list[tuple[ArmControl, str]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"camera:{device}", daemon=True)

    def publish(self, opened: bool, jpeg: bytes | None = None) -> None:
        with self.lock:
            subscribers = list(self.subscribers)
        stamp = time.time()
        for control, name in subscribers:
            control.camera_status(name, opened)
            if jpeg is not None:
                control.camera_frame(name, jpeg, stamp)

    def run(self) -> None:
        while not self.stop.is_set():
            cap = self.capture(self.device, cv2.CAP_V4L2)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.publish(cap.isOpened())
                while cap.isOpened() and not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                    self.publish(True, jpg)
            finally:
                cap.release()
                self.publish(False)
            self.stop.wait(1.0)


class CameraHub:
    """Shares only camera capture, never motor state, arm locks or control-loop scheduling."""

    def __init__(self, capture: Callable[..., Any] = cv2.VideoCapture) -> None:
        self.capture = capture
        self.lock = threading.Lock()
        self.streams: dict[str, CameraStream] = {}

    def subscribe(self, control: ArmControl, cameras: dict[str, str]) -> None:
        with self.lock:
            for name, device in cameras.items():
                stream = self.streams.get(device)
                new = stream is None
                if new:
                    stream = self.streams[device] = CameraStream(device, self.capture)
                with stream.lock:
                    stream.subscribers.append((control, name))
                if new:
                    stream.thread.start()

    def unsubscribe(self, control: ArmControl) -> None:
        with self.lock:
            for device, stream in list(self.streams.items()):
                with stream.lock:
                    stream.subscribers = [(c, n) for c, n in stream.subscribers if c is not control]
                    empty = not stream.subscribers
                if empty:
                    stream.stop.set()
                    del self.streams[device]


@dataclasses.dataclass
class ArmBinding:
    side: str
    config: Any
    wrist: str
    overhead: str
    lock_path: pathlib.Path


def acquire_arm(binding: ArmBinding, share: bool) -> tuple[Any, bool]:
    """Normal per-arm ownership, plus the existing explicit --share compatibility behavior."""
    import os

    from dream_hackathon import station_lock

    label = f"pah arm server ({binding.side} arm only)"
    try:
        ownership = station_lock.acquire_sides(label, [binding.side], path=binding.lock_path)
    except station_lock.StationBusyError:
        if not share:
            raise
        # Preserve --share for an older whole-station owner using only the opposite arm.
        busy = subprocess.run(
            [
                "fuser",
                str(pathlib.Path(binding.config.can_channel).resolve()),
                str(pathlib.Path(binding.wrist).resolve()),
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if busy:
            raise
        return station_lock.acquire(label, path=binding.lock_path.with_name(f"arms.{binding.side}.lock")), True
    others = share and any(h.get("pid") != str(os.getpid()) for h in station_lock.holders(binding.lock_path))
    return ownership, bool(others)


def run_arm(
    binding: ArmBinding,
    control: ArmControl,
    hub: CameraHub,
    port: int,
    *,
    no_overhead: bool,
    share: bool,
    acquire: Callable = acquire_arm,
    open_arm: Callable,
    command: Callable,
    sleep: Callable = time.sleep,
) -> None:
    """A complete legacy arm-server lifetime, independent of every other arm's lifetime."""
    ownership, sharing = acquire(binding, share)
    with ownership, ThreadingHTTPServer(("127.0.0.1", port), handler_for(control)) as server:
        server.daemon_threads = True
        sleep(1.5)  # retain live.py's existing overhead-camera handoff interval
        cameras = {"wrist": binding.wrist}
        if not no_overhead and not sharing:
            cameras["overhead"] = binding.overhead
        hub.subscribe(control, cameras)
        try:
            arm = open_arm(binding)
            try:
                http = threading.Thread(target=server.serve_forever, name=f"http:{binding.side}", daemon=True)
                http.start()
                print(f"{binding.side} arm server on 127.0.0.1:{server.server_port}", flush=True)
                try:
                    control.control(arm, command)
                finally:
                    arm.close()  # same vendor park and torque-release sequence
            finally:
                if "http" in locals():
                    server.shutdown()
                    http.join()
        finally:
            hub.unsubscribe(control)
            with control.lock:
                control.state["quit"] = True
    print(f"{binding.side} released", flush=True)


def load_bindings(path: pathlib.Path, sides: tuple[str, ...]) -> list[ArmBinding]:
    """Use Dream's exact saved arm calibration and camera assignments, not copied constants."""
    from robots import config

    station = config.load(path)
    if station.body != "metal":
        raise ValueError("armserver expects a Metal station")
    cameras = {camera.name: camera.device for camera in station.cameras}
    return [
        ArmBinding(
            side,
            getattr(station, "follower_l" if side == "left" else "follower_r"),
            cameras["wrist_l" if side == "left" else "wrist_r"],
            cameras["scene"],
            pathlib.Path("~/.dreamlabs/arms.lock").expanduser(),
        )
        for side in sides
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right", "both"), default="left")
    parser.add_argument("--port", type=int, help="single-arm port, or left port in Both mode (left 8691, right 8692)")
    parser.add_argument("--right-port", type=int, default=8692, help="right port in Both mode")
    parser.add_argument("--no-overhead", action="store_true")
    parser.add_argument("--share", action="store_true", help="preserve legacy shared-operation, wrist-only behavior")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("~/.dreamlabs/station.yaml"))
    args = parser.parse_args(argv)
    sides = ("left", "right") if args.side == "both" else (args.side,)
    ports = {
        "left": args.port if args.port is not None else 8691,
        "right": args.right_port if args.side == "both" else (args.port if args.port is not None else 8692),
    }
    if len(sides) == 2 and ports["left"] == ports["right"]:
        parser.error("left and right listeners need distinct ports")
    bindings = load_bindings(args.config.expanduser(), sides)
    from robots import metal, motor_arm, robstride, types

    def open_arm(binding: ArmBinding) -> Any:
        return motor_arm.open_arm(
            metal.ARM, binding.config, binding.side, opener=robstride.open_can_bus, energize=True
        )

    hub, workers, controls, failures = CameraHub(), [], [], []

    def worker(binding: ArmBinding, control: ArmControl) -> None:
        try:
            run_arm(
                binding,
                control,
                hub,
                ports[binding.side],
                no_overhead=args.no_overhead,
                share=args.share,
                open_arm=open_arm,
                command=types.ArmCommand,
            )
        except Exception as error:
            failures.append((binding.side, error))
            print(f"{binding.side} arm server failed: {error}", flush=True)

    for binding in bindings:
        control = ArmControl()
        controls.append(control)
        thread = threading.Thread(target=worker, args=(binding, control), name=f"arm:{binding.side}")
        workers.append(thread)
        thread.start()
    try:
        for thread in workers:
            thread.join()
    except KeyboardInterrupt:
        for control in controls:
            control.post("/quit", {})
        for thread in workers:
            thread.join()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
