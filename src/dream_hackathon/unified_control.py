"""One station page: independent arm servers and assignable SO-101 leader control."""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dream_hackathon.armserver import ArmBinding, CameraHub
from dream_hackathon.remote_arm import call
from dream_hackathon.so101_dashboard import Controls


class LeaderAssignment:
    """Only the leader child changes when assignment changes; follower ownership stays in the servers."""

    def __init__(
        self,
        config_path,
        leader_port,
        arm_urls,
        dataset_root,
        *,
        port=8083,
        scale=1.0,
        torque=0.5,
        log_dir=pathlib.Path("/tmp"),
    ):
        self.config_path, self.leader_port = config_path, leader_port
        self.arm_urls, self.dataset_root = arm_urls, dataset_root
        self.port = port
        self.log_dir = log_dir
        self.side = None
        self.process = None
        self.lock = threading.Lock()
        self.settings = {side: dict(scale=scale, grip_torque_nm=torque) for side in arm_urls}
        self.message = "SO-101 is not assigned"

    def state(self):
        side, process = self.side, self.process
        if side is not None and process is not None and process.poll() is None:
            try:
                return call(f"http://127.0.0.1:{self.port}", "/api/state", timeout=1)
            except (OSError, RuntimeError) as error:
                self.message = str(error)
        state = Controls(1.0, "none").snapshot()
        state["message"] = self.message
        return state

    def action(self, path, body):
        if self.side is None or self.process is None or self.process.poll() is not None:
            raise ValueError("Select an arm for SO-101 first")
        return call(f"http://127.0.0.1:{self.port}", path, body)

    def select(self, side):
        if side not in (None, "left", "right"):
            raise ValueError("SO-101 assignment must be left, right, or null")
        with self.lock:
            if side == self.side and (side is None or self.process.poll() is None):
                return
            previous = self.state()
            if self.side is not None:
                self.settings[self.side] = {k: previous[k] for k in ("scale", "grip_torque_nm")}
            if self.process is not None and self.process.poll() is None:
                # Existing child cleanup saves any current episode and disconnects the leader.
                # RemoteArms.close does not home or disable either follower.
                self.process.send_signal(signal.SIGINT)
                self.process.wait()
            self.process, self.side = None, None
            if side is None:
                self.message = "SO-101 is not assigned; powered arms hold their current targets"
                return
            settings = self.settings[side]
            log_path = self.log_dir / f"unified-so101-{side}.log"
            with log_path.open("ab") as log:
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "dream_hackathon.so101",
                        "--config",
                        str(self.config_path),
                        "--leader-port",
                        self.leader_port,
                        "--side",
                        side,
                        "--remote-server",
                        self.arm_urls[side],
                        "--dashboard-port",
                        str(self.port),
                        "--dataset-dir",
                        str(self.dataset_root / side),
                        "--translation-scale",
                        str(settings["scale"]),
                        "--gripper-hold-torque-nm",
                        str(settings["grip_torque_nm"]),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            while self.process.poll() is None:
                try:
                    call(f"http://127.0.0.1:{self.port}", "/api/state", timeout=1)
                    self.side = side
                    self.message = "Untracked. Position the leader, then Track."
                    return
                except (OSError, RuntimeError):
                    time.sleep(0.05)
            raise RuntimeError(f"SO-101 did not start: {log_path.read_text()[-2000:]}")

    def close(self):
        self.select(None)


class DreamService:
    """Run Dream IK in the station's existing Dream environment, separate from the leader SDK."""

    def __init__(self, python, urls, port=8084):
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = pathlib.Path("/tmp/unified-dream-api.log")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(pathlib.Path(__file__).resolve().parent.parent), env.get("PYTHONPATH")))
        )
        with self.log_path.open("ab") as log:
            self.process = subprocess.Popen(
                [
                    str(python),
                    "-u",
                    "-m",
                    "dream_hackathon.dream_caller",
                    "--main-controller",
                    "--port",
                    str(port),
                    "--left-url",
                    urls["left"],
                    "--right-url",
                    urls["right"],
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        while self.process.poll() is None:
            try:
                call(self.url, "/api/state", timeout=1)
                return
            except (OSError, RuntimeError):
                time.sleep(0.05)
        raise RuntimeError(f"Dream worker did not start: {self.log_path.read_text()[-2000:]}")

    def close(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            self.process.wait()


class DreamCallerProxy:
    def __init__(self, service, selection):
        self.url, self.selection = service.url, selection
        self.sides = ("left", "right") if selection == "both" else (selection,)

    def snapshot(self):
        return call(self.url, f"/api/{self.selection}/state")

    def action(self, name, body=None):
        return call(self.url, f"/api/{self.selection}/{name}", body or {})

    def connect(self):
        self.action("connect")

    def start(self, body, *, run):
        self.action("run" if run else "call", body)

    def stop(self):
        self.action("stop")

    def cancel_playback(self):
        self.action("cancel")


class UnifiedController:
    def __init__(self, nodes, leader, callers=None):
        self.nodes, self.leader = nodes, leader
        self.callers = callers or {}
        self.inputs = {side: "so101" if leader.side == side else None for side in nodes}
        self.input_lock = threading.RLock()

    def state(self):
        state = self.leader.state()
        state["assigned_side"] = self.leader.side
        state["arms"] = {side: node.snapshot() for side, node in self.nodes.items()}
        state["dream"] = {side: caller.snapshot() for side, caller in self.callers.items()}
        state["inputs"] = dict(self.inputs)
        if self.leader.side in state["arms"]:
            state["arm_open"] = state["arms"][self.leader.side]["arm_open"]
        return state

    def pause(self, sides):
        for caller in self.callers.values():
            if set(sides).intersection(caller.sides):
                caller.cancel_playback()
        if self.leader.side in sides:
            self.leader.action("/api/control", {"tracking": False})
        for side in sides:
            self.nodes[side].cancel_home()
            state = self.nodes[side].snapshot()
            if state["arm_open"]:
                self.nodes[side].control.post("/stop", {})

    def select_input(self, side, source):
        if side not in self.nodes or source not in (None, "so101", "dream"):
            raise ValueError("Choose SO-101, Dream API, or no input for an arm")
        with self.input_lock:
            if self.inputs[side] == source:
                return self.state()
            self.pause([side])
            if source == "so101":
                previous = self.leader.side
                if previous is not None and previous != side:
                    self.pause([previous])
                    self.inputs[previous] = None
                self.leader.select(side)
            elif self.leader.side == side:
                self.leader.select(None)
            self.inputs[side] = source
            return self.state()

    def arm_home(self, side):
        with self.input_lock:
            self.pause([side])
            return self.nodes[side].return_zero()

    def leader_action(self, path, body):
        with self.input_lock:
            if self.leader.side is None or self.inputs[self.leader.side] != "so101":
                raise ValueError("Choose SO-101 as an arm's input first")
            if body.get("tracking"):
                self.nodes[self.leader.side].cancel_home()
            return self.leader.action(path, body)

    def dream_action(self, selection, action, body):
        if selection not in self.callers:
            raise ValueError("Choose left, right, or both for the API")
        caller = self.callers[selection]
        with self.input_lock:
            if action == "stop":
                caller.cancel_playback()
                for side in caller.sides:
                    if self.inputs[side] == "dream":
                        node = self.nodes[side]
                        if node.snapshot()["arm_open"]:
                            node.control.post("/stop", {})
            elif action == "connect":
                caller.connect()
            elif action in ("call", "run"):
                if action == "run":
                    if any(self.inputs[side] != "dream" for side in caller.sides):
                        raise ValueError("Select Dream API as the input on each arm you want to run")
                    for other in self.callers.values():
                        if other is not caller and set(other.sides).intersection(caller.sides):
                            other.cancel_playback()
                    for side in caller.sides:
                        self.nodes[side].cancel_home()
                caller.start(body, run=action == "run")
            else:
                raise ValueError("Unknown Dream API action")

    def power(self, side, enabled):
        if side not in self.nodes:
            raise ValueError("Unknown arm")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be true or false")
        if not enabled:
            # An explicit arm-off also cancels playback for that arm, including a pending API reply.
            for caller in self.callers.values():
                if side in caller.sides:
                    caller.cancel_playback()
        if not enabled and side == self.leader.side:
            self.leader.action("/api/kill", {})
        else:
            result = self.nodes[side].power(enabled)
            if result["problems"]:
                raise RuntimeError("; ".join(result["problems"]))
        return self.state()


class UnifiedPage:
    def __init__(self, controller, port=8081):
        self.controller = controller
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, code, body, content_type="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path == "/":
                    self.reply(200, PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/api/state":
                    self.reply(200, json.dumps(controller.state()).encode())
                elif self.path.startswith("/api/camera/"):
                    role = self.path.rsplit("/", 1)[-1].split(".")[0]
                    side = "left" if role in ("left", "overhead") else "right"
                    if role == "overhead" and not controller.nodes[side].snapshot()["arm_open"]:
                        side = "right"
                    path = "/overhead.jpg" if role == "overhead" else "/wrist.jpg"
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{controller.nodes[side].port}" + path, timeout=5
                        ) as response:
                            self.reply(200, response.read(), "image/jpeg")
                    except OSError:
                        self.reply(503, b"No camera frame", "text/plain")
                else:
                    self.reply(404, b"Not found", "text/plain")

            def do_POST(self):  # noqa: N802
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("expected an object")
                    if self.path == "/api/assignment":
                        side = body.get("side")
                        if side is not None:
                            controller.select_input(side, "so101")
                        elif controller.leader.side is not None:
                            controller.select_input(controller.leader.side, None)
                    elif self.path.startswith("/api/arms/") and self.path.endswith("/input"):
                        controller.select_input(self.path.split("/")[3], body.get("source"))
                    elif self.path.startswith("/api/arms/") and self.path.endswith(("/home", "/zero")):
                        side = self.path.split("/")[3]
                        if side not in controller.nodes:
                            raise ValueError("Unknown arm")
                        if self.path.endswith("/home"):
                            controller.arm_home(side)
                        else:
                            controller.nodes[side].set_zero()
                    elif self.path.startswith("/api/dream/"):
                        parts = self.path.strip("/").split("/")
                        if len(parts) != 4 or parts[2] not in controller.callers:
                            raise ValueError("Choose left, right, or both for the API")
                        controller.dream_action(parts[2], parts[3], body)
                    elif self.path in ("/api/arms/left/power", "/api/arms/right/power"):
                        controller.power(self.path.split("/")[3], body.get("enabled"))
                    elif self.path == "/api/home":
                        if controller.leader.side is None:
                            raise ValueError("Choose an arm for Return to 0")
                        controller.arm_home(controller.leader.side)
                    elif self.path == "/api/kill":
                        if controller.leader.side is None:
                            raise ValueError("Choose an arm to turn off")
                        controller.power(controller.leader.side, False)
                    elif self.path in ("/api/control", "/api/reject"):
                        controller.leader_action(self.path, body)
                    else:
                        self.reply(404, b"Not found", "text/plain")
                        return
                    self.reply(200, json.dumps(controller.state()).encode())
                except (ValueError, TypeError) as error:
                    self.reply(400, json.dumps({"error": str(error)}).encode())
                except (RuntimeError, OSError) as error:
                    self.reply(500, json.dumps({"error": str(error)}).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=owner.server.serve_forever, name="unified-page", daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


PAGE = pathlib.Path(__file__).with_name("unified_control.html").read_text(encoding="utf-8")


def main(argv=None):
    from dream_hackathon import config, station_lock
    from dream_hackathon.managed_arms import ManagedArm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/station.yaml"))
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--left-port", type=int, default=8691)
    parser.add_argument("--right-port", type=int, default=8692)
    parser.add_argument("--leader-dashboard-port", type=int, default=8083)
    parser.add_argument("--dream-port", type=int, default=8084)
    parser.add_argument("--dream-python", type=pathlib.Path, default=pathlib.Path("~/dream-env/bin/python"))
    parser.add_argument("--dataset-dir", type=pathlib.Path)
    parser.add_argument("--enable", nargs="*", choices=("left", "right"), default=[])
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--gripper-hold-torque-nm", type=float, default=0.5)
    args = parser.parse_args(argv)
    station = config.load(args.config)
    cameras = {camera.role: camera.device for camera in station.cameras}
    hub, nodes = CameraHub(), {}
    try:
        for side, port in (("left", args.left_port), ("right", args.right_port)):
            binding = ArmBinding(
                side,
                station.arms[side],
                cameras["wrist_l" if side == "left" else "wrist_r"],
                cameras["overhead"],
                station_lock.LOCK_PATH,
            )
            nodes[side] = ManagedArm(station, side, binding, hub, port)
            nodes[side].set_torque(args.gripper_hold_torque_nm)
        urls = {side: f"http://127.0.0.1:{node.port}" for side, node in nodes.items()}
        leader = LeaderAssignment(
            args.config.resolve(),
            args.leader_port,
            urls,
            (args.dataset_dir or station.dataset_dir / "unified" / "so101").expanduser(),
            port=args.leader_dashboard_port,
            scale=args.translation_scale,
            torque=args.gripper_hold_torque_nm,
        )
        dream = DreamService(args.dream_python.expanduser(), urls, args.dream_port)
        callers = {selection: DreamCallerProxy(dream, selection) for selection in ("left", "right", "both")}
        page = UnifiedPage(UnifiedController(nodes, leader, callers), args.port)
        try:
            print(f"Unified arm controller: http://127.0.0.1:{page.port}; SO-101 assignment: None", flush=True)
            for side in args.enable:
                state = nodes[side].power(True)
                print(f"{side}: {state['message']}", flush=True)
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            dream.close()
            leader.close()
            page.close()
    finally:
        for node in nodes.values():
            node.close()


if __name__ == "__main__":
    main()
