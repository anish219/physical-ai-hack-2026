"""One station page: independent arm servers and assignable SO-101 leader control."""

import argparse
import copy
import json
import os
import pathlib
import queue
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
from dream_hackathon.sequences import SequenceManager
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
                        "--server-gripper-torque",
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
    def __init__(self, nodes, leader, callers=None, *, sequence_path=None):
        self.nodes, self.leader = nodes, leader
        self.callers = callers or {}
        self.inputs = {side: "so101" if leader.side == side else None for side in nodes}
        self.input_lock = threading.RLock()
        self.combined_input = None
        self.shared_lock = threading.RLock()
        self.listeners = set()
        self.draft_revision = 0
        self.api_drafts = {
            side: {"task": "", "checkpoint": "dreamlabs/dream-1-beta", "revision": 0}
            for side in ("left", "right", "both")
        }
        self.sequences = SequenceManager(self.callers, nodes, self.publish, sequence_path)

    def shared_state(self):
        with self.shared_lock:
            return {"api_drafts": copy.deepcopy(self.api_drafts), "draft_revision": self.draft_revision}

    def publish(self, event):
        with self.shared_lock:
            for listener in self.listeners:
                listener.put(event)

    def update_draft(self, selection, values):
        if selection not in self.api_drafts:
            raise ValueError("Choose left, right, or both for the API")
        fields = {key: values[key] for key in ("task", "checkpoint") if key in values}
        if any(not isinstance(value, str) for value in fields.values()):
            raise ValueError("Task and checkpoint must be text")
        with self.shared_lock:
            self.draft_revision += 1
            self.api_drafts[selection].update(fields, revision=self.draft_revision)
            event = dict(
                type="draft",
                selection=selection,
                fields=fields,
                revision=self.draft_revision,
                client_id=values.get("client_id", ""),
            )
            self.publish(event)
            return event

    def publish_selection(self):
        self.publish({"type": "selection", "inputs": dict(self.inputs), "combined_input": self.combined_input})

    def state(self):
        state = self.leader.state()
        state["assigned_side"] = self.leader.side
        state["arms"] = {side: node.snapshot() for side, node in self.nodes.items()}
        state["dream"] = {side: caller.snapshot() for side, caller in self.callers.items()}
        state["inputs"] = dict(self.inputs)
        state["combined_input"] = self.combined_input
        state.update(self.shared_state())
        state["sequences"] = self.sequences.snapshot()
        if self.leader.side in state["arms"]:
            state["arm_open"] = state["arms"][self.leader.side]["arm_open"]
            state["grip_torque_nm"] = state["arms"][self.leader.side]["grip_torque_nm"]
        return state

    def pause(self, sides):
        self.sequences.stop_sides(sides)
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
        if side not in self.nodes or source not in (None, "so101", "dream", "sequence"):
            raise ValueError("Choose SO-101, Dream API, Sequences, or no input for an arm")
        with self.input_lock:
            if self.combined_input is not None:
                self.pause(list(self.nodes))
                self.combined_input = None
                self.inputs = dict.fromkeys(self.nodes)
            if self.inputs[side] == source:
                self.publish_selection()
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
            self.publish_selection()
            return self.state()

    def select_both(self, source):
        if source not in (None, "dream", "sequence"):
            raise ValueError("Both supports Dream API and Sequences; Full Auto Control is not available yet")
        with self.input_lock:
            if source == self.combined_input:
                return self.state()
            self.pause(list(self.nodes))
            if self.leader.side is not None:
                self.leader.select(None)
            self.inputs = dict.fromkeys(self.nodes)
            self.combined_input = source
            self.publish_selection()
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
            if "grip_torque_nm" in body:
                self.nodes[self.leader.side].set_torque(body["grip_torque_nm"])
            return self.leader.action(path, body)

    def set_arm_torque(self, side, torque):
        if side not in self.nodes:
            raise ValueError("Unknown arm")
        with self.input_lock:
            self.nodes[side].set_torque(torque)
            return self.state()

    def dream_action(self, selection, action, body):
        if selection not in self.callers:
            raise ValueError("Choose left, right, or both for the API")
        caller = self.callers[selection]
        with self.input_lock:
            if action == "stop":
                caller.cancel_playback()
                for side in caller.sides:
                    if self.inputs[side] == "dream" or self.combined_input == "dream":
                        node = self.nodes[side]
                        if node.snapshot()["arm_open"]:
                            node.control.post("/stop", {})
            elif action == "connect":
                caller.connect()
            elif action in ("call", "run"):
                if action == "run":
                    selected = (
                        self.combined_input == "dream"
                        if selection == "both"
                        else (self.combined_input is None and self.inputs[selection] == "dream")
                    )
                    if not selected:
                        raise ValueError("Select Dream API as the input on each arm you want to run")
                    for other in self.callers.values():
                        if other is not caller and set(other.sides).intersection(caller.sides):
                            other.cancel_playback()
                    for side in caller.sides:
                        self.nodes[side].cancel_home()
                if "task" in body or "checkpoint" in body:
                    self.update_draft(selection, body)
                with self.shared_lock:
                    values = dict(self.api_drafts[selection])
                caller.start(values, run=action == "run")
            else:
                raise ValueError("Unknown Dream API action")

    def sequence_action(self, selection, action, body):
        if selection not in self.sequences.runs:
            raise ValueError("Choose left, right, or both for Sequences")
        with self.input_lock:
            if action == "save":
                self.sequences.update(selection, body.get("definition"))
            elif action == "stop":
                self.sequences.runs[selection].stop()
            elif action == "run":
                selected = (
                    self.combined_input == "sequence"
                    if selection == "both"
                    else (self.combined_input is None and self.inputs[selection] == "sequence")
                )
                if not selected:
                    raise ValueError("Select Sequences as the input for this arm selection")
                if self.sequences.runs[selection].snapshot()["busy"]:
                    raise ValueError("This sequence is already running or finishing a call")
                self.pause(self.sequences.runs[selection].sides)
                self.sequences.start(selection)
            else:
                raise ValueError("Unknown sequence action")

    def power(self, side, enabled):
        if side not in self.nodes:
            raise ValueError("Unknown arm")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be true or false")
        if not enabled:
            self.sequences.stop_sides([side])
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
        self.closed = threading.Event()

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
                elif self.path in ("/controller.js", "/sequences.js"):
                    self.reply(
                        200,
                        pathlib.Path(__file__)
                        .with_name("unified_control.js" if self.path == "/controller.js" else "sequences.js")
                        .read_bytes(),
                        "text/javascript; charset=utf-8",
                    )
                elif self.path == "/api/state":
                    self.reply(200, json.dumps(controller.state()).encode())
                elif self.path == "/api/events":
                    listener = queue.Queue()
                    with controller.shared_lock:
                        controller.listeners.add(listener)
                        listener.put({"type": "drafts", **controller.shared_state()})
                        listener.put({"type": "sequences", "sequences": controller.sequences.snapshot()})
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        while not owner.closed.is_set():
                            try:
                                event = listener.get(timeout=15)
                            except queue.Empty:
                                event = None
                            data = (
                                b": keepalive\n\n"
                                if event is None
                                else ("data: " + json.dumps(event) + "\n\n").encode()
                            )
                            self.wfile.write(data)
                            self.wfile.flush()
                    except (OSError, ConnectionError):
                        pass
                    finally:
                        with controller.shared_lock:
                            controller.listeners.discard(listener)
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
                    elif self.path == "/api/both/input":
                        controller.select_both(body.get("source"))
                    elif self.path.startswith("/api/arms/") and self.path.endswith("/input"):
                        controller.select_input(self.path.split("/")[3], body.get("source"))
                    elif self.path.startswith("/api/arms/") and self.path.endswith("/grip-torque"):
                        controller.set_arm_torque(self.path.split("/")[3], body.get("torque_nm"))
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
                        if parts[3] == "draft":
                            event = controller.update_draft(parts[2], body)
                            self.reply(200, json.dumps(event).encode())
                            return
                        controller.dream_action(parts[2], parts[3], body)
                    elif self.path.startswith("/api/sequences/"):
                        parts = self.path.strip("/").split("/")
                        if len(parts) != 4:
                            raise ValueError("Choose a sequence and action")
                        controller.sequence_action(parts[2], parts[3], body)
                        if parts[3] == "save":
                            self.reply(200, json.dumps({"sequences": controller.sequences.snapshot()}).encode())
                            return
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
        self.server.daemon_threads = True
        self.port = self.server.server_port
        self.thread = threading.Thread(target=owner.server.serve_forever, name="unified-page", daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set()
        self.controller.sequences.close()
        self.controller.publish(None)
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
    parser.add_argument(
        "--sequence-file", type=pathlib.Path, default=pathlib.Path("~/.cache/dream_hackathon/sequences.json")
    )
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
        controller = UnifiedController(nodes, leader, callers, sequence_path=args.sequence_file)
        page = UnifiedPage(controller, args.port)
        try:
            print(f"Unified arm controller: http://127.0.0.1:{page.port}; SO-101 assignment: None", flush=True)
            for side in args.enable:
                state = nodes[side].power(True)
                print(f"{side}: {state['message']}", flush=True)
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            controller.sequences.close()
            dream.close()
            leader.close()
            page.close()
    finally:
        for node in nodes.values():
            node.close()


if __name__ == "__main__":
    main()
