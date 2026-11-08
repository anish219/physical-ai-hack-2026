"""Persistent per-arm HTTP listeners with explicit on/off and server-owned CAN adapters."""

import dataclasses
import json
import pathlib
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np

from dream_hackathon.armserver import ArmControl, handler_for


class PowerOffRequestedError(Exception):
    """The operator explicitly turned this arm off."""


class ManagedArm:
    def __init__(self, station, side, binding, hub, port, *, acquire=None, opener=None, kill=None, zero_path=None):
        self.station = dataclasses.replace(station, arms={side: station.arms[side]})
        self.side, self.binding, self.hub = side, binding, hub
        self.control = ArmControl()
        self.lock, self.requests = threading.Lock(), threading.Lock()
        self.wake, self.completed = threading.Event(), threading.Event()
        self.stopping = False
        self.desired = self.opened = False
        self.message = ""
        self.problems = []
        self.start_joints = None
        self.zero_path = pathlib.Path(zero_path or f"~/.cache/dream_hackathon/zero/{side}.json").expanduser()
        self.zero_joints = json.loads(self.zero_path.read_text()) if self.zero_path.exists() else None
        self.home_cancel = threading.Event()
        self.home_lock = threading.Lock()
        self.homing = False
        self.home_worker = None
        self.torque = 0.5
        self.feedback = dict(grip_measured_torque_nm=None, grip_applied_torque_nm=None, grip_holding=False)
        self.acquire = acquire or self._acquire
        self.opener = opener or self._open
        self.kill = kill or self._kill
        self.server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.server.daemon_threads = True
        self.port = self.server.server_port
        self.http = threading.Thread(target=self.server.serve_forever, name=f"arm-http:{side}", daemon=True)
        self.worker = threading.Thread(target=self._run, name=f"arm-motor:{side}", daemon=True)
        self.http.start()
        self.worker.start()

    def _acquire(self):
        from dream_hackathon import station_lock

        return station_lock.acquire_sides(f"unified arm server {self.side}", [self.side])

    def _open(self):
        from dream_hackathon.arms.makermods import MakerModsArms

        return MakerModsArms(self.station, energize=True)

    def _kill(self):
        from dream_hackathon.so101_power import kill_controlled_arm

        return kill_controlled_arm(self.station, self.side)

    def snapshot(self):
        state = self.control.snapshot()
        with self.control.lock:
            state["target_mode"] = "direct" if self.control.direct_target else "stepped"
        with self.lock:
            state.update(
                arm_open=self.opened,
                enabled=self.opened,
                requested_power=self.desired,
                message=self.message,
                problems=list(self.problems),
                start_joints=self.start_joints,
                zero_joints=self.zero_joints,
                homing=self.homing,
                grip_torque_nm=self.torque,
                **self.feedback,
            )
        return state

    def power(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be true or false")
        if not enabled:
            self.cancel_home()
        # Same-arm requests are ordered; the other arm has an entirely separate worker and request mutex.
        with self.requests:
            with self.lock:
                if enabled and self.opened:
                    return self.snapshot_unlocked()
                self.desired = enabled
                self.problems = []
                self.message = "Opening arm" if enabled else "Disabling arm torque"
            self.completed.clear()
            self.wake.set()
            self.completed.wait()
            return self.snapshot()

    def snapshot_unlocked(self):
        # Called with only self.lock held; do not acquire the control lock in reverse order.
        return dict(
            arm_open=self.opened,
            enabled=self.opened,
            requested_power=self.desired,
            start_joints=self.start_joints,
            problems=list(self.problems),
            message=self.message,
        )

    def set_torque(self, torque):
        if isinstance(torque, bool) or not isinstance(torque, (int, float)) or not 0.1 <= torque <= 2:
            raise ValueError("Grip holding torque must be between 0.1 and 2.0 N·m")
        with self.lock:
            self.torque = float(torque)

    def save_zero(self, joints):
        self.zero_path.parent.mkdir(parents=True, exist_ok=True)
        self.zero_path.write_text(json.dumps(list(joints)))
        with self.lock:
            self.zero_joints = list(joints)

    def set_zero(self):
        if not self.snapshot()["arm_open"]:
            raise ValueError("Turn on the arm to set zero from its current measured pose")
        joints = self.control.snapshot()["q"]
        if joints is None:
            raise ValueError("Turn on the arm once to measure its pose")
        self.save_zero(joints)
        return self.snapshot()

    def cancel_home(self):
        self.home_cancel.set()
        with self.home_lock:
            pass
        if self.home_worker is not None:
            self.home_worker.join()
        with self.lock:
            self.homing = False

    def return_zero(self):
        from dream_hackathon.teleop import home_step

        self.cancel_home()
        result = self.power(True)
        if result["problems"]:
            raise RuntimeError("; ".join(result["problems"]))
        state = self.control.snapshot()
        with self.lock:
            goal = np.asarray(self.zero_joints, dtype=float)
            self.homing = True
            self.message = "Returning to saved zero"
        cancel = self.home_cancel = threading.Event()

        def run():
            current = np.asarray(state["q"], dtype=float)
            while not cancel.is_set():
                with self.home_lock:
                    if cancel.is_set():
                        break
                    current, arrived = home_step(current, goal, 1 / 30)
                    self.control.target(current, state["goal_grip"])
                if arrived:
                    with self.lock:
                        self.message = "Returned to saved zero"
                    break
                cancel.wait(1 / 30)
            with self.lock:
                self.homing = False

        self.home_worker = threading.Thread(target=run, name=f"return-zero:{self.side}", daemon=True)
        self.home_worker.start()
        return self.snapshot()

    def _bridge(self, robot):
        node = self

        class Bridge:
            applied = None
            ready_sent = False

            def read(self):
                from dream_hackathon.so101_gripper import feedback

                joints, grips = robot.read()
                measured_feedback = feedback(robot, node.side)
                with node.lock:
                    node.feedback = measured_feedback
                return SimpleNamespace(positions_rad=joints[node.side], gripper=grips[node.side])

            def write(self, command):
                from dream_hackathon.so101_gripper import apply_hold_torque

                with node.lock:
                    wanted, torque = node.desired, node.torque
                if not wanted:
                    raise PowerOffRequestedError
                if torque != self.applied:
                    apply_hold_torque(robot, node.side, torque)
                    self.applied = torque
                robot.command({node.side: command.positions_rad}, {node.side: command.gripper})
                if not self.ready_sent:
                    with node.lock:
                        node.opened = True
                        node.message = "Arm on; holding position"
                    self.ready_sent = True
                    node.completed.set()

        return Bridge()

    def _run(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            if self.stopping:
                return
            with self.lock:
                wanted = self.desired
            robot = None
            problems = []
            try:
                with self.acquire():
                    if not wanted:
                        problems = self.kill()
                    else:
                        robot = self.opener()
                        try:
                            joints, grips = robot.read()
                            if self.zero_joints is None:
                                self.save_zero(joints[self.side].tolist())
                            with self.control.lock:
                                self.control.direct_target = False
                                self.control.state.update(
                                    q=joints[self.side].tolist(),
                                    goal=joints[self.side].tolist(),
                                    grip=grips[self.side],
                                    goal_grip=grips[self.side],
                                    quit=False,
                                    error=None,
                                    moving=False,
                                    stalled=False,
                                )
                            with self.lock:
                                self.start_joints = robot.start_joints[self.side].tolist()
                            self.hub.subscribe(
                                self.control, {"wrist": self.binding.wrist, "overhead": self.binding.overhead}
                            )
                            self.control.control(self._bridge(robot), SimpleNamespace)
                        except PowerOffRequestedError:
                            problems = robot.power_off()
                        finally:
                            try:
                                robot.close()
                            finally:
                                self.hub.unsubscribe(self.control)
            except Exception as error:
                problems.append(str(error))
            finally:
                with self.lock:
                    self.opened = self.desired = False
                    self.problems = problems
                    self.message = "; ".join(problems) if problems else "Arm off"
                    self.feedback = dict(grip_measured_torque_nm=None, grip_applied_torque_nm=None, grip_holding=False)
                self.wake.clear()
                self.completed.set()

    def _handler(self):
        node = self
        base = handler_for(self.control)

        class Handler(base):
            def do_GET(self):  # noqa: N802
                if self.path == "/state":
                    self.reply(200, json.dumps(node.snapshot()).encode())
                else:
                    super().do_GET()

            def do_POST(self):  # noqa: N802
                try:
                    if self.path in ("/zero", "/home"):
                        state = node.set_zero() if self.path == "/zero" else node.return_zero()
                        self.reply(200, json.dumps(state).encode())
                    elif self.path in ("/power", "/grip-torque"):
                        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                        if self.path == "/power":
                            state = node.power(body.get("enabled"))
                            self.reply(500 if state["problems"] else 200, json.dumps(state).encode())
                        else:
                            node.set_torque(body.get("torque_nm"))
                            self.reply(200, json.dumps(node.snapshot()).encode())
                    elif self.path in ("/move", "/target") and not node.snapshot()["arm_open"]:
                        self.reply(409, b'{"error": "Arm is off"}')
                    elif self.path == "/target":
                        node.cancel_home()
                        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                        node.control.target(body["q"], body["grip"])
                        self.reply(200, b'{"ok": true}')
                    else:
                        if self.path in ("/move", "/stop", "/quit"):
                            node.cancel_home()
                        super().do_POST()
                except (ValueError, TypeError) as error:
                    self.reply(400, json.dumps({"error": str(error)}).encode())

        return Handler

    def close(self):
        self.cancel_home()
        self.stopping = True
        if self.opened:
            self.control.post("/quit", {})
            while self.opened:
                time.sleep(0.01)
        self.wake.set()
        self.worker.join()
        self.server.shutdown()
        self.server.server_close()
        self.http.join()
