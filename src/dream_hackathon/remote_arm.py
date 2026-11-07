"""SO-101 follower and camera adapters for the unified server; never open CAN or USB cameras."""

import json
import urllib.error
import urllib.request

import cv2
import numpy as np


def call(base, path, body=None):
    request = urllib.request.Request(
        base + path, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        payload = error.read().decode()
        raise RuntimeError(payload) from error


class RemoteArms:
    """The server owns the arm; closing a leader adapter leaves its power/held target unchanged."""

    energized = True

    def __init__(self, base, side):
        self.base, self.side = base, side
        state = call(base, "/power", {"enabled": True})
        self.start_joints = {side: np.asarray(state["start_joints"], dtype=float)}

    def read(self):
        state = call(self.base, "/state")
        if not state["arm_open"]:
            raise RuntimeError(state.get("message") or "Arm is off")
        return {self.side: np.asarray(state["q"], dtype=float)}, {self.side: float(state["grip"])}

    def command(self, joints, grips):
        call(self.base, "/move", {"q": joints[self.side].tolist(), "grip": grips[self.side], "rate": 0.6})

    def set_hold_torque(self, torque):
        call(self.base, "/grip-torque", {"torque_nm": torque})

    def gripper_feedback(self):
        state = call(self.base, "/state")
        return {k: state[k] for k in (
            "grip_measured_torque_nm", "grip_applied_torque_nm", "grip_holding",
        )}

    @staticmethod
    def disable(base):
        return call(base, "/power", {"enabled": False})["problems"]

    def power_off(self):
        return self.disable(self.base)

    def close(self):
        pass


def remote_cameras(base):
    """Borrow server JPEG streams for the existing recorder, rather than reopening physical cameras."""
    from dream_hackathon import cameras

    class RemoteCamera:
        def __init__(self, path, width, height):
            self.path, self.size = path, (width, height)
            self.latest()

        def latest(self):
            try:
                with urllib.request.urlopen(base + self.path, timeout=5) as response:
                    frame = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Invalid camera JPEG")
                return cv2.resize(frame, self.size)
            except (OSError, ValueError) as error:
                raise cameras.CameraError(f"Server camera {self.path}: {error}") from error

        def close(self):
            pass

    def open_cameras(configs, width, height):
        return {cfg.role: RemoteCamera(
            "/overhead.jpg" if cfg.role == "overhead" else "/wrist.jpg", width, height,
        ) for cfg in configs}

    return open_cameras
