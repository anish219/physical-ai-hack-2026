"""Read the single Star Arm 102 leader and compute its six-axis wrist pose.

Encoder signs and joint transforms follow Fashion Star's ROS2 driver/model:
https://github.com/servodevelop/Star-Arm-102/tree/main/ros2-humble/src
The reference is link6's origin, with no inferred handle/TCP extension.
"""

import numpy as np

from dream_hackathon.so101 import base_rotation

JOINTS = (
    ((0, 0, 0.0283500000000012), (0, 0, 0)),
    ((0.000100000000001007, 0, 0.0620250000000017), (-1.5707963267949, 0, 3.14159265358979)),
    ((0.120565306190578, -0.0100999999999999, 0.0259999999999978), (0, 0, -1.5707963267949)),
    ((0.0162194771204603, -0.107604056191361, -0.0259999999999976), (0, 0, -1.5707963267949)),
    ((0.0350000000000005, -0.0013799999999997, 0), (-1.57079632679489, 0, 0)),
    ((0.15168, 0, 0.005225), (-1.5708, 0, -1.5708)),
)


class StarFK:
    """URDF origin then local -Z rotation for IDs 0–5; ID 6 is independent."""

    def __init__(self):
        self.origins = []
        for xyz, rpy in JOINTS:
            transform = np.eye(4)
            transform[:3, :3] = base_rotation(np.degrees(rpy))
            transform[:3, 3] = xyz
            self.origins.append(transform)

    def pose(self, action):
        transform = np.eye(4)
        for i, origin in enumerate(self.origins):
            turn = np.eye(4)
            turn[:3, :3] = base_rotation([0, 0, -action[f"joint{i + 1}.pos"]])
            transform = transform @ origin @ turn
        return transform


class StarLeader:
    """Only query encoder feedback; preserve origins, multi-turn counts and torque."""

    def __init__(self, port):
        self.port = port
        self.bus = None

    @property
    def is_connected(self):
        return self.bus is not None

    def connect(self, calibrate=False):
        from motorbridge_smart_servo import FashionStarServo

        self.bus = FashionStarServo(self.port, baudrate=1_000_000)

    def get_action(self):
        angles = [self.bus.read_raw_angle(i, multi_turn=False) for i in range(7)]
        action = {f"joint{i + 1}.pos": angle for i, angle in enumerate(angles[:6])}
        # Manufacturer's lever range: 0° closed to 90° open; existing worker expects percent.
        action["gripper.pos"] = float(np.clip(angles[6] / 90, 0, 1) * 100)
        return action

    def disconnect(self):
        self.bus.close()
        self.bus = None
