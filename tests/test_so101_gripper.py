"""Selected-gripper effort commands, telemetry, and installed SDK holding behavior."""

import copy
import threading
import types
import unittest
from unittest.mock import Mock

from dream_hackathon.so101_gripper import apply_hold_torque, feedback


class GripperTests(unittest.TestCase):
    def test_effort_update_addresses_only_controlled_gripper(self):
        right, left = Mock(), Mock()
        robot = types.SimpleNamespace(_devices={"right": right, "left": left})
        apply_hold_torque(robot, "right", 1.25)
        right.bus.apply_hold_torque.assert_called_once_with(1.25)
        self.assertFalse(left.mock_calls)

    def test_feedback_uses_cached_motor_torque_and_holding_state(self):
        bus = types.SimpleNamespace(
            _lock=threading.RLock(),
            _enabled=True,
            controller=types.SimpleNamespace(torque_nm=1.25, holding=True),
            _base=types.SimpleNamespace(_last_known_states={"gripper": {"torque": -1.1}}),
        )
        robot = types.SimpleNamespace(_devices={"right": types.SimpleNamespace(bus=bus)})
        self.assertEqual(
            feedback(robot, "right"),
            {
                "grip_measured_torque_nm": -1.1,
                "grip_applied_torque_nm": 1.25,
                "grip_holding": True,
            },
        )

    def test_higher_hold_target_commands_more_closing_effort_after_contact(self):
        try:
            from makermodslab.metal_gripper_hold import HoldingController
        except ImportError:
            self.skipTest("needs installed MakerMods SDK")
        controller = HoldingController(0.5, (0, 2.4))
        for now in (0.0, 0.11):
            controller.update(goal=0.0, position=1.0, velocity=0.0, effort=-0.6, kp=20.0, now=now)
        self.assertTrue(controller.holding)
        stronger = copy.deepcopy(controller)
        stronger.torque_nm = 1.25
        inputs = dict(goal=0.0, position=1.0, velocity=0.0, effort=-0.6, kp=20.0, now=0.13)
        self.assertLess(stronger.update(**inputs), controller.update(**inputs))


if __name__ == "__main__":
    unittest.main()
