"""Explicit kill avoids homing; recorded frames contain the selected arm and cameras."""

import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dream_hackathon.so101_power import kill_controlled_arm
from dream_hackathon.so101_recording import ExampleRecorder


class KillTests(unittest.TestCase):
    def test_adapter_power_off_skips_return_pose(self):
        try:
            from dream_hackathon.arms.makermods import MakerModsArms
        except ImportError:
            self.skipTest("needs arm-model dependencies")
        robot = MakerModsArms.__new__(MakerModsArms)
        robot._devices = {"right": Mock()}
        robot._rest_poses = [(Mock(), {"shoulder_pan": 1})]
        robot._family = Mock()
        robot._family.release_torque.return_value = []
        robot.energized = True
        self.assertEqual(robot.power_off(), [])
        self.assertEqual(robot._rest_poses, [])
        self.assertFalse(robot.energized)
        robot._family.return_to_rest.assert_not_called()
        robot._family.release_torque.assert_called_once()

    def test_only_selected_bus_disabled_with_both_arms_configured(self):
        follower = Mock()
        module = types.ModuleType("lerobot.robots.metal_follower")
        module.MetalFollower = Mock(return_value=follower)
        module.MetalFollowerConfig = Mock(side_effect=lambda **kw: types.SimpleNamespace(**kw))
        torque = types.SimpleNamespace(de_energize_can_bus=Mock(return_value=[]))
        maker = types.ModuleType("makermodslab")
        maker.torque = torque
        station = types.SimpleNamespace(
            arms={side: types.SimpleNamespace(port=side) for side in ("left", "right")},
            makermods=types.SimpleNamespace(can_interface="slcan"),
        )
        with patch.dict(sys.modules, {"lerobot.robots.metal_follower": module, "makermodslab": maker}):
            for side in ("left", "right"):
                with self.subTest(side=side):
                    module.MetalFollower.reset_mock()
                    torque.de_energize_can_bus.reset_mock()
                    self.assertEqual(kill_controlled_arm(station, side), [])
                    module.MetalFollower.assert_called_once()
                    self.assertEqual(module.MetalFollower.call_args.args[0].port, side)
                    torque.de_energize_can_bus.assert_called_once_with(follower.bus, f"{side} follower arm")
        follower.connect.assert_not_called()
        follower.send_action.assert_not_called()

    def test_open_arm_is_powered_off_before_close(self):
        active = Mock()
        active._devices = {"right": Mock()}
        active.power_off.return_value = []
        calls = []
        active.power_off.side_effect = lambda: calls.append("off") or []
        active.close.side_effect = lambda: calls.append("close")
        module = types.ModuleType("lerobot.robots.metal_follower")
        module.MetalFollower, module.MetalFollowerConfig = Mock(), Mock()
        maker = types.ModuleType("makermodslab")
        maker.torque = types.SimpleNamespace(de_energize_can_bus=Mock(return_value=[]))
        station = types.SimpleNamespace(arms={"left": Mock(), "right": Mock()}, makermods=Mock())
        with patch.dict(sys.modules, {"lerobot.robots.metal_follower": module, "makermodslab": maker}):
            self.assertEqual(kill_controlled_arm(station, "right", active), [])
        self.assertEqual(calls, ["off", "close"])
        module.MetalFollower.assert_not_called()


class RecordingTests(unittest.TestCase):
    def test_real_dataset_contains_measured_action_vectors_and_videos(self):
        try:
            from dream_hackathon import dataset, layouts
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("needs video and dataset dependencies")
        station = types.SimpleNamespace(arm="metal", family="makermods")
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            recorder = ExampleRecorder(station, "right", root, "move the cube")
            keys = layouts.video_keys("makermods")
            recorder.cameras = {role: Mock() for role in keys}
            for camera in recorder.cameras.values():
                camera.latest.return_value = np.zeros((90, 160, 3), dtype=np.uint8)
            names = [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]
            recorder.writer = dataset.DatasetWriter(root, 30, "metal", names, list(keys.values()), 160, 90)
            recorder.start()
            for _ in range(dataset.min_frames(30)):
                recorder.add(np.zeros(6), 0.25, np.ones(6), 0.75)
            self.assertEqual(recorder.stop(), 0)
            table = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
            np.testing.assert_allclose(table["observation.state"][0].as_py(), [0, 0, 0, 0, 0, 0, 25])
            np.testing.assert_allclose(table["action"][0].as_py(), [1, 1, 1, 1, 1, 1, 75])
            self.assertEqual(len(list((root / "videos").rglob("*.mp4"))), 3)
            recorder.close()


if __name__ == "__main__":
    unittest.main()
