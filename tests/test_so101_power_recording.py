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
        maker.metal_gripper = types.SimpleNamespace(install_metal_gripper=Mock())
        station = types.SimpleNamespace(
            arms={side: types.SimpleNamespace(port=side) for side in ("left", "right")},
            makermods=types.SimpleNamespace(can_interface="slcan"),
        )
        with patch.dict(sys.modules, {"lerobot.robots.metal_follower": module, "makermodslab": maker}):
            for side in ("left", "right"):
                with self.subTest(side=side):
                    module.MetalFollower.reset_mock()
                    maker.metal_gripper.install_metal_gripper.reset_mock()
                    torque.de_energize_can_bus.reset_mock()
                    self.assertEqual(kill_controlled_arm(station, side), [])
                    module.MetalFollower.assert_called_once()
                    self.assertEqual(module.MetalFollower.call_args.args[0].port, side)
                    maker.metal_gripper.install_metal_gripper.assert_called_once_with(follower, "")
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
        maker.metal_gripper = types.SimpleNamespace(install_metal_gripper=Mock())
        station = types.SimpleNamespace(arms={"left": Mock(), "right": Mock()}, makermods=Mock())
        with patch.dict(sys.modules, {"lerobot.robots.metal_follower": module, "makermodslab": maker}):
            self.assertEqual(kill_controlled_arm(station, "right", active), [])
        self.assertEqual(calls, ["off", "close"])
        module.MetalFollower.assert_not_called()


class RecordingTests(unittest.TestCase):
    def test_reject_discards_current_or_only_latest_saved_and_writer_can_resume(self):
        from dream_hackathon import cameras, dataset

        station = types.SimpleNamespace(
            arm="metal",
            family="makermods",
            fps=30,
            record_width=160,
            record_height=90,
            cameras=(types.SimpleNamespace(role="overhead"),),
        )
        camera = Mock()
        camera.latest.return_value = np.full((90, 160, 3), 123, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            recorder = ExampleRecorder(station, "right", pathlib.Path(folder), "task", ("overhead",))
            with patch.object(cameras, "open_cameras", return_value={"overhead": camera}):
                for expected in (0, 1):
                    recorder.start()
                    for _ in range(dataset.min_frames(30)):
                        recorder.add(np.zeros(6), 0, np.ones(6), 1)
                    self.assertEqual(recorder.stop(), expected)
                self.assertEqual(recorder.saved_count(), 2)
                self.assertEqual(recorder.reject(), "Rejected saved example 1")
                self.assertEqual(recorder.saved_count(), 1)
                self.assertEqual(recorder.reject(current_only=True), "Rejected pending example")
                self.assertEqual(recorder.saved_count(), 1)
                recorder.start()
                recorder.add(np.zeros(6), 0, np.ones(6), 1)
                self.assertEqual(recorder.reject(current_only=True), "Rejected current example")
                self.assertIsNone(recorder.episode)
                self.assertEqual(recorder.saved_count(), 1)
                recorder.start()
                for _ in range(dataset.min_frames(30)):
                    recorder.add(np.zeros(6), 0, np.ones(6), 1)
                self.assertEqual(recorder.stop(), 1)
                self.assertEqual(recorder.saved_count(), 2)
            recorder.close()

    def test_only_overhead_and_controlled_wrist_are_opened_other_view_is_blank(self):
        from dream_hackathon import cameras, dataset, layouts

        for side, inactive in (("right", "wrist_l"), ("left", "wrist_r")):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as folder:
                roles = ("overhead", "wrist_l", "wrist_r")
                station = types.SimpleNamespace(
                    arm="metal",
                    family="makermods",
                    fps=30,
                    record_width=160,
                    record_height=90,
                    cameras=tuple(types.SimpleNamespace(role=role) for role in roles),
                )
                live = {role: Mock() for role in roles if role != inactive}
                for camera in live.values():
                    camera.latest.return_value = np.full((90, 160, 3), 123, dtype=np.uint8)
                writer = Mock()
                with patch.object(cameras, "open_cameras", return_value=live) as opened:
                    with patch.object(dataset, "DatasetWriter", return_value=writer):
                        recorder = ExampleRecorder(station, side, pathlib.Path(folder), "task")
                        recorder.start()
                    selected_configs = opened.call_args.args[0]
                    self.assertEqual({cfg.role for cfg in selected_configs}, set(roles) - {inactive})
                    recorder.add(np.zeros(6), 0, np.zeros(6), 0)
                images = writer.start_episode.return_value.add_frame.call_args.args[2]
                keys = layouts.video_keys("makermods")
                self.assertEqual(set(images), set(keys.values()))
                self.assertFalse(images[keys[inactive]].any())
                for role in set(roles) - {inactive}:
                    self.assertTrue((images[keys[role]] == 123).all())
                recorder.close()

    def test_explicit_overhead_only_dataset_has_one_video_feature(self):
        try:
            import json

            from dream_hackathon import dataset
        except ImportError:
            self.skipTest("needs video and dataset dependencies")
        station = types.SimpleNamespace(arm="metal", family="makermods", record_width=160, record_height=90)
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            recorder = ExampleRecorder(station, "right", root, "move the cube", ("overhead",))
            camera = Mock()
            camera.latest.return_value = np.zeros((90, 160, 3), dtype=np.uint8)
            recorder.cameras = {"overhead": camera}
            names = [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]
            recorder.writer = dataset.DatasetWriter(root, 30, "metal", names, ["observation.images.scene"], 160, 90)
            recorder.start()
            for _ in range(dataset.min_frames(30)):
                recorder.add(np.zeros(6), 0.25, np.ones(6), 0.75)
            recorder.stop()
            info = json.loads((root / "meta/info.json").read_text())
            video_features = {key for key, value in info["features"].items() if value["dtype"] == "video"}
            self.assertEqual(video_features, {"observation.images.scene"})
            self.assertEqual(len(list((root / "videos").rglob("*.mp4"))), 1)

    def test_real_dataset_contains_measured_action_vectors_and_videos(self):
        try:
            import av
            import pyarrow.parquet as pq

            from dream_hackathon import dataset, layouts
        except ImportError:
            self.skipTest("needs video and dataset dependencies")
        station = types.SimpleNamespace(arm="metal", family="makermods", record_width=160, record_height=90)
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            recorder = ExampleRecorder(station, "right", root, "move the cube")
            keys = layouts.video_keys("makermods")
            recorder.cameras = {role: Mock() for role in keys}
            for camera in recorder.cameras.values():
                camera.latest.return_value = np.full((90, 160, 3), 123, dtype=np.uint8)
            names = [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]
            recorder.writer = dataset.DatasetWriter(root, 30, "metal", names, list(keys.values()), 160, 90)
            recorder.start()
            for _ in range(dataset.min_frames(30)):
                recorder.add(np.zeros(6), 0.25, np.ones(6), 0.75)
            recorder.cameras["wrist_l"].latest.assert_not_called()
            self.assertEqual(recorder.stop(), 0)
            table = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
            np.testing.assert_allclose(table["observation.state"][0].as_py(), [0, 0, 0, 0, 0, 0, 25])
            np.testing.assert_allclose(table["action"][0].as_py(), [1, 1, 1, 1, 1, 1, 75])
            self.assertEqual(len(list((root / "videos").rglob("*.mp4"))), 3)
            for role, key in keys.items():
                path = root / "videos/chunk-000" / key / "episode_000000.mp4"
                with av.open(str(path)) as video:
                    frame = next(video.decode(video=0)).to_ndarray(format="bgr24")
                if role == "wrist_l":
                    self.assertLessEqual(int(frame.max()), 1)
                else:
                    self.assertGreater(float(frame.mean()), 100)
            recorder.close()


if __name__ == "__main__":
    unittest.main()
