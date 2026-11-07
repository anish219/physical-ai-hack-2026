"""Tool-space mapping, coordinate frames, and SO-101 FK without hardware."""

import importlib.util
import unittest

import numpy as np

from dream_hackathon.so101 import LEADER_JOINTS, LeaderFK, PoseMapping, base_rotation


class PoseMappingTests(unittest.TestCase):
    def setUp(self):
        self.leader_start = np.eye(4)
        self.leader_start[:3, 3] = [0.1, 0.2, 0.3]
        self.follower_start = np.eye(4)
        self.follower_start[:3, :3] = base_rotation([20, 30, 40])
        self.follower_start[:3, 3] = [0.4, -0.1, 0.2]

    def test_initial_poses_need_not_match(self):
        mapping = PoseMapping(self.leader_start, self.follower_start, base_rotation([0, 0, 90]))
        np.testing.assert_allclose(mapping.target(self.leader_start), self.follower_start, atol=1e-12)

    def test_translation_uses_base_alignment_and_scale(self):
        mapping = PoseMapping(self.leader_start, self.follower_start, base_rotation([0, 0, 90]), 2)
        moved = self.leader_start.copy()
        moved[0, 3] += 0.05
        target = mapping.target(moved)
        np.testing.assert_allclose(target[:3, 3], self.follower_start[:3, 3] + [0, 0.1, 0], atol=1e-12)
        np.testing.assert_allclose(target[:3, :3], self.follower_start[:3, :3], atol=1e-12)

    def test_orientation_applies_relative_rotation_in_follower_base(self):
        self.leader_start[:3, :3] = base_rotation([10, -20, 30])
        alignment = base_rotation([0, 0, 90])
        turned = self.leader_start.copy()
        delta = base_rotation([25, 0, 0])
        turned[:3, :3] = delta @ self.leader_start[:3, :3]
        mapping = PoseMapping(self.leader_start, self.follower_start, alignment)
        target = mapping.target(turned)
        expected = alignment @ delta @ alignment.T @ self.follower_start[:3, :3]
        np.testing.assert_allclose(target[:3, :3], expected, atol=1e-12)
        np.testing.assert_allclose(target[:3, 3], self.follower_start[:3, 3])

    def test_target_does_not_change_start_poses(self):
        before_l = self.leader_start.copy()
        before_f = self.follower_start.copy()
        mapping = PoseMapping(self.leader_start, self.follower_start, np.eye(3))
        mapping.target(np.eye(4))
        np.testing.assert_array_equal(before_l, self.leader_start)
        np.testing.assert_array_equal(before_f, self.follower_start)


@unittest.skipUnless(importlib.util.find_spec("mujoco") and importlib.util.find_spec("mink"), "needs MuJoCo/Mink")
class LeaderFKTests(unittest.TestCase):
    def test_each_arm_joint_changes_pose_and_trigger_does_not(self):
        fk = LeaderFK()
        action = {f"{name}.pos": 0.0 for name in LEADER_JOINTS}
        action["gripper.pos"] = 0
        start = fk.pose(action)
        for name in LEADER_JOINTS:
            with self.subTest(joint=name):
                moved = dict(action)
                moved[f"{name}.pos"] = 20
                self.assertFalse(np.allclose(fk.pose(moved), start))
        action["gripper.pos"] = 100
        np.testing.assert_allclose(fk.pose(action), start)
        np.testing.assert_allclose(start[:3, :3].T @ start[:3, :3], np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(start[:3, :3]), 1)


if __name__ == "__main__":
    unittest.main()
