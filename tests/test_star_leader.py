"""Star six-axis pose, independent gripper, and shared follower handoffs."""

from types import SimpleNamespace

import numpy as np

from dream_hackathon.so101_dashboard import Controls
from dream_hackathon.star_leader import StarFK, StarLeader, recording_destination
from dream_hackathon.unified_control import UnifiedController


def test_star_recording_destination_keeps_separate_sides_and_task(tmp_path):
    import json

    default = tmp_path / "old-dataset"
    settings = tmp_path / "star-recording.json"
    assert recording_destination(default, "right", "old task", settings) == (default, "old task")
    settings.write_text(json.dumps({"dataset_root": str(tmp_path / "usb-c-insertion"),
                                   "task": "Insert the USB-C cable into the USB-C port"}))
    for side in ("left", "right"):
        root, task = recording_destination(default, side, "old task", settings)
        assert root == tmp_path / "usb-c-insertion" / side
        assert task == "Insert the USB-C cable into the USB-C port"


def test_all_six_axes_change_pose_and_gripper_does_not():
    fk = StarFK()
    action = {f"joint{i}.pos": 0 for i in range(1, 7)}
    start = fk.pose(action)
    for i in range(1, 7):
        assert not np.allclose(fk.pose(dict(action, **{f"joint{i}.pos": 20})), start)
    np.testing.assert_allclose(fk.pose(dict(action, **{"gripper.pos": 100})), start)
    np.testing.assert_allclose(start[:3, :3].T @ start[:3, :3], np.eye(3), atol=1e-12)
    # Crossing the encoder's single-turn boundary must remain a small physical motion.
    a = fk.pose(dict(action, **{"joint3.pos": 179.9}))
    b = fk.pose(dict(action, **{"joint3.pos": -179.9}))
    assert np.linalg.norm(a[:3, 3] - b[:3, 3]) < 0.002


def test_reader_preserves_raw_joints_and_maps_seventh_encoder_to_gripper():
    angles = [10, 20, -30, 40, -50, 60, 45]
    reader = StarLeader("fake")
    reader.bus = SimpleNamespace(read_raw_angle=lambda i, multi_turn: angles[i])
    action = reader.get_action()
    assert [action[f"joint{i}.pos"] for i in range(1, 7)] == angles[:6]
    assert action["gripper.pos"] == 50
    angles[6] = 0
    assert reader.get_action()["gripper.pos"] == 0
    angles[6] = 90
    assert reader.get_action()["gripper.pos"] == 100


def test_single_star_moves_between_sides_and_coexists_with_so101(tmp_path):
    class Leader:
        side = None

        def state(self):
            return Controls(1, self.side or "none").snapshot()

        def select(self, side):
            self.side = side

        def action(self, path, body):
            return body

    nodes = {
        side: SimpleNamespace(snapshot=lambda: {"arm_open": False, "grip_torque_nm": 1}, cancel_home=lambda: None)
        for side in ("left", "right")
    }
    so101, star = Leader(), Leader()
    controller = UnifiedController(nodes, so101, star=star, sequence_path=tmp_path / "sequences.json")
    controller.select_input("left", "so101")
    controller.select_input("right", "star")
    assert (so101.side, star.side) == ("left", "right")
    controller.select_input("left", "star")
    assert so101.side is None and star.side == "left"
    assert controller.inputs == {"left": "star", "right": None}
    assert controller.state()["star"]["assigned_side"] == "left"
    controller.select_both("dream")
    assert star.side is None
