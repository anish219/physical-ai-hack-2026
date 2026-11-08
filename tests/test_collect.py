"""The collector loop over fake arms and a fake Quest: teleop, recording, dry run, homing and the lag guard."""

import dataclasses
import json
import pathlib
import re
import sys

import dream
import numpy as np
import pytest
from conftest import FLANGE, JOINT_NAMES, arm_urdf, filled_template

from dream_hackathon import cameras, collect, config, dataset, ik, layouts, quest

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"
START = np.array([0.0, -0.3, 1.2, 0.6, 0.0, 0.0])  # elbow bent: 5 cm forward is in reach


class FakeArms:
    """Arms that track every command exactly, unless stuck."""

    def __init__(self, energized: bool = True) -> None:
        self.energized = energized
        self.start_joints = {side: START.copy() for side in ("left", "right")}
        self.joints = {side: START.copy() for side in ("left", "right")}
        self.grips = {"left": 1.0, "right": 1.0}
        self.commands = 0
        self.stuck = False
        self.last_command_sides = set()

    def read(self):
        return {side: q.copy() for side, q in self.joints.items()}, dict(self.grips)

    def command(self, joints, grips) -> None:
        self.commands += 1
        self.last_command_sides = set(joints)
        if not self.stuck:
            self.joints.update({side: q.copy() for side, q in joints.items()})
            self.grips.update(grips)

    def close(self) -> None:
        pass


class FakeQuest:
    """A Quest whose next reading the test sets."""

    def __init__(self) -> None:
        self.poses = {"left": np.eye(4), "right": np.eye(4)}
        self.grip = {"left": 0.0, "right": 0.0}
        self.trigger = {"left": 0.0, "right": 0.0}
        self.press: set[str] = set()

    def read(self) -> quest.QuestState:
        pressed, self.press = frozenset(self.press), set()
        return quest.QuestState(
            {s: p.copy() for s, p in self.poses.items()}, dict(self.grip), dict(self.trigger), pressed
        )


def _collector(
    tmp_path: pathlib.Path, arms: FakeArms, headset: FakeQuest, logs: list[str], selection: str = "both"
) -> collect.Collector:
    station = config.load(filled_template("yam", tmp_path))
    # A box large enough that it never clamps the test arm's tool.
    roomy = {
        side: dataclasses.replace(arm, workspace_min_m=np.full(3, -2.0), workspace_max_m=np.full(3, 2.0))
        for side, arm in station.arms.items()
    }
    station = dataclasses.replace(
        station, dataset_dir=tmp_path / "task", record_width=320, record_height=180, arms=roomy
    )
    model = ik.urdf_model(arm_urdf(), FLANGE, (0.0, 0.0, 0.08))
    iks = {side: ik.ArmIK(model, ik.TCP_SITE, JOINT_NAMES, 2.0, 0.02) for side in ("left", "right")}
    keys = list(layouts.video_keys("yam").values())
    writer = dataset.DatasetWriter(station.dataset_dir, 30, "yam", [f"v{i}" for i in range(14)], keys, 320, 180)
    frame = np.zeros((180, 320, 3), np.uint8)
    frames = lambda: {role: frame for role in layouts.CAMERA_ROLES}  # noqa: E731
    controls = collect.SessionControls(lambda: writer, "pick up the cube")
    controls.set_controlled_arms(selection)
    return collect.Collector(
        station, controls.task, arms, headset, frames, iks, writer, log=logs.append, controls=controls
    )


def test_a_held_grip_drives_the_tool_with_the_hand(tmp_path: pathlib.Path) -> None:
    arms, headset = FakeArms(), FakeQuest()
    collector = _collector(tmp_path, arms, headset, [])
    solver = collector._iks["right"]
    before = solver.fk(arms.joints["right"])[:3, 3]
    headset.grip["right"] = 1.0
    collector.tick()  # engages at the hand's pose
    headset.poses["right"][2, 3] = -0.05  # hand 5 cm forward (Quest -z)
    headset.trigger["right"] = 1.0  # trigger pulled: close the gripper
    for _ in range(120):
        collector.tick()
    after = solver.fk(arms.joints["right"])[:3, 3]
    np.testing.assert_allclose(after - before, [0.05, 0.0, 0.0], atol=2e-3)
    np.testing.assert_allclose(arms.joints["left"], START)  # the other arm held still
    assert arms.grips == {"left": 1.0, "right": 0.0}


@pytest.mark.parametrize("side", ["left", "right"])
def test_single_arm_uses_right_controller_and_leaves_other_arm_independent(tmp_path, side):
    arms_, headset = FakeArms(), FakeQuest()
    collector = _collector(tmp_path, arms_, headset, [], selection=side)
    other = "right" if side == "left" else "left"
    headset.grip = {"left": 1.0, "right": 1.0}
    collector.tick()
    headset.poses["right"][2, 3] = -0.05
    headset.trigger["right"] = 0.8
    headset.trigger["left"] = 0.3
    # A separate source moves the other arm; Oculus must neither command it nor record an obsolete target.
    arms_.joints[other] = START + 0.1
    arms_.grips[other] = 0.6
    for _ in range(60):
        collector.tick()
    assert arms_.last_command_sides == {side}
    assert not np.allclose(arms_.joints[side], START)
    np.testing.assert_allclose(arms_.joints[other], START + 0.1)
    np.testing.assert_allclose(collector._command.joints[other], arms_.joints[other])
    assert arms_.grips[side] == pytest.approx(0.2) and arms_.grips[other] == 0.6
    snapshot = collector.snapshot()
    assert snapshot["controlled_arms"] == side
    assert snapshot["arms"][side]["controlled"] and not snapshot["arms"][other]["engaged"]


def test_switching_arms_reanchors_held_controller_and_both_restores_matching_inputs(tmp_path):
    arms_, headset = FakeArms(), FakeQuest()
    collector = _collector(tmp_path, arms_, headset, [], selection="right")
    headset.grip["right"] = 1.0
    collector.tick()
    headset.poses["right"][2, 3] = -0.1
    for _ in range(30):
        collector.tick()
    right = arms_.joints["right"].copy()
    assert collector._controls.set_controlled_arms("left")[0]
    collector.tick()
    np.testing.assert_allclose(arms_.joints["left"], START, atol=1e-6)
    np.testing.assert_allclose(arms_.joints["right"], right)
    assert arms_.last_command_sides == {"left"}
    collector._controls.set_controlled_arms("both")
    headset.grip["left"] = 1.0
    headset.trigger = {"left": 0.3, "right": 0.8}
    collector.tick()
    assert arms_.last_command_sides == {"left", "right"}
    assert arms_.grips["left"] == pytest.approx(0.7)
    assert arms_.grips["right"] == pytest.approx(0.2)


def test_joystick_homes_only_selected_arm(tmp_path):
    arms_, headset = FakeArms(), FakeQuest()
    collector = _collector(tmp_path, arms_, headset, [], selection="left")
    for side in ("left", "right"):
        arms_.joints[side] = START + 0.1
        collector._command.joints[side] = START + 0.1
    headset.press.add("RJ")
    for _ in range(60):
        collector.tick()
    np.testing.assert_allclose(arms_.joints["left"], START, atol=1e-3)
    np.testing.assert_allclose(arms_.joints["right"], START + 0.1)
    assert arms_.last_command_sides == {"left"}


def test_one_open_arm_records_without_reading_or_fabricating_the_other(tmp_path):
    import pyarrow.parquet as pq

    robot, headset = FakeArms(), FakeQuest()
    robot.joints = {"left": START.copy()}
    robot.grips = {"left": 0.5}
    robot.start_joints = {"left": START.copy()}
    collector = _collector(tmp_path, robot, headset, [], selection="left")
    writer = dataset.DatasetWriter(
        tmp_path / "left",
        30,
        "yam",
        [f"left_{i}" for i in range(7)],
        list(layouts.video_keys("yam").values()),
        320,
        180,
    )
    collector._controls.writer = writer
    headset.press.add("A")
    for _ in range(2 * dataset.min_frames(30)):
        collector.tick()
    headset.press.add("A")
    collector.tick()
    assert set(collector.snapshot()["arms"]) == {"left"}
    assert "right" not in collect._status_line(collector.snapshot())
    info = json.loads((writer.root / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [7]
    table = pq.read_table(writer.root / "data/chunk-000/episode_000000.parquet")
    assert len(table["action"][0].as_py()) == 7


def test_live_ownership_change_replaces_the_read_and_command_sides(tmp_path):
    robot, headset = FakeArms(), FakeQuest()
    robot.joints = {"left": START.copy()}
    robot.grips = {"left": 0.5}
    robot.start_joints = {"left": START.copy()}
    collector = _collector(tmp_path, robot, headset, [], selection="left")
    solvers = dict(collector._iks)
    transitions = []

    def change(selection):
        transitions.append(selection)
        robot.joints = {selection: START.copy()}
        robot.grips = {selection: 0.5}
        robot.start_joints = {selection: START.copy()}
        return {selection: solvers[selection]}

    collector._on_selection = change
    collector.tick()
    collector._controls.set_controlled_arms("right")
    headset.grip["right"] = 1.0
    collector.tick()
    assert transitions == ["right"]
    assert set(collector._command.joints) == robot.last_command_sides == {"right"}
    assert set(collector.snapshot()["arms"]) == {"right"}
    collector.go_home(timeout_s=0.1)
    assert robot.last_command_sides == {"right"}


def test_a_and_a_again_saves_an_episode_dream_accepts(tmp_path: pathlib.Path) -> None:
    arms, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms, headset, logs)
    headset.press.add("A")
    for _ in range(2 * dataset.min_frames(30)):  # 60 Hz loop, every 2nd tick recorded
        collector.tick()
    headset.press.add("A")
    collector.tick()
    assert collector.episode is None and logs[-1].startswith("saved episode 0")
    assert dream.datasets.validate_dataset(tmp_path / "task", layouts.embodiment("yam")).ok


def test_b_discards_and_a_dry_run_never_records_or_commands(tmp_path: pathlib.Path) -> None:
    arms, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms, headset, logs)
    headset.press.add("A")
    collector.tick()
    headset.press.add("B")
    collector.tick()
    assert collector.episode is None and logs[-1] == "episode discarded"

    dry = FakeArms(energized=False)
    collector = _collector(tmp_path, dry, headset, logs)
    headset.press.add("A")
    headset.grip["left"] = 1.0
    for _ in range(10):
        collector.tick()
    assert collector.episode is None and dry.commands == 0 and "a dry run does not record" in logs


def test_homing_returns_to_the_start_but_not_while_recording(tmp_path: pathlib.Path) -> None:
    arms, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms, headset, logs)
    collector._command.joints["left"] = START + 0.1
    headset.press.add("RJ")
    for _ in range(60):
        collector.tick()
    assert not collector.homing
    np.testing.assert_allclose(arms.joints["left"], START, atol=1e-3)
    headset.press.add("A")
    collector.tick()
    headset.press.add("RJ")
    collector.tick()
    assert not collector.homing and "homing is off while recording" in logs[-1]


def test_an_arm_that_cannot_follow_is_resynced_and_released(tmp_path: pathlib.Path) -> None:
    arms, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms, headset, logs)
    arms.stuck = True
    collector._command.joints["right"] = START + 0.5
    collector.tick()
    np.testing.assert_allclose(collector._command.joints["right"], START)
    assert "right arm is" in logs[-1]


def test_a_dry_run_follows_the_real_arm_until_engaged(tmp_path: pathlib.Path) -> None:
    dry, headset = FakeArms(energized=False), FakeQuest()
    collector = _collector(tmp_path, dry, headset, [])
    dry.joints["left"] = START + 0.2  # moved by hand
    collector.tick()
    np.testing.assert_allclose(collector._command.joints["left"], START + 0.2)
    headset.grip["left"] = 1.0
    collector.tick()
    dry.joints["left"] = START  # moved again, but the clutch now drives the (virtual) command
    collector.tick()
    np.testing.assert_allclose(collector._command.joints["left"], START + 0.2, atol=1e-3)


def test_snapshot_reports_the_session(tmp_path: pathlib.Path) -> None:
    arms_, headset = FakeArms(), FakeQuest()
    collector = _collector(tmp_path, arms_, headset, [])
    headset.press.add("A")
    collector.tick()
    snapshot = collector.snapshot()
    assert snapshot["mode"] == "live" and snapshot["recording"] and snapshot["episodes"] == 0
    assert set(snapshot["arms"]) == {"left", "right"} and len(snapshot["arms"]["left"]["tool_m"]) == 3
    json.dumps(snapshot)  # the dashboard sends it as JSON
    assert collect._status_line(snapshot).startswith("REC")


def test_a_stalled_camera_discards_the_episode_and_teleop_goes_on(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs)

    def stalled():
        raise cameras.CameraError("camera 'wrist_l' stalled")

    collector._frames = stalled
    headset.press.add("A")
    collector.tick()
    assert collector.episode is None
    assert any("stalled" in line for line in logs)
    assert any("stalled" in warning for warning in collector.snapshot()["warnings"])
    collector.tick()  # the loop keeps running


def test_a_hang_up_ends_the_session_like_ctrl_c_and_logs_to_a_file(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setattr(collect, "LOG_PATH", tmp_path / "collect.log")
    stdout, stderr = sys.stdout, sys.stderr
    try:
        with pytest.raises(KeyboardInterrupt):
            collect._hang_up(1, None)
        collect._say("returning the arms")
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = stdout, stderr
    text = (tmp_path / "collect.log").read_text()
    assert "SSH connection was lost" in text and "returning the arms" in text


def test_homing_gives_up_after_its_timeout(tmp_path: pathlib.Path) -> None:
    arms_, headset, logs = FakeArms(), FakeQuest(), []
    collector = _collector(tmp_path, arms_, headset, logs)
    arms_.stuck = True
    collector._command.joints["left"] = START + 0.1
    arms_.joints["left"] = START + 0.1
    collector.go_home(timeout_s=0.05)
    assert not collector.homing and any("did not reach their start" in line for line in logs)


def test_the_task_falls_back_to_a_neutral_text():
    assert collect.choose_task(" stack the cups ", "from the config") == "stack the cups"
    assert collect.choose_task(None, "from the config") == "from the config"
    assert collect.choose_task("  ", None) == collect.DEFAULT_TASK
    assert collect.DEFAULT_TASK.strip()


def test_a_missing_tool_center_point_is_one_line_not_a_traceback(tmp_path: pathlib.Path) -> None:
    path = filled_template("maker", tmp_path)
    path.write_text(re.sub(r"tcp_xyz_m: .*", "tcp_xyz_m: null", path.read_text()))  # a config without its point
    with pytest.raises(SystemExit) as stopped:
        collect.main(["--config", str(path), "--dataset-dir", str(tmp_path / "data")])
    assert str(stopped.value.code).startswith("config: 'arms.right.tcp_xyz_m' is not set")


def test_run_initializes_only_selected_ik_and_dataset_without_global_ownership(tmp_path, monkeypatch):
    station = config.load(filled_template("yam", tmp_path))
    selected_iks, launched = [], []

    def open_ik(selected):
        selected_iks.append(set(selected.arms))
        return {}

    monkeypatch.setattr(collect.arms, "open_ik", open_ik)
    monkeypatch.setattr(collect, "_run_locked", lambda station, controls, iks, **kw: launched.append(controls))
    collect.run(station, "test", dry_run=True, sim=False, quest_ip=None, controlled_arms="left")
    assert selected_iks == [{"left"}]
    assert launched[0].writer.root == station.dataset_dir / "left"
    assert launched[0].controlled_arms == "left"
