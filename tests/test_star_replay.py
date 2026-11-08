"""Replay approach cap, recorded timing, saved units, and explicit controls."""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dream_hackathon.so101_dashboard import Controls
from dream_hackathon.star_replay import Replay, Take, latest_take


def test_star_recording_borrows_both_wrist_streams_without_camera_or_motor_ownership(monkeypatch):
    import io

    import cv2

    from dream_hackathon.remote_arm import remote_cameras

    urls = []
    ok, encoded = cv2.imencode(".jpg", np.full((16, 16, 3), 100, dtype=np.uint8))
    assert ok

    def open_url(url, **kwargs):
        urls.append(url)
        return io.BytesIO(encoded.tobytes())

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    roles = [SimpleNamespace(role=role) for role in ("overhead", "wrist_l", "wrist_r")]
    cameras = remote_cameras("http://right", side="right", other_base="http://left")(roles, 16, 16)
    assert urls == ["http://right/overhead.jpg", "http://left/wrist.jpg", "http://right/wrist.jpg"]
    assert set(cameras) == {"overhead", "wrist_l", "wrist_r"}


def test_approach_obeys_each_joint_speed_then_uses_recorded_timing():
    take = Take(
        3,
        np.array([0, 0.5, 2.0]),
        np.array([[1, -2, 0, 0, 0, 0], [2, -2, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0]]),
        np.array([0.2, 0.8, 0.4]),
    )
    replay = Replay(take, np.zeros(6), 0, 0.5, 10)
    previous = np.zeros(6)
    for now in np.arange(10, 14.01, 0.1):
        q, grip, finished = replay.step(float(now))
        assert np.max(np.abs(q - previous)) <= 0.05 + 1e-12
        assert not finished
        previous = q
    np.testing.assert_allclose(previous, take.joints[0], atol=1e-12)
    # Anchor following at the command that reaches the start, without consuming take time.
    replay.step(14.1)
    assert replay.phase == "playing"
    play_start = replay.play_started
    q, grip, finished = replay.step(play_start + 0.5)
    np.testing.assert_allclose(q, take.joints[1])
    assert grip == pytest.approx(0.8) and not finished
    q, grip, finished = replay.step(play_start + 2)
    np.testing.assert_allclose(q, take.joints[-1])
    assert finished and replay.phase == "complete"


def test_gripper_only_start_approach_and_single_frame_take():
    take = Take(0, np.array([0.0]), np.zeros((1, 6)), np.array([1.0]))
    replay = Replay(take, np.zeros(6), 0, 0.5, 0)
    _, grip, finished = replay.step(1 / 3)
    assert grip == pytest.approx(0.5) and not finished
    replay.step(2 / 3)
    assert replay.step(1)[2]


@pytest.mark.parametrize("both", [False, True])
def test_latest_take_selects_assigned_arm_and_restores_normalized_grip(tmp_path, both):
    (tmp_path / "meta").mkdir()
    sides = ("left", "right") if both else ("right",)
    names = [name for side in sides for name in [*[f"{side}_joint_{i}" for i in range(6)], f"{side}_gripper"]]
    info = {"features": {"action": {"names": names}}, "data_path": "data/episode_{episode_index:06d}.parquet"}
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    (tmp_path / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 5, "length": 2, "tasks": ["task"]}))
    (tmp_path / "data").mkdir()
    action = [*([99] * 7 if both else []), 1, 2, 3, 4, 5, 6, 75]
    pq.write_table(
        pa.table({"timestamp": [4.0, 4.1], "action": [action, action]}), tmp_path / "data/episode_000005.parquet"
    )
    take = latest_take(tmp_path, "right", "metal")
    assert take.episode == 5
    np.testing.assert_allclose(take.timestamps, [0, 0.1])
    np.testing.assert_allclose(take.joints[0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(take.grips, [0.75, 0.75])


def test_replay_explicitly_replaces_tracking_and_saves_recording_before_playback():
    controls = Controls(1, "right")
    controls.report(arm_open=True)
    controls.change({"tracking": True, "recording_requested": True})
    state = controls.change({"replay_requested": True, "replay_start_speed": 0.3})
    assert state["replay_requested"] and state["replay_phase"] == "pending"
    assert not state["tracking"] and not state["recording_requested"]
    assert state["replay_start_speed"] == 0.3
    assert not controls.change({"tracking": True})["replay_requested"]
    controls.change({"replay_requested": True})
    assert not controls.change({"tracking": False})["replay_requested"]


def test_record_is_an_explicit_toggle_and_does_not_start_with_tracking():
    controls = Controls(1, "right")
    controls.report(arm_open=True)
    assert not controls.change({"tracking": True})["recording_requested"]
    assert controls.change({"recording_requested": True})["recording_requested"]
    assert not controls.change({"recording_requested": False})["recording_requested"]


def test_empty_take_reports_an_error(tmp_path):
    with pytest.raises(ValueError, match="Record and save"):
        latest_take(tmp_path, "right", "metal")
