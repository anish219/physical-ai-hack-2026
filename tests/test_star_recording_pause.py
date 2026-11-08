"""An untracked interval pauses one Star take without creating gaps or extra takes."""

from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from dream_hackathon import dataset
from dream_hackathon.so101_dashboard import Controls
from dream_hackathon.so101_recording import ExampleRecorder, recording_frame_enabled


def test_track_untrack_resumes_one_take_and_stop_can_save_while_paused(tmp_path):
    station = SimpleNamespace(
        arm="metal",
        family="makermods",
        fps=30,
        record_width=16,
        record_height=16,
        cameras=(SimpleNamespace(role="overhead"),),
    )
    camera = SimpleNamespace(latest=lambda: np.zeros((16, 16, 3), dtype=np.uint8), close=lambda: None)
    recorder = ExampleRecorder(
        station, "right", tmp_path, "insert USB-C", ("overhead",), camera_factory=lambda *args: {"overhead": camera}
    )
    controls = Controls(1, "right", pause_recording_untracked=True)
    controls.report(arm_open=True)
    controls.change({"recording_requested": True})
    recorder.start()
    original_episode = recorder.episode
    assert controls.snapshot()["recording_paused"]
    for tracking, ticks in ((True, 30), (False, 90), (True, 30)):
        controls.change({"tracking": tracking})
        for _ in range(ticks):
            if recording_frame_enabled(controls.snapshot(), pause_untracked=True):
                recorder.add(np.zeros(6), 0, np.ones(6), 1)
        assert recorder.episode is original_episode
    assert recorder.frames == 60
    controls.change({"tracking": False})
    assert controls.snapshot()["recording_paused"]
    controls.change({"recording_requested": False})
    assert not controls.snapshot()["recording_paused"]
    assert recorder.stop() == 0
    assert len(dataset.list_episodes(tmp_path)) == 1
    table = pq.read_table(tmp_path / "data/chunk-000/episode_000000.parquet")
    assert len(table) == 60
    np.testing.assert_allclose(table["timestamp"].to_pylist(), np.arange(60) / 30, atol=1e-6)


def test_tracking_alone_does_not_record_and_so101_retains_existing_behavior():
    controls = Controls(1, "right", pause_recording_untracked=True)
    controls.report(arm_open=True)
    controls.change({"tracking": True})
    assert not recording_frame_enabled(controls.snapshot(), pause_untracked=True)
    controls.change({"recording_requested": True, "tracking": False})
    assert not recording_frame_enabled(controls.snapshot(), pause_untracked=True)
    assert recording_frame_enabled(controls.snapshot(), pause_untracked=False)
