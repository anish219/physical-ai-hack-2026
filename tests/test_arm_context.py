import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from dream_hackathon import arm_context, dataset
from dream_hackathon.so101_recording import ExampleRecorder


class ContextTests(unittest.TestCase):
    def test_shared_pose_retains_static_measurement_and_can_be_updated(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "No measured left"):
                arm_context.read("left", directory=folder)
            arm_context.publish("left", np.arange(6), .25, directory=folder)
            joints, grip = arm_context.read("left", directory=folder)
            np.testing.assert_array_equal(joints, np.arange(6))
            self.assertEqual(grip, .25)
            arm_context.publish("left", np.ones(6), .75, directory=folder)
            self.assertEqual(arm_context.read("left", directory=folder)[1], .75)

    def test_each_side_records_full_layout_with_other_action_holding_measured_pose(self):
        for side in ("left", "right"):
            recorder = ExampleRecorder(SimpleNamespace(arm="metal", family="makermods"), side,
                                       pathlib.Path("unused"), "task", (),
                                       context_pose=lambda: (np.full(6, 2), .3))
            recorder.episode = Mock()
            recorder.add(np.zeros(6), .25, np.ones(6), .75)
            state, action, images = recorder.episode.add_frame.call_args.args
            own = slice(7, 14) if side == "right" else slice(0, 7)
            other = slice(0, 7) if side == "right" else slice(7, 14)
            np.testing.assert_allclose(state[own], [0]*6 + [25])
            np.testing.assert_allclose(action[own], [1]*6 + [75])
            np.testing.assert_allclose(state[other], [2]*6 + [30])
            np.testing.assert_array_equal(action[other], state[other])
            self.assertEqual(images, {})

    def test_historical_export_preserves_original_and_recomputes_stats(self):
        import pyarrow.parquet as pq

        tools = pathlib.Path(__file__).resolve().parents[1] / "tools"
        sys.path.insert(0, str(tools))
        try:
            spec = importlib.util.spec_from_file_location("add_arm_context", tools / "add_arm_context.py")
            export = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(export)
        finally:
            sys.path.remove(str(tools))
        for side in ("left", "right"):
            with tempfile.TemporaryDirectory() as folder:
                source, target = pathlib.Path(folder)/"source", pathlib.Path(folder)/"target"
                names = [f"{side}_joint_{i}" for i in range(6)] + [f"{side}_gripper"]
                writer = dataset.DatasetWriter(source, 30, "metal", names, [], 160, 90)
                episode = writer.start_episode("task")
                for i in range(dataset.min_frames(30)):
                    episode.add_frame(np.full(7, i), np.full(7, i + 1), {})
                episode.save()
                pose = {"side": "left" if side == "right" else "right",
                        "joints_rad": [2]*6, "grip": .3, "measured_at": 123}
                export.convert(source, target, side, pose)
                relative = "data/chunk-000/episode_000000.parquet"
                original, result = pq.read_table(source/relative), pq.read_table(target/relative)
                self.assertEqual(len(original["action"][0].as_py()), 7)
                values = result["action"][0].as_py()
                self.assertEqual(len(values), 14)
                self.assertEqual(values[:7] if side == "right" else values[7:], [2]*6 + [30])
                stats = json.loads((target/"meta/episodes_stats.jsonl").read_text())
                self.assertEqual(len(stats["stats"]["action"]["mean"]), 14)
                with self.assertRaises(FileExistsError):
                    export.convert(source, target, side, pose)


if __name__ == "__main__":
    unittest.main()
