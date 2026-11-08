"""Move to a saved Star take's start at a joint-speed cap, then play its timing."""

import dataclasses
import json

import numpy as np


@dataclasses.dataclass
class Take:
    episode: int
    timestamps: np.ndarray
    joints: np.ndarray
    grips: np.ndarray


def latest_take(root, side, arm):
    import pyarrow.parquet as pq

    from dream_hackathon import dataset, layouts

    episodes = dataset.list_episodes(root)
    if not episodes:
        raise ValueError("Record and save a take before replaying")
    episode = episodes[-1].index
    info = json.loads((root / "meta/info.json").read_text())
    names = info["features"]["action"]["names"]
    indices = [names.index(f"{side}_joint_{i}") for i in range(6)]
    grip_index = names.index(f"{side}_gripper")
    path = root / info["data_path"].format(episode_chunk=episode // dataset.CHUNKS_SIZE, episode_index=episode)
    table = pq.read_table(path, columns=["timestamp", "action"])
    times = np.asarray(table["timestamp"].to_pylist(), dtype=float)
    actions = np.asarray(table["action"].to_pylist(), dtype=float)
    if not len(times):
        raise ValueError("The saved take has no frames")
    return Take(episode, times - times[0], actions[:, indices], actions[:, grip_index] / layouts.GRIP_SCALE[arm])


class Replay:
    def __init__(self, take, joints, grip, start_speed, now):
        self.take = take
        self.start_joints = np.asarray(joints, dtype=float).copy()
        self.start_grip = float(grip)
        self.approach_duration = max(
            float(np.max(np.abs(take.joints[0] - self.start_joints))) / start_speed,
            abs(float(take.grips[0]) - self.start_grip) / 1.5,
        )
        self.started = now
        self.play_started = None
        self.phase = "approaching"
        self.elapsed = 0.0

    def step(self, now):
        if self.play_started is None:
            fraction = min(1.0, (now - self.started) / self.approach_duration) if self.approach_duration else 1.0
            joints = self.start_joints + fraction * (self.take.joints[0] - self.start_joints)
            grip = self.start_grip + fraction * (self.take.grips[0] - self.start_grip)
            if fraction == 1:
                self.play_started = now
                self.phase = "playing"
            return joints, float(grip), False
        self.elapsed = min(now - self.play_started, float(self.take.timestamps[-1]))
        joints = np.array([np.interp(self.elapsed, self.take.timestamps, self.take.joints[:, i]) for i in range(6)])
        grip = float(np.interp(self.elapsed, self.take.timestamps, self.take.grips))
        finished = self.elapsed >= self.take.timestamps[-1]
        if finished:
            self.phase = "complete"
        return joints, grip, finished
