"""Single-arm LeRobot demonstrations from the SO-101 controller."""

import pathlib

import numpy as np


class ExampleRecorder:
    def __init__(self, station, side: str, root: pathlib.Path, task: str,
                 camera_roles=("overhead", "wrist_l", "wrist_r")) -> None:
        self.station, self.side, self.root, self.task = station, side, root, task
        self.camera_roles = tuple(camera_roles)
        self.cameras = {}
        self.writer = None
        self.episode = None

    @property
    def frames(self) -> int:
        return self.episode.frames if self.episode is not None else 0

    def start(self) -> None:
        from dream_hackathon import cameras, dataset, layouts

        if not self.cameras:
            self.cameras = cameras.open_cameras(
                tuple(c for c in self.station.cameras if c.role in self.camera_roles),
                self.station.record_width, self.station.record_height
            )
        if self.writer is None:
            names = [f"{self.side}_joint_{i}" for i in range(6)] + [f"{self.side}_gripper"]
            self.writer = dataset.DatasetWriter(
                self.root, self.station.fps, self.station.arm, names,
                [key for role, key in layouts.video_keys(self.station.family).items() if role in self.camera_roles],
                self.station.record_width, self.station.record_height,
            )
        self.episode = self.writer.start_episode(self.task)

    def add(self, joints, grip, command, commanded_grip) -> None:
        from dream_hackathon import layouts

        scale = layouts.GRIP_SCALE[self.station.arm]
        measured = np.append(joints, grip * scale).astype(np.float32)
        action = np.append(command, commanded_grip * scale).astype(np.float32)
        images = {key: self.cameras[role].latest() for role, key in layouts.video_keys(self.station.family).items()
                  if role in self.camera_roles}
        self.episode.add_frame(measured, action, images)

    def stop(self) -> int | None:
        episode, self.episode = self.episode, None
        try:
            return episode.save() if episode is not None else None
        finally:
            for camera in self.cameras.values():
                camera.close()
            self.cameras = {}

    def close(self) -> None:
        try:
            self.stop()
        finally:
            for camera in self.cameras.values():
                camera.close()
            self.cameras = {}
