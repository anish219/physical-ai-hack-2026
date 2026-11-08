"""Single-arm LeRobot demonstrations from the SO-101 controller."""

import pathlib

import numpy as np


class ExampleRecorder:
    def __init__(
        self,
        station,
        side: str,
        root: pathlib.Path,
        task: str,
        camera_roles=("overhead", "wrist_l", "wrist_r"),
        *,
        context_pose=None,
        camera_factory=None,
        include_inactive_wrist=False,
    ) -> None:
        self.station, self.side, self.root, self.task = station, side, root, task
        self.camera_roles = tuple(camera_roles)
        self.inactive_wrist = "wrist_l" if side == "right" else "wrist_r"
        self.cameras = {}
        self.writer = None
        self.episode = None
        self.context_pose = context_pose
        self.camera_factory = camera_factory
        self.include_inactive_wrist = include_inactive_wrist

    @property
    def frames(self) -> int:
        return self.episode.frames if self.episode is not None else 0

    def saved_count(self) -> int:
        from dream_hackathon import dataset

        return len(dataset.list_episodes(self.root))

    def reject(self, *, current_only: bool = False) -> str:
        """Discard the active example, otherwise remove only the latest saved one."""
        from dream_hackathon import dataset

        if self.episode is not None:
            episode, self.episode = self.episode, None
            try:
                episode.discard()
                return "Rejected current example"
            finally:
                for camera in self.cameras.values():
                    camera.close()
                self.cameras = {}
        if current_only:
            return "Rejected pending example"
        episodes = dataset.list_episodes(self.root)
        if not episodes:
            raise dataset.DatasetError("No example to reject")
        index = episodes[-1].index
        dataset.drop_episodes(self.root, {index})
        self.writer = None
        return f"Rejected saved example {index}"

    def start(self) -> None:
        from dream_hackathon import cameras, dataset, layouts

        if self.context_pose is not None:
            self.context_pose()  # Report a missing context before opening cameras.
        if not self.cameras:
            self.cameras = (self.camera_factory or cameras.open_cameras)(
                tuple(
                    c
                    for c in self.station.cameras
                    if c.role in self.camera_roles and (self.include_inactive_wrist or c.role != self.inactive_wrist)
                ),
                self.station.record_width,
                self.station.record_height,
            )
        if self.writer is None:
            sides = ("left", "right") if self.context_pose is not None else (self.side,)
            names = [
                name for side in sides for name in ([f"{side}_joint_{i}" for i in range(6)] + [f"{side}_gripper"])
            ]
            self.writer = dataset.DatasetWriter(
                self.root,
                self.station.fps,
                self.station.arm,
                names,
                [key for role, key in layouts.video_keys(self.station.family).items() if role in self.camera_roles],
                self.station.record_width,
                self.station.record_height,
            )
        self.episode = self.writer.start_episode(self.task)

    def add(self, joints, grip, command, commanded_grip) -> None:
        from dream_hackathon import layouts

        scale = layouts.GRIP_SCALE[self.station.arm]
        measured = np.append(joints, grip * scale).astype(np.float32)
        action = np.append(command, commanded_grip * scale).astype(np.float32)
        if self.context_pose is not None:
            context_joints, context_grip = self.context_pose()
            context = np.append(context_joints, context_grip * scale).astype(np.float32)
            # The inactive arm's action is hold-at-measured-pose. Its model output
            # will be ignored by the independently controlled arm's API caller.
            measured = np.concatenate((context, measured) if self.side == "right" else (measured, context))
            action = np.concatenate((context, action) if self.side == "right" else (action, context))
        images = {}
        for role, key in layouts.video_keys(self.station.family).items():
            if role not in self.camera_roles:
                continue
            if role == self.inactive_wrist and not self.include_inactive_wrist:
                images[key] = np.zeros((self.station.record_height, self.station.record_width, 3), dtype=np.uint8)
            else:
                images[key] = self.cameras[role].latest()
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
