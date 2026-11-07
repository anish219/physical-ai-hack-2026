"""Hardware and per-arm ownership for a collector that can change its selected sides."""

import dataclasses
from contextlib import ExitStack

from dream_hackathon import arms, config, station_lock


def selected_station(station: config.StationConfig, selection: str) -> config.StationConfig:
    if selection not in ("left", "right", "both"):
        raise ValueError("choose left, right, or both")
    sides = ("left", "right") if selection == "both" else (selection,)
    return dataclasses.replace(station, arms={s: station.arms[s] for s in sides})


class ArmSession:
    """Open, read, command, return and close only the arms owned by this process."""

    def __init__(self, station: config.StationConfig, *, energize: bool, sim: bool) -> None:
        self.station = station
        self.energized = energize
        self.sim = sim
        self._robots: dict[str, arms.Arms] = {}
        self._locks: dict[str, station_lock.ArmLocks] = {}

    @property
    def start_joints(self) -> arms.Joints:
        return {s: robot.start_joints[s] for s, robot in self._robots.items()}

    def select(self, selection: str) -> None:
        wanted = selected_station(self.station, selection)
        added = []
        try:
            for side in wanted.arms:
                if side in self._robots:
                    continue
                lock = (
                    station_lock.ArmLocks()
                    if self.sim
                    else station_lock.acquire_sides(
                        "dhk-collect", [side], self.station.dataset_dir.resolve() / selection
                    )
                )
                try:
                    robot = arms.open_arms(selected_station(self.station, side), energize=self.energized, sim=self.sim)
                except BaseException:
                    lock.close()
                    raise
                self._robots[side], self._locks[side] = robot, lock
                added.append(side)
        except BaseException:
            for side in added:
                self._close_side(side)
            raise
        for side in self._robots.keys() - wanted.arms.keys():
            self._close_side(side)
        for lock in self._locks.values():
            lock.set_dataset(self.station.dataset_dir.resolve() / selection)

    def read(self) -> tuple[arms.Joints, arms.Grips]:
        joints, grips = {}, {}
        for robot in self._robots.values():
            q, g = robot.read()
            joints.update(q)
            grips.update(g)
        return joints, grips

    def command(self, joints: arms.Joints, grips: arms.Grips) -> None:
        for side in joints:
            self._robots[side].command({side: joints[side]}, {side: grips[side]})

    def _close_side(self, side: str) -> None:
        try:
            self._robots.pop(side).close()
        finally:
            self._locks.pop(side).close()

    def close(self) -> None:
        with ExitStack() as cleanup:
            for side in list(self._robots):
                cleanup.callback(self._close_side, side)
