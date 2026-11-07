"""Selected hardware is the only hardware opened, read, homed or closed."""

import dataclasses

import numpy as np
import pytest
from conftest import filled_template

from dream_hackathon import arms, config, station_lock
from dream_hackathon.arms.session import ArmSession


class Follower:
    energized = True

    def __init__(self, side):
        self.side = side
        self.start_joints = {side: np.zeros(6)}
        self.closed = False

    def read(self):
        assert not self.closed
        return self.start_joints, {self.side: 0.5}

    def command(self, joints, grips):
        assert set(joints) == set(grips) == {self.side}

    def close(self):
        self.closed = True


def test_only_selected_followers_open_and_switching_releases_old_owner(tmp_path, monkeypatch):
    station = config.load(filled_template("yam", tmp_path))
    opened, held = [], set()

    def acquire(holder, sides, dataset):
        (side,) = sides
        assert side not in held
        held.add(side)
        lock = station_lock.ArmLocks()
        lock.callback(held.remove, side)
        return lock

    def open_arm(selected, **kwargs):
        (side,) = selected.arms
        follower = Follower(side)
        opened.append(follower)
        return follower

    monkeypatch.setattr(station_lock, "acquire_sides", acquire)
    monkeypatch.setattr(arms, "open_arms", open_arm)
    session = ArmSession(station, energize=True, sim=False)
    session.select("left")
    assert held == {"left"} and [f.side for f in opened] == ["left"]
    assert set(session.read()[0]) == set(session.start_joints) == {"left"}
    session.select("both")
    assert held == {"left", "right"} and len(opened) == 2
    session.select("right")
    assert held == {"right"} and opened[0].closed and not opened[1].closed
    assert set(session.read()[0]) == {"right"}
    session.close()
    assert not held and all(f.closed for f in opened)


def test_busy_other_arm_is_never_opened(tmp_path, monkeypatch):
    station = config.load(filled_template("yam", tmp_path))
    opened = []

    def acquire(holder, sides, dataset):
        if "right" in sides:
            raise station_lock.StationBusyError("custom controller owns right")
        return station_lock.ArmLocks()

    def open_arm(selected, **kwargs):
        (side,) = selected.arms
        opened.append(side)
        return Follower(side)

    monkeypatch.setattr(station_lock, "acquire_sides", acquire)
    monkeypatch.setattr(arms, "open_arms", open_arm)
    session = ArmSession(station, energize=True, sim=False)
    session.select("left")
    with pytest.raises(station_lock.StationBusyError):
        session.select("both")
    assert opened == ["left"]
    assert set(session.read()[0]) == {"left"}
    session.close()


def test_yam_opens_only_configured_side(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from dream_hackathon.arms.yam import YamArms

    station = config.load(filled_template("yam", tmp_path))
    station = dataclasses.replace(station, arms={"left": station.arms["left"]})
    calls = []
    vendor = ModuleType("i2rt.robots.get_robot")
    vendor.get_yam_robot = lambda **kwargs: (
        calls.append(kwargs)
        or SimpleNamespace(
            get_observations=lambda: {"joint_pos": np.zeros(6), "gripper_pos": [0.5]}, close=lambda: None
        )
    )
    utils = ModuleType("i2rt.robots.utils")
    utils.GripperType = SimpleNamespace(from_string_name=lambda name: name)
    monkeypatch.setitem(sys.modules, "i2rt.robots.get_robot", vendor)
    monkeypatch.setitem(sys.modules, "i2rt.robots.utils", utils)
    robot = YamArms(station, energize=True, sim=True)
    assert len(calls) == 1 and calls[0]["channel"] == station.arms["left"].port
    assert set(robot.read()[0]) == {"left"}
    robot.close()
