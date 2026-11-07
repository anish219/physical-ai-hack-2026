"""Partial commands reach only the requested vendor follower, without opening hardware."""

import concurrent.futures
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dream_hackathon.arms.makermods import MakerModsArms
from dream_hackathon.arms.yam import YamArms


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("adapter", [YamArms, MakerModsArms])
def test_partial_command_leaves_other_follower_untouched(adapter, side):
    robot = adapter.__new__(adapter)
    robot.energized = True
    followers = {s: Mock() for s in ("left", "right")}
    other = "right" if side == "left" else "left"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        if adapter is YamArms:
            robot._robots = followers
        else:
            robot._devices = followers
            robot._pool = pool
            robot._arms = {s: SimpleNamespace(gripper_closed_deg=0, gripper_open_deg=90) for s in followers}
        robot.command({side: np.full(6, np.pi / 6)}, {side: 0.5})
        assert not followers[other].mock_calls
        if adapter is YamArms:
            followers[side].command_joint_pos.assert_called_once()
            np.testing.assert_allclose(followers[side].command_joint_pos.call_args.args[0], [*[np.pi / 6] * 6, 0.5])
        else:
            followers[side].send_action.assert_called_once()
            action = followers[side].send_action.call_args.args[0]
            assert action["shoulder_pan.pos"] == pytest.approx(30)
            assert action["gripper.pos"] == 45
