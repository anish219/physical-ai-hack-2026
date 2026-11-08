"""The two follower arms of a station, through the arm vendor's own SDK.

- YAM: i2rt's SDK (arms/yam.py), which runs each arm's motor loop.
- Metal and Maker arms: MakerMods' LeRobot followers and makermodslab's start-up and shut-down sequence
  (arms/makermods.py).

Every family offers the same few calls, in the units the recordings use: joints in radians (six per arm, base to
wrist) and the gripper from 0 (closed) to 1 (open). The teleop IK of each family is built apart (open_ik()), so the
arms can be read before the IK is set up.

@see Arms, open_arms(), open_ik()
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt

from dream_hackathon import config, ik

## Per side, six joint angles in radians.
Joints = dict[str, npt.NDArray[np.float64]]
## Per side, the gripper from 0 (closed) to 1 (open).
Grips = dict[str, float]


class ArmsError(RuntimeError):
    """The arms cannot be opened yet; the message says why and what to do."""


class Arms(Protocol):
    """Both arms of a station.

    Attributes:
    - start_joints: Per side, the joints when the arms were opened: homing and the shutdown return there.
    - energized: Whether command() drives the motors; False for a dry run, which only reads.
    """

    start_joints: Joints
    energized: bool

    def read(self) -> tuple[Joints, Grips]:
        """The measured joints and grippers."""
        ...

    def command(self, joints: Joints, grips: Grips) -> None:
        """Sends targets only to the sides supplied in joints and grips; omitted arms receive no command."""
        ...

    def close(self) -> None:
        """Shuts the arms down the vendor's way; call only with the arms at start_joints."""
        ...


def open_arms(station: config.StationConfig, *, energize: bool, sim: bool = False) -> Arms:
    """Opens both arms of a station.

    @param station The station config.
    @param energize True to drive the arms; False for a dry run that only reads them.
    @param sim YAM only: i2rt's MuJoCo simulation instead of the real arms.
    @return The arms.
    """
    if station.family == "yam":
        from dream_hackathon.arms import yam

        return yam.YamArms(station, energize=energize, sim=sim)
    if sim:
        raise ValueError("--sim runs a simulated YAM only; MakerMods arms have no simulator here")
    from dream_hackathon.arms import makermods

    return makermods.MakerModsArms(station, energize=energize)


def open_ik(station: config.StationConfig) -> dict[str, ik.ArmIK]:
    """Per side, the teleop IK of a station's arms.

    @throws ConfigError A MakerMods arm's tool center point has not been measured yet.
    """
    if station.family == "yam":
        from dream_hackathon.arms import yam

        return yam.teleop_ik(station)
    from dream_hackathon.arms import makermods

    return makermods.teleop_ik(station)
