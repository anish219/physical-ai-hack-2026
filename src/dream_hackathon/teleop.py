"""Clutched delta-pose teleoperation: while the grip button is held, the tool follows the hand.

Pressing the grip engages the clutch and records where the controller and the tool are. From then on, the tool's
target is its pose at engaging moved by the controller's motion since engaging, both turned into the arm's base
frame. Releasing the grip freezes the tool where it is, so the operator can re-position the hand and engage again,
like lifting a mouse off the table. The arm never jumps to the controller's absolute pose.

@see Clutch
"""

import dataclasses

import numpy as np
import numpy.typing as npt

## Grip-button values that engage and release the clutch; the gap keeps a half-pressed grip from chattering.
GRIP_ENGAGE = 0.6
GRIP_RELEASE = 0.4

# The existing collector Home control and the SO-101 Return to 0 button share these.
HOME_SPEED_RAD_S = 0.5
HOME_TOLERANCE_RAD = 1e-3


def home_step(current: npt.NDArray[np.float64], start: npt.NDArray[np.float64], dt_s: float):
    """One tick toward the captured startup pose, using the collector's Home motion."""
    joints = step_toward(current, start, HOME_SPEED_RAD_S * dt_s)
    return joints, bool(np.all(np.abs(joints - start) <= HOME_TOLERANCE_RAD))


@dataclasses.dataclass
class Clutch:
    """One arm's clutch.

    Attributes:
    - quest_to_base: 3x3 rotation from the Quest's frame into the arm's base frame.
    - workspace_min_m, workspace_max_m: The box the tool target is kept inside, in the base frame.
    - translation_scale: Tool motion per unit of hand motion.
    - engaged: Whether the tool follows the hand now.
    """

    quest_to_base: npt.NDArray[np.float64]
    workspace_min_m: npt.NDArray[np.float64]
    workspace_max_m: npt.NDArray[np.float64]
    translation_scale: float
    engaged: bool = False
    _controller_start: npt.NDArray[np.float64] | None = None
    _tool_start: npt.NDArray[np.float64] | None = None

    def update(
        self,
        grip: float,
        controller: npt.NDArray[np.float64] | None,
        tool: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64] | None:
        """The tool's target this tick.

        @param grip The grip button, 0 released to 1 fully pressed.
        @param controller The controller's `[4, 4]` pose in the Quest's frame, or None when it is not tracked.
        @param tool The tool's `[4, 4]` pose now (of the joints last commanded), in the base frame.
        @return The `[4, 4]` target in the base frame, or None while disengaged: hold the arm where it is.
        """
        if controller is None or (self.engaged and grip < GRIP_RELEASE):
            self.engaged = False
            return None
        if not self.engaged:
            if grip < GRIP_ENGAGE:
                return None
            self.engaged = True
            self._controller_start = controller.copy()
            self._tool_start = tool.copy()
        return self._target(controller)

    def _target(self, controller: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert self._controller_start is not None and self._tool_start is not None
        rotation = self.quest_to_base
        moved = rotation @ (controller[:3, 3] - self._controller_start[:3, 3]) * self.translation_scale
        # The hand's turn since engaging, as a rotation in the base frame, applied to the tool's orientation.
        turned = rotation @ controller[:3, :3] @ self._controller_start[:3, :3].T @ rotation.T
        start = self._tool_start[:3, 3]
        target = np.eye(4)
        target[:3, :3] = turned @ self._tool_start[:3, :3]
        # The box is widened to hold the start, so a tool engaged outside it is never pulled in with a jump.
        target[:3, 3] = np.clip(
            start + moved, np.minimum(self.workspace_min_m, start), np.maximum(self.workspace_max_m, start)
        )
        return target


def step_toward(
    current: npt.NDArray[np.float64],
    goal: npt.NDArray[np.float64],
    max_step: float,
) -> npt.NDArray[np.float64]:
    """Moves joints toward a goal by at most `max_step` each, for homing at a bounded speed.

    @param current The joints now.
    @param goal The joints to reach.
    @param max_step The largest change of any joint, in the joints' unit.
    @return The joints after one step.
    """
    return current + np.clip(goal - current, -max_step, max_step)
