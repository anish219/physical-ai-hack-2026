"""Explicit selected-gripper holding-effort commands and cached motor feedback."""


def apply_hold_torque(robot, side: str, torque_nm: float) -> None:
    if hasattr(robot, "set_hold_torque"):
        robot.set_hold_torque(torque_nm)
        return
    robot._devices[side].bus.apply_hold_torque(torque_nm)


def feedback(robot, side: str) -> dict:
    if hasattr(robot, "gripper_feedback"):
        return robot.gripper_feedback()
    bus = robot._devices[side].bus
    with bus._lock:
        measured = bus._base._last_known_states.get("gripper", {}).get("torque")
        return {
            "grip_measured_torque_nm": float(measured) if measured is not None else None,
            "grip_applied_torque_nm": float(bus.controller.torque_nm),
            "grip_holding": bool(bus._enabled and bus.controller.holding),
        }
