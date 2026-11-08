"""Competition gates and independent arm scheduling, without any hardware."""

import pytest

from dream_hackathon.demo_master import STEPS, DemoMaster

USB = {"measured_removed": 1, "visual_removed": 1}
TOWER = {
    "height_inches": 5.0,
    "continuous_freestanding_s": 3.0,
    "visual_freestanding": True,
    "visual_height_at_least_4": True,
}


def finish(master, side, evidence):
    task = master.arms[side].task
    for _ in STEPS[task]:
        assert master.complete(side, master.arms[side].token(side), True, "yes", evidence)


def test_global_gates_and_latched_tower():
    master = DemoMaster("orange")
    assert master.snapshot()["arms"]["left"]["task"] == "wait"
    finish(master, "right", USB)
    assert master.phase == 2 and master.usb_gate
    assert master.arms["right"].task == "wait"
    finish(master, "left", TOWER)
    assert master.phase == 3 and master.tower_gate
    assert all(task.task == "balls" for task in master.arms.values())
    # Ball evidence contains no standing tower; qualification remains latched.
    finish(master, "right", {"frame": "after-collapse.jpg"})
    assert master.tower_gate and master.balls == {"left": 0, "right": 1}


def test_concurrent_extra_usbs_and_stack():
    master = DemoMaster("blue", usb_target=4)
    finish(master, "right", USB)
    assert master.arms["right"].task == "usb"
    assert master.arms["left"].task == "stack"
    finish(master, "left", TOWER)
    assert master.arms["left"].task == "balls"
    assert master.arms["right"].task == "usb"
    for count in range(2, 5):
        finish(master, "right", {"measured_removed": count, "visual_removed": count})
    assert master.arms["right"].task == "balls"


@pytest.mark.parametrize("deterministic,vlm", [(False, "yes"), (True, "no"), (True, "unknown")])
def test_each_step_needs_both_verifiers(deterministic, vlm):
    master = DemoMaster("blue")
    token = master.arms["right"].token("right")
    assert not master.complete("right", token, deterministic, vlm, {})
    assert master.arms["right"].token("right") == token and master.phase == 1


def test_duplicate_or_wrong_arm_completion_cannot_advance():
    master = DemoMaster("blue")
    token = master.arms["right"].token("right")
    master.complete("right", token, True, "yes", {})
    with pytest.raises(ValueError):
        master.complete("right", token, True, "yes", {})
    with pytest.raises(ValueError):
        master.complete("left", master.arms["left"].token("left"), True, "yes", {})


@pytest.mark.parametrize("evidence", [{}, {"measured_removed": 1, "visual_removed": 0}])
def test_usb_gate_needs_actual_removal(evidence):
    master = DemoMaster("blue")
    for _ in range(len(STEPS["usb"]) - 1):
        master.complete("right", master.arms["right"].token("right"), True, "yes", {})
    assert not master.complete("right", master.arms["right"].token("right"), True, "yes", evidence)
    assert master.phase == 1


@pytest.mark.parametrize(
    "change",
    [
        {"height_inches": 3.5},
        {"continuous_freestanding_s": 2.99},
        {"visual_freestanding": False},
        {"visual_height_at_least_4": False},
        {"height_inches": float("nan")},
    ],
)
def test_tower_gate_cannot_advance_early(change):
    master = DemoMaster("blue")
    finish(master, "right", USB)
    for _ in range(len(STEPS["stack"]) - 1):
        master.complete("left", master.arms["left"].token("left"), True, "yes", {})
    assert not master.complete("left", master.arms["left"].token("left"), True, "yes", TOWER | change)
    assert master.phase == 2 and not master.tower_gate
    assert master.arms["right"].task == "wait"
