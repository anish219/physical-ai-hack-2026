"""Local event-driven demo coordinator. Emits tasks; does not command hardware."""

import argparse
import json
import math
import pathlib
import threading
from dataclasses import asdict, dataclass

STEPS = {
    "usb": ("locate", "grasp", "extract", "release", "verify_removed"),
    "stack": ("place_base_horizontal", "place_second_horizontal", "place_top_vertical", "release", "verify_tower"),
    "balls": ("locate_assigned_color", "grasp", "deposit_white_bin", "verify_deposit"),
}


@dataclass
class ArmTask:
    task: str
    cycle: int = 0
    step: int = 0

    def token(self, side: str) -> str:
        return f"{side}:{self.task}:{self.cycle}:{self.step}"


class DemoMaster:
    """One owner serializes events from both arm workers.

    Phase 2 begins after one verified USB. Right can finish extra USBs while
    left stacks. Balls begin only after both gates, independently per ready arm.
    Each action completion needs deterministic AND VLM evidence for its token.
    """

    def __init__(self, assigned_color: str, usb_target: int = 1):
        if not assigned_color.strip() or not 1 <= usb_target <= 4:
            raise ValueError("Choose an assigned color and USB target from 1 through 4")
        self.assigned_color, self.usb_target = assigned_color, usb_target
        self.phase = 1
        self.usb_removed = 0
        self.usb_gate = False
        self.tower_gate = False
        self.balls = {"left": 0, "right": 0}
        self.arms = {"left": ArmTask("wait"), "right": ArmTask("usb")}
        self.history: list[dict] = []
        self.lock = threading.RLock()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "phase": self.phase,
                "usb_gate": self.usb_gate,
                "tower_gate": self.tower_gate,
                "usb_removed": self.usb_removed,
                "usb_target": self.usb_target,
                "assigned_color": self.assigned_color,
                "verified_deposits": dict(self.balls),
                "arms": {
                    side: {
                        **asdict(task),
                        "step_name": STEPS[task.task][task.step] if task.task != "wait" else "wait_for_phase_gate",
                        "token": task.token(side),
                    }
                    for side, task in self.arms.items()
                },
            }

    def complete(
        self, side: str, token: str, deterministic: bool, vlm: str, evidence: dict
    ) -> bool:
        """Evidence carries frame/measurement references for local audit.

        USB final step additionally requires independently measured and visually
        confirmed cumulative removal counts. Tower final step supplies calibrated
        height and continuous free-standing duration from a deterministic tracker,
        plus visual confirmations. VLM alone never measures inches or duration.
        """
        with self.lock:
            task = self.arms[side]
            if token != task.token(side) or task.task == "wait":
                raise ValueError("Completion token does not identify the current action")
            if deterministic is not True or vlm != "yes":
                self.history.append(dict(side=side, token=token, accepted=False, evidence=evidence))
                return False
            final = task.step == len(STEPS[task.task]) - 1
            if final and task.task == "usb":
                expected = self.usb_removed + 1
                if not (
                    type(evidence.get("measured_removed")) is int
                    and type(evidence.get("visual_removed")) is int
                    and expected <= evidence["measured_removed"] <= 4
                    and expected <= evidence["visual_removed"] <= 4
                ):
                    return False
            if final and task.task == "stack":
                height = evidence.get("height_inches")
                duration = evidence.get("continuous_freestanding_s")
                if not (
                    type(height) in (float, int) and math.isfinite(height) and height >= 4
                    and type(duration) in (float, int) and math.isfinite(duration) and duration >= 3
                    and evidence.get("visual_freestanding") is True
                    and evidence.get("visual_height_at_least_4") is True
                ):
                    return False
            self.history.append(dict(side=side, token=token, accepted=True, evidence=evidence))
            if not final:
                task.step += 1
                return True
            if task.task == "usb":
                self.usb_removed += 1
                self.usb_gate = True
                if self.phase == 1:
                    self.phase = 2
                    self.arms["left"] = ArmTask("stack")
                self.arms[side] = (
                    ArmTask("usb", task.cycle + 1) if self.usb_removed < self.usb_target else ArmTask("wait")
                )
            elif task.task == "stack":
                self.tower_gate = True  # Latched: a later collapse cannot revoke qualification.
                self.phase = 3
                self.arms[side] = ArmTask("wait")
            else:
                self.balls[side] += 1
                self.arms[side] = ArmTask("balls", task.cycle + 1)
            if self.phase == 3:
                for arm, current in self.arms.items():
                    if current.task == "wait":
                        self.arms[arm] = ArmTask("balls")
            return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", required=True)
    parser.add_argument("--usb-target", type=int, default=1)
    parser.add_argument("--events", type=pathlib.Path, help="Replay local JSONL completion events")
    args = parser.parse_args()
    master = DemoMaster(args.color, args.usb_target)
    print(json.dumps(master.snapshot()))
    if args.events:
        for line in args.events.read_text(encoding="utf-8").splitlines():
            if line.strip():
                accepted = master.complete(**json.loads(line))
                print(json.dumps({"accepted": accepted, "state": master.snapshot()}))


if __name__ == "__main__":
    main()
