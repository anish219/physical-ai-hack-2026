#!/usr/bin/env python3
"""Serve the unified station page with fake arms. Does not talk to the NUC or motors."""

import argparse
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from dream_hackathon.so101_dashboard import Controls
from dream_hackathon.unified_control import UnifiedController, UnifiedPage


class PreviewArm:
    def __init__(self, side):
        self.side, self.port = side, 0
        self.control = SimpleNamespace(post=lambda *_a, **_k: {})
        self._state = dict(
            arm_open=False,
            enabled=False,
            requested_power=False,
            homing=False,
            zero_joints=[0.0] * 6,
            grip_torque_nm=1.0,
            grip_applied_torque_nm=None,
            grip_measured_torque_nm=None,
            grip_holding=False,
            message="UI preview · no hardware",
            error=None,
            problems=[],
        )

    def snapshot(self):
        return dict(self._state)

    def cancel_home(self):
        self._state["homing"] = False

    def set_torque(self, torque):
        self._state["grip_torque_nm"] = float(torque)

    def power(self, enabled):
        self._state.update(
            arm_open=enabled,
            enabled=enabled,
            requested_power=False,
            grip_applied_torque_nm=self._state["grip_torque_nm"] if enabled else None,
            message="Preview arm on" if enabled else "UI preview · no hardware",
            problems=[],
        )
        return self.snapshot()

    def return_zero(self):
        self._state["homing"] = False
        self._state["message"] = "Preview return to 0 (no motors)"
        return self.snapshot()

    def set_zero(self):
        if not self._state["arm_open"]:
            raise ValueError("Turn on the arm to set zero from its current measured pose")
        self._state["message"] = "Preview zero saved"
        return self.snapshot()


class PreviewLeader:
    def __init__(self, kind="so101"):
        self.kind, self.side, self.process = kind, None, None
        self._controls = Controls(1.0, "none", dry_run=True)

    def state(self):
        state = self._controls.snapshot()
        state["message"] = f"{self.kind} preview · no leader hardware"
        return state

    def select(self, side):
        self.side = side
        self._controls = Controls(1.0, side or "none", dry_run=True)

    def action(self, path, body):
        if path.endswith("/reject"):
            return self._controls.request_reject()
        if path.endswith("/kill"):
            self._controls = Controls(1.0, self.side or "none", dry_run=True)
            return self.state()
        return self._controls.change(body or {})

    def close(self):
        pass


class PreviewCaller:
    def __init__(self, selection):
        self.selection = selection
        self.sides = ("left", "right") if selection == "both" else (selection,)
        self._state = dict(busy=False, observation=None, rows=None, error="", observation_error="", mode="idle", calls=0, commands=0, running=False)

    def snapshot(self):
        return dict(self._state)

    def connect(self):
        self._state["mode"] = "connected"

    def start(self, values, *, run):
        self._state.update(busy=True, running=run, mode="run" if run else "call", calls=self._state["calls"] + 1)

    def cancel_playback(self):
        self._state.update(busy=False, running=False, mode="idle")

    def stop(self):
        self.cancel_playback()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8082, help="Local preview port (default 8082, not the NUC)")
    args = parser.parse_args(argv)
    nodes = {"left": PreviewArm("left"), "right": PreviewArm("right")}
    callers = {side: PreviewCaller(side) for side in ("left", "right", "both")}
    cache = Path.home() / ".cache" / "dream_hackathon"
    controller = UnifiedController(
        nodes,
        PreviewLeader("so101"),
        callers,
        sequence_path=cache / "preview-sequences.json",
        flow_path=cache / "preview-flows.json",
        star=PreviewLeader("star"),
        model_loader=lambda: [{"id": "dreamlabs/dream-1-beta", "label": "preview checkpoint"}],
    )
    page = UnifiedPage(controller, args.port)
    print(f"Local UI preview (no arms, not the NUC): http://127.0.0.1:{page.port}/", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        page.close()


if __name__ == "__main__":
    main()
