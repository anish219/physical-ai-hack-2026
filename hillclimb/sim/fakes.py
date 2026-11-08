"""Stand-ins for the gate and the stacking backends, so the left controller can be tested alone (and against the
station sim). FakeBackend follows a script of outcomes; with a sim URL it also moves a loose block onto the tower
through the sim's /reset, so the cameras show what the records say."""
from __future__ import annotations

import json
import time
import urllib.request


class FakeGate:
    """Opens (and latches) open_after_s seconds after the first call."""

    def __init__(self, open_after_s: float = 3.0, unplugged: int = 1):
        self.t0: float | None = None
        self.open_after_s, self.unplugged = open_after_s, unplugged

    def __call__(self) -> dict:
        if self.t0 is None:
            self.t0 = time.time()
        is_open = time.time() - self.t0 >= self.open_after_s
        return {"known": True, "open": is_open, "unplugged_count": self.unplugged if is_open else 0,
                "plugged_count": 4 - (self.unplugged if is_open else 0), "gripper_released": is_open, "cable_on_table": is_open,
                "fully_inserted": False, "checks": {"socket_emptied": "yes" if is_open else "no", "jaw_released": "yes" if is_open else "no",
                                                    "fast_vlm": "yes" if is_open else "no", "agentic_vlm": "yes" if is_open else "no", "healthy": "yes"}}


class FakeBackend:
    """outcomes: a list cycled per place_next: "ok", "fail" (no growth), "raise". Each step takes step_s seconds.
    It also tries one arm.target() per step, so the controller's motion guard is exercised."""

    def __init__(self, arm=None, outcomes=("ok",), step_s: float = 0.5, sim_url: str | None = None):
        self.arm, self.outcomes, self.step_s, self.sim_url = arm, list(outcomes), step_s, sim_url
        self.i = 0
        self.calls: list[str] = []
        self.tower_xy = (0.31, 0.02)
        self.ids = ["orange", "purple", "green", "yellow#1", "yellow#2", "blue#1", "blue#2", "blue#3"]
        self.placed: list[str] = []
        self.refused_motion = 0

    def _try_motion(self):
        if self.arm is None:
            return
        st = self.arm.state()
        q = (st or {}).get("goal") or (st or {}).get("q")
        if q and not self.arm.target(q, (st or {}).get("goal_grip", 0.8)):
            self.refused_motion += 1

    def prepare(self, scene):
        self.calls.append("prepare")
        return {"ok": True}

    def place_next(self, scene, deadline_s):
        self.calls.append("place_next")
        out = self.outcomes[self.i % len(self.outcomes)]
        self.i += 1
        self._try_motion()
        time.sleep(self.step_s)
        if out == "raise":
            raise RuntimeError("fake backend failure")
        if out == "fail":
            return {"ok": True, "grew": False, "why": "missed grasp", "verified": True}
        pid = self.ids[len(self.placed) % len(self.ids)]
        self.placed.append(pid)
        self._sim_place()
        return {"ok": True, "grew": True, "verified": True, "placed_id": pid, "timing": {"total_s": self.step_s}}

    def recover(self, scene, deadline_s):
        self.calls.append("recover")
        time.sleep(self.step_s)
        return {"ok": True, "grew": False, "tower_blocks": len(scene.get("tower", [])), "why": "nothing to fix", "verified": True}

    def retreat(self):
        self.calls.append("retreat")
        return {"ok": True}

    def hold(self):
        self.calls.append("hold")
        return {"ok": True}

    def _sim_place(self):
        if not self.sim_url:
            return
        spots = [(0.22, -0.12), (0.22, 0.15), (0.38, -0.11), (0.40, 0.15), (0.30, -0.16), (0.30, 0.19), (0.20, 0.02), (0.43, 0.03)]
        blocks = {}
        for lvl, pid in enumerate(self.placed):
            blocks[pid] = {"x": self.tower_xy[0], "y": self.tower_xy[1], "yaw": 0.0, "level": lvl}
        for k, pid in enumerate([i for i in self.ids if i not in self.placed]):
            blocks[pid] = {"x": spots[k][0], "y": spots[k][1], "yaw": 0.3, "level": 0}
        try:
            req = urllib.request.Request(self.sim_url + "/reset", json.dumps({"blocks": blocks, "q": "look", "grip": 0.8}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:  # noqa: BLE001
            pass


def make_backend(settings: dict, arm):
    return FakeBackend(arm)
