"""Stand-in backends for the stress tests.

ScriptLike mirrors hillclimb.runner.backend.RunnerBackend's contract as read on Nov 8 (no runner code is run):
  - it talks to the arm server directly with urllib, NOT through the controller's guarded ArmClient (as opstack does)
  - place_next refuses while GATE_FILE is missing; prepare and recover never look at GATE_FILE
  - hold() sets a module-global abort event; prepare/place_next/recover clear it at their start (O.ABORT)
  - an arm-off check (q null, grip null, enabled false, or unreachable) returns a not-ok result before any motion
  - recover() separates every raised block (tears the tower down) and returns ok with no tower_blocks
  - place_next returns verified=True with the whole tower list
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

ABORT = threading.Event()  # module-global, like opstack.ABORT


class ScriptLike:
    def __init__(self, url: str, step_s: float = 0.4, prepare_s: float = 0.2, moves: int = 2, raise_on: tuple = (),
                 recover_s: float = 0.4):
        self.url, self.step_s, self.prepare_s, self.moves, self.recover_s = url, step_s, prepare_s, moves, recover_s
        self.raise_on = set(raise_on)
        self.tower: list[str] = []
        self.prepared = False
        self.calls: list[tuple[float, str]] = []
        self.moves_sent = 0
        self.n = 0

    def _state(self) -> dict:
        with urllib.request.urlopen(self.url + "/state", timeout=5) as r:
            return json.loads(r.read())

    def _off(self, what: str):
        try:
            st = self._state()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "grew": False, "arm_off": True, "why": f"left arm server unreachable: {type(e).__name__}: {e}"[:200]}
        if not st or st.get("q") is None or st.get("grip") is None or st.get("enabled") is False:
            return {"ok": False, "grew": False, "arm_off": True,
                    "why": "LEFT ARM IS OFF: its server reports no joint readings or not enabled; power it on, then press Start"}
        return None

    def _move(self) -> None:
        req = urllib.request.Request(self.url + "/move", json.dumps({"q": [0] * 6, "grip": 0.8, "rate": 0.3}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()  # an HTTP 400 raises here, as opstack._post does
        self.moves_sent += 1

    def _maybe_raise(self, what: str) -> None:
        if what in self.raise_on or "all" in self.raise_on:
            raise RuntimeError(f"injected {what} failure")

    def prepare(self, scene=None):
        self.calls.append((time.time(), "prepare"))
        ABORT.clear()
        off = self._off("prepare")
        if off:
            return off
        self._maybe_raise("prepare")
        time.sleep(self.prepare_s)
        self._move()  # the look poses
        self.prepared = True
        return {"ok": True, "what": "prepare"}

    def place_next(self, scene=None, deadline_s=120.0):
        self.calls.append((time.time(), "place_next"))
        gate = os.environ.get("GATE_FILE")
        if gate and not os.path.exists(gate):
            return {"ok": False, "grew": False, "verified": False, "why": "USB gate shut", "tower": list(self.tower)}
        ABORT.clear()
        off = self._off("place_next")
        if off:
            return off
        self._maybe_raise("place_next")
        if not self.prepared:
            r = self.prepare(scene)
            if not r.get("ok"):
                return {"ok": False, "grew": False, "why": r.get("why")}
        for _ in range(self.moves):
            if ABORT.is_set():
                raise SystemExit("stopped by the harness")
            self._move()
            time.sleep(self.step_s / self.moves)
        self.n += 1
        self.tower.append(f"b{self.n}")
        return {"ok": True, "grew": True, "verified": True, "placed_id": f"b{self.n}", "tower": list(self.tower)}

    def recover(self, scene=None, deadline_s=180.0):
        self.calls.append((time.time(), "recover"))
        ABORT.clear()
        off = self._off("recover")
        if off:
            return off
        self._maybe_raise("recover")
        for _ in range(max(1, len(self.tower))):  # one move per raised block: the tower comes apart
            self._move()
            time.sleep(self.recover_s / max(1, len(self.tower)))
        self.tower, self.prepared = [], False
        return {"ok": True, "what": "recover"}

    def retreat(self):
        self.calls.append((time.time(), "retreat"))
        try:
            self._move()
            return {"ok": True}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "why": repr(e)}

    def hold(self):
        self.calls.append((time.time(), "hold"))
        ABORT.set()
        try:
            urllib.request.urlopen(urllib.request.Request(self.url + "/stop", b"{}", {"Content-Type": "application/json"}), timeout=5).read()
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True}

    def count(self, what: str, since: float = 0.0) -> int:
        return sum(1 for t, w in self.calls if t >= since and w == what)


class SwitchGate:
    """The controller's gate function, switchable from a test."""

    def __init__(self, is_open: bool = False, known: bool = True):
        self.open, self.known, self.raise_ = is_open, known, False

    def __call__(self) -> dict:
        if self.raise_:
            raise RuntimeError("gate read failed")
        return {"known": self.known, "open": self.open, "unplugged_count": 1 if self.open else 0, "plugged_count": 3 if self.open else 4}
