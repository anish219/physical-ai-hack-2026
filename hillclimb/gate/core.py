"""The gate's decision rule, with no I/O, so it can be tested with fake reads.

Opens only when all of these hold at once, then latches for the round (SPEC section 7):
  1. socket: the verifier's proof is "yes", or it is "unknown" and SOCKET_REQUIRED is off (it never says plain "no")
  2. jaw: a hold-then-open release of the right jaw happened since the gate was armed
  3. vision: two agentic overhead checks in a row said yes, the second on a "now" frame at least AGENTIC_MIN_GAP_S
     newer than the first; any fast "no", agentic "no", error or timeout in between resets the streak
  4. healthy: the overhead frames and the right arm's state are fresh at the moment of opening
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateCore:
    min_gap_s: float = 1.5
    socket_required: bool = False
    armed: bool = False
    armed_at: float | None = None
    open: bool = False
    latched_at: float | None = None
    streak: int = 0
    last_yes_now_t: float | None = None
    fast: str = "unknown"
    agentic: str = "unknown"
    history: list = field(default_factory=list)

    def arm(self, t: float) -> None:
        self.armed, self.armed_at = True, t
        self.open, self.latched_at = False, None
        self.streak, self.last_yes_now_t = 0, None
        self.fast, self.agentic = "unknown", "unknown"
        self.history.clear()

    def _note(self, **row) -> None:
        self.history.append(row)
        del self.history[:-20]

    def fast_read(self, t: float, yes: bool, ok: bool) -> bool:
        """Records a fast read; returns True when the agentic check should run next."""
        if self.open or not self.armed:
            return False
        self.fast = "yes" if (ok and yes) else "no"
        self._note(t=t, kind="fast", yes=bool(ok and yes), ok=ok)
        if not (ok and yes):
            self.streak, self.last_yes_now_t = 0, None
            self.agentic = "no" if self.agentic == "yes" else self.agentic
            return False
        return True

    def agentic_read(self, t: float, now_t: float, yes: bool, ok: bool) -> None:
        if self.open or not self.armed:
            return
        good = ok and yes
        self.agentic = "yes" if good else "no"
        self._note(t=t, kind="agentic", yes=good, ok=ok, now_t=now_t)
        if not good:
            self.streak, self.last_yes_now_t = 0, None
            return
        if self.last_yes_now_t is None:
            self.streak, self.last_yes_now_t = 1, now_t
        elif now_t - self.last_yes_now_t >= self.min_gap_s:
            self.streak, self.last_yes_now_t = self.streak + 1, now_t
        # a yes on a frame not newer by min_gap_s is not fresh evidence: keep the streak, don't count it

    def checks(self, socket: str, jaw_released: bool, healthy: bool) -> dict:
        socket_ok = socket == "yes" or (socket == "unknown" and not self.socket_required)
        return {
            "socket_emptied": socket,
            "socket_ok": socket_ok,
            "jaw_released": "yes" if jaw_released else "no",
            "fast_vlm": self.fast,
            "agentic_vlm": self.agentic,
            "agentic_streak": self.streak,
            "healthy": "yes" if healthy else "no",
        }

    def try_open(self, t: float, socket: str, jaw_released: bool, healthy: bool) -> bool:
        if self.open:
            return True
        if not self.armed:
            return False
        c = self.checks(socket, jaw_released, healthy)
        if c["socket_ok"] and jaw_released and healthy and self.streak >= 2:
            self.open, self.latched_at = True, t
            self._note(t=t, kind="open", checks=c)
        return self.open
