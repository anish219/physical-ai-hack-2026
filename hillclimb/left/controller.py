"""Left arm stacking controller: the round clock, the state machine of SPEC section 8, and the score estimate.

States: IDLE (not armed) -> LOCKED (gate shut, no motion) -> STACK -> SETTLE_CHECK -> STACK | REST; REST -> RECOVER ->
STACK; any -> FREEZE (last freeze_s: no new placement) -> END; FAULT (a backend failed or raised: hold, wait, recover).

It never raises out of its loop: every tick and every backend call is wrapped, and a failure becomes a state, a log
row and a health flag. The state is written to disk every second, so a restarted process resumes the round from the
wall clock with the tower count it had.

Backends implement INTERFACES.md's StackBackend and are built by `make_backend(settings, arm)` in hillclimb.runner
(scripted) or hillclimb.dream (Dream); "fake" comes from hillclimb.sim.fakes for tests.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import pathlib
import threading
import time
import traceback
from typing import Any, Callable

from hillclimb.common import vision
from hillclimb.common.arm import ArmClient
from hillclimb.common.config import CFG
from hillclimb.common.log import JsonlLog

IN = 25.4


@dataclasses.dataclass
class Settings:
    backend: str = "scripted"  # scripted | dream | fake
    checkpoint: str = "dream-1-ft:f9ed26ed-blocks-left-fresh500-20261108"
    wording: str = ""  # empty: the checkpoint's own table
    mode: str = "practice"  # practice | match (match locks every setting at go)
    gate_bypass: bool = False  # practice only: treat the USB gate as open (stacking tests without the USB arm); refused in match
    drive: str = "auto"  # auto: this controller places blocks; external: the main controller calls /left/stack_next
    round_s: float = 300.0
    max_goal: int = 9
    crunch_at_s: float = 90.0  # time left at which crunch_goal takes over
    crunch_goal: int = 5
    freeze_s: float = 20.0
    block_height_mm: float = 34.5
    stage2_in: float = 4.0
    settle_s: float = 3.0
    rest_check_s: float = 5.0
    step_timeout_s: float = 120.0
    fault_wait_s: float = 3.0
    vision_confirm: bool = True  # cross-check each placement with Terra when the backend can't verify on its own
    bonus_livekit: bool = False
    bonus_dimensional: bool = False


class LeftController:
    STATES = ("IDLE", "LOCKED", "STACK", "SETTLE_CHECK", "REST", "RECOVER", "FREEZE", "END", "FAULT")

    def __init__(self, gate: Callable[[], dict] | None = None, arm: ArmClient | None = None,
                 overhead_arm: ArmClient | None = None, state_file: str | None = None, log_dir: str | None = None,
                 backend_factory: Callable[[Settings, ArmClient], Any] | None = None, clock: Callable[[], float] = time.time):
        self.log = JsonlLog("left", log_dir)
        self.clock = clock
        self.gate_fn = gate or default_gate
        self.arm = arm or ArmClient(CFG["LEFT_URL"])
        self.arm.guard = self.motion_allowed
        self.overhead = overhead_arm or (self.arm if CFG["OVERHEAD_FROM"] == "left" else ArmClient(CFG["RIGHT_URL"]))
        self.state_file = pathlib.Path(state_file or CFG["STATE_FILE"])
        self.backend_factory = backend_factory
        self.lock = threading.RLock()
        self.settings = Settings()
        self.state = "IDLE"
        self.reason = ""
        self.go_wall: float | None = None
        self.tower: list[dict] = []  # confirmed placements, bottom first
        self.stage2_passed = False
        self.estop = False
        self.state_since = clock()
        self.last_step: dict = {}
        self.faults = 0
        self.fell_reads = 0
        self.last_rest_check = 0.0
        self.gate_cache: dict = {"known": False, "open": False}
        self.health: dict = {}
        self.backend = None
        self._job: threading.Thread | None = None
        self._job_result: dict | None = None
        self._prepared = False
        self._stop = threading.Event()
        self._waiting = False
        self._last_save = 0.0
        self._load()

    # ---------------------------------------------------------------- settings and round control

    def arm_round(self, cfg: dict) -> tuple[bool, str]:
        """Apply settings and get ready (IDLE or after END). Refused once a match round has started."""
        with self.lock:
            if self.go_wall is not None and self.state not in ("END", "IDLE"):  # any mode: re-arming mid-round would reset a moving arm
                return False, "a round is running: stop it first (settings are locked until it ends)"
            new = dataclasses.replace(self.settings)
            for k, v in (cfg or {}).items():
                if not hasattr(new, k):
                    return False, f"unknown setting {k}"
                cur = getattr(new, k)
                if isinstance(cur, bool):
                    if isinstance(v, str):
                        v = v.strip().lower() in ("1", "true", "yes", "on")
                    setattr(new, k, bool(v))
                else:
                    setattr(new, k, type(cur)(v))
            if new.mode != "practice" and "gate_bypass" not in (cfg or {}):
                new.gate_bypass = False  # a bypass left over from a practice round never carries into a match
            if new.gate_bypass and new.mode != "practice":
                return False, "gate_bypass is practice only"
            if new.gate_bypass and os.environ.get("HILLCLIMB_MATCH_LOCK") == "1":
                return False, "HILLCLIMB_MATCH_LOCK=1 on this station: the USB gate can't be bypassed"
            if new.crunch_goal > new.max_goal:
                return False, "crunch_goal must not exceed max_goal"
            self.settings = new
            self.go_wall, self.tower, self.stage2_passed, self.estop = None, [], False, False
            self.faults, self.last_step, self._prepared = 0, {}, False
            self._gate_file(False)
            self._set("IDLE", "armed")
            self.backend = self._make_backend()
            if new.mode == "match" and hasattr(self.backend, "table_reset"):
                # a new match starts on a table the judges reset: towers from earlier rounds are no longer there
                self._safe(lambda: self.backend.table_reset("match round armed"))
            self.log.write("arm", settings=dataclasses.asdict(new))
            self._save(force=True)
            return True, "armed"

    def _gate_file(self, is_open: bool) -> None:
        """The scripted runner holds its first pick until GATE_FILE exists: cleared at arm, written when the gate opens."""
        path = os.environ.setdefault("GATE_FILE", os.path.join(os.path.dirname(self.state_file) or ".", "usb_gate_open"))
        try:
            if is_open:
                with open(path, "w") as f:
                    f.write(str(self.clock()))
            elif os.path.exists(path):
                os.remove(path)
        except OSError as e:
            self.log.write("gate_file_error", error=str(e))

    def go(self) -> tuple[bool, str]:
        with self.lock:
            if self.go_wall is not None and self.state != "END":
                return False, "round already running"
            if self.backend is None:
                self.backend = self._make_backend()
            if self.go_wall is not None:  # Go after END without a new arm: a clean round, nothing carried over
                self.tower, self.stage2_passed, self.faults, self.last_step, self._prepared = [], False, 0, {}, False
                self._job_result = None
                self._gate_file(False)
                if hasattr(self.backend, "prepared"):
                    self.backend.prepared = False
            self.go_wall, self.estop = self.clock(), False
            self._set("LOCKED", "go")
            self._save(force=True)
            return True, "go"

    def stop_round(self, why: str = "stop called") -> None:
        with self.lock:
            self._safe(lambda: self.backend and self.backend.hold())
            self.arm.stop()
            self._set("END", why)
            self._save(force=True)

    def emergency_stop(self) -> None:
        self.estop = True  # a plain flag write; /stop must not wait behind a tick holding the lock
        self.arm.stop()
        self._safe(lambda: self.backend and self.backend.hold())
        with self.lock:
            self._set("END", "estop")
            self._save(force=True)

    # ---------------------------------------------------------------- derived values

    def time_left(self) -> float | None:
        if self.go_wall is None:
            return None
        return max(0.0, self.settings.round_s - (self.clock() - self.go_wall))

    def active_goal(self) -> int:
        tl = self.time_left()
        s = self.settings
        return min(s.crunch_goal, s.max_goal) if tl is not None and tl <= s.crunch_at_s else s.max_goal

    def tower_in(self) -> float:
        return len(self.tower) * self.settings.block_height_mm / IN

    def gate_open(self) -> bool:
        if self.settings.gate_bypass and self.settings.mode == "practice":
            return True
        return bool(self.gate_cache.get("open")) and self.gate_cache.get("known", True) is not False

    def motion_allowed(self) -> bool:
        return (not self.estop and self.go_wall is not None and self.gate_open()
                and self.state in ("STACK", "SETTLE_CHECK", "RECOVER", "FREEZE", "REST") and (self.time_left() or 0) > 0)

    def score(self) -> dict:
        g = self.gate_cache
        unplugs = int(g.get("unplugged_count") or 0) if g.get("known") else 0
        plugins = int(g.get("plugged_in_count") or 0) if g.get("known") else 0
        usb = (10 + 5 * (unplugs - 1) if unplugs >= 1 else 0) + 20 * plugins
        tower = (20 + 3 * math.floor(self.tower_in())) if self.stage2_passed else 0
        bonus = 10 * self.settings.bonus_livekit + 5 * self.settings.bonus_dimensional
        return {"usb": usb, "tower": tower, "bonus": bonus, "total": usb + tower + bonus,
                "not_counted": ["taller tower +10 (bracket only)", "balls"], "unplugs": unplugs, "plug_ins": plugins}

    def call_stop_now(self) -> tuple[bool, str]:
        if not self.stage2_passed:
            return False, "stage 2 not passed"
        if len(self.tower) < self.active_goal() or self.state != "REST":
            return False, "still building"
        if self.clock() - self.state_since < 5:
            return False, "rest not held 5 s yet"
        g = self.gate_cache
        if not g.get("known") or int(g.get("plugged_count", 1) or 0) > 0:
            return False, "right arm may still score"
        return True, f"locks in {self.score()['total']} points"

    def snapshot(self) -> dict:
        with self.lock:
            stop_now, why = self.call_stop_now()
            return {"state": self.state, "reason": self.reason, "time_left_s": None if self.time_left() is None else round(self.time_left(), 1),
                    "tower_blocks": len(self.tower), "tower_in": round(self.tower_in(), 2), "active_goal": self.active_goal(),
                    "stage2_passed": self.stage2_passed, "last_step": self.last_step, "backend": self.settings.backend,
                    "settings": dataclasses.asdict(self.settings), "health": self.health, "gate_open": self.gate_open(),
                    "score": self.score(), "call_stop_now": stop_now, "call_stop_why": why, "estop": self.estop,
                    "busy": self._job is not None and self._job.is_alive(), "in_state_s": round(self.clock() - self.state_since, 1)}

    def verify_tower(self) -> dict:
        """The main controller's verify_tower contract, from the confirmed count plus a fresh vision read."""
        n = len(self.tower)
        frames = self._frames()
        ans = vision.tower_standing(frames, n) if frames.get("overhead") else None
        if ans is None:
            return {"known": self.go_wall is not None, "block_count": n, "collapsed": False, "standing_free": True, "source": "records"}
        fell = bool(ans.get("fell")) or not ans.get("standing", True)
        return {"known": True, "block_count": n if not fell else int(ans.get("tower_blocks") or 0), "collapsed": fell,
                "standing_free": not fell, "source": "records+vision", "vision": ans}

    # ---------------------------------------------------------------- the loop

    def start(self) -> None:
        threading.Thread(target=self._loop, name="left-tick", daemon=True).start()

    def shutdown(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001  the loop itself never dies
                self.log.write("tick_error", error=repr(e), tb=traceback.format_exc()[-800:])
                self.health["tick_error"] = repr(e)[:200]
            self.health["heartbeat"] = round(self.clock(), 2)
            time.sleep(0.25)

    def tick(self) -> None:
        self._refresh_gate()
        # the arm check is an HTTP call: made before taking the lock so Stop, E-stop and the page never wait on it
        pre_off = self._arm_off_reason() if self.go_wall is not None and self.state in ("LOCKED", "FAULT", "RECOVER") else None
        with self.lock:
            s, st = self.settings, self.state
            tl = self.time_left()
            if self.go_wall is None or st in ("IDLE", "END"):
                self._save()
                return
            if tl is not None and tl <= 0:
                if self._busy():
                    return  # let the placement in flight finish; motion_allowed already refuses new commands
                self._set("END", "time")
                self._safe(lambda: self.backend and self.backend.hold())
                self._save(force=True)
                return
            if self._busy():
                self._save()
                return
            if self._job_result is not None and not self._waiting:
                self._finish_job()
                self._save()
                return
            off = pre_off if st in ("LOCKED", "FAULT", "RECOVER") else None
            if off and st in ("LOCKED", "FAULT"):
                # nothing else starts while the arm is off. A round that is running turns its own arm on (POST /power,
                # holds position, no motion), as the team's arm client does; at most every 10 s, 3 tries a round
                tries = getattr(self, "_power_tries", (None, 0, 0.0))
                n = tries[1] if tries[0] == self.go_wall else 0
                was_on = getattr(self, "_arm_on_round", None) == self.go_wall  # someone turned it off mid-round: leave it off
                if was_on:
                    self.reason = "LEFT ARM WAS TURNED OFF DURING THE ROUND: not turning it back on (stop and Start again to resume)"
                    self._save()
                    return
                if "ARM IS OFF" in off and n < 3 and time.time() - (tries[2] if tries[0] == self.go_wall else 0.0) >= 10.0:
                    self._power_tries = (self.go_wall, n + 1, time.time())
                    self._arm_chk_t = 0.0  # re-read the arm right after the try
                    self._start_job("power", lambda: self._power_on())
                    self.reason = f"left arm is off: turning it on (try {n + 1} of 3)"
                else:
                    self.reason = off if n < 3 or "ARM IS OFF" not in off else off + "; 3 power-on tries failed"
                self._save()
                return
            if st in ("STACK", "SETTLE_CHECK", "REST", "RECOVER", "FAULT") and not self.gate_open():
                self._set("LOCKED", "USB gate not open: no block is touched")  # a resume, a reset gate, a lost reading
                self._save()
                return
            if off is None and ("ARM IS OFF" in (self.reason or "") or (self.reason or "").startswith("left arm is off")):
                self.reason = "left arm on"
            if st == "LOCKED":
                if not self._prepared:
                    self._prepared = True
                    self._start_job("prepare", lambda: self.backend.prepare(self._scene()))
                elif self.gate_open():
                    self._gate_file(True)
                    self._set("STACK", "gate open")
            elif st in ("STACK", "FREEZE"):
                if tl <= s.freeze_s:
                    if st != "FREEZE":
                        self._set("FREEZE", "freeze window")
                        self._start_job("retreat", lambda: self.backend.retreat())
                elif len(self.tower) >= self.active_goal():
                    self._set("REST", "goal met")
                    self._start_job("retreat", lambda: self.backend.retreat())
                elif s.drive == "auto":
                    self._start_job("place", lambda: self.backend.place_next(self._scene(), max(1.0, min(s.step_timeout_s, tl - s.freeze_s))))
            elif st == "REST":
                if len(self.tower) < self.active_goal() and tl > s.freeze_s:
                    self._set("STACK", "below goal")
                elif self.clock() - self.last_rest_check >= s.rest_check_s:
                    self.last_rest_check = self.clock()
                    self._check_fall()
            elif st == "RECOVER":
                if tl <= s.freeze_s:
                    self._set("FREEZE", "freeze window")
                elif s.drive == "auto":
                    self._start_job("recover", lambda: self.backend.recover(self._scene(), max(1.0, min(s.step_timeout_s, tl - s.freeze_s))))
            elif st == "FAULT":
                self._fault_streak = getattr(self, "_fault_streak", 0)
                if self.clock() - self.state_since >= min(30.0, s.fault_wait_s * (2 ** min(self._fault_streak, 4))):
                    self._fault_streak += 1
                    if self.backend is not None and hasattr(self.backend, "prepared"):
                        self.backend.prepared = False  # the next place_next looks again and re-plans around the standing tower
                    self._set("STACK", "after fault: looking again (the tower is kept; only a seen fall rebuilds)")
            self._save()

    # ---------------------------------------------------------------- one step at a time

    def stack_next(self) -> dict:
        """Main-controller binding: one placement attempt, synchronous. Refused unless the gate is open."""
        with self.lock:
            if self.settings.drive != "external":
                return {"ok": False, "grew": False, "tower_blocks": len(self.tower), "why": "controller is in auto drive"}
            if self.go_wall is None:
                self.go()
            if not self.gate_open():
                return {"ok": False, "grew": False, "tower_blocks": len(self.tower), "why": "gate shut"}
            if (self.time_left() or 0) <= self.settings.freeze_s:
                return {"ok": False, "grew": False, "tower_blocks": len(self.tower), "why": "freeze window"}
            if self.state in ("LOCKED", "REST", "RECOVER"):
                self._set("STACK", "binding")
            self._waiting = True
            self._start_job("place", lambda: self.backend.place_next(self._scene(), self.settings.step_timeout_s))
        return self._wait_job()

    def rebuild(self) -> dict:
        with self.lock:
            if not self.gate_open():
                return {"ok": False, "why": "gate shut", "tower_blocks": len(self.tower)}
            self._set("RECOVER", "binding")
            self._waiting = True
            self._start_job("recover", lambda: self.backend.recover(self._scene(), self.settings.step_timeout_s))
        return self._wait_job()

    def _wait_job(self) -> dict:
        self._waiting = True
        try:
            t_end = time.time() + self.settings.step_timeout_s + 30
            while self._busy() and time.time() < t_end:
                time.sleep(0.1)
            with self.lock:
                if self._job_result is not None:
                    return self._finish_job()
                return {"ok": False, "grew": False, "tower_blocks": len(self.tower), "why": "still running"}
        finally:
            self._waiting = False

    def _busy(self) -> bool:
        return self._job is not None and self._job.is_alive()

    def _start_job(self, kind: str, fn: Callable[[], Any]) -> None:
        if self.backend is None:
            self._set("FAULT", "no backend")
            return
        before = self._frames() if kind in ("place", "recover") else {}

        def run():
            t0 = time.time()
            res: dict = {"ok": False, "why": "job did not finish", "raised": True}
            try:
                got = fn() or {}
                res = dict(got) if isinstance(got, dict) else {"ok": bool(got)}
            except BaseException as e:  # noqa: BLE001  a backend failure is a state, never a dead thread
                res = {"ok": False, "why": f"backend raised {type(e).__name__}: {e}"[:300], "raised": True}
                self.log.write("backend_error", step=kind, tb=traceback.format_exc()[-800:])
            finally:
                res["kind"], res["s"] = kind, round(time.time() - t0, 1)
                res["_before"] = before
                self._job_result = res

        self._job_result = None
        self._job = threading.Thread(target=run, name=f"left-{kind}", daemon=True)
        self._job.start()
        self.log.write("job_start", kind=kind, state=self.state, tower_count=len(self.tower))

    def _finish_job(self) -> dict:
        res, self._job_result = self._job_result or {}, None
        before = res.pop("_before", {})
        kind = res.get("kind")
        out = {"ok": bool(res.get("ok")), "grew": False, "tower_blocks": len(self.tower), "why": res.get("why", "")}
        if res.get("arm_off"):  # the arm is off or unreachable: wait in FAULT (tick re-checks it), never count it as a failed step
            self.last_step = {**res, "counted": False}
            if kind == "prepare":
                self._prepared = False
            if self.state != "LOCKED":
                self._set("FAULT", res.get("why", "left arm is off"))
            self.log.write("step", **self.last_step)
            return out
        if kind == "power":  # a power-on try: the tick re-reads the arm; never a fault or a placement
            self.last_step = {**res, "counted": False}
            self.log.write("step", **self.last_step)
            return out
        if kind in ("prepare", "retreat"):
            self.last_step = {k: v for k, v in res.items()}
            if not res.get("ok", True) and res.get("raised"):
                self.health[f"{kind}_error"] = res.get("why")
            return out
        if res.get("raised") or not res.get("ok"):
            self.faults += 1
            self.last_step = {**res, "counted": False}
            if res.get("raised") or self.faults >= 3:
                self._set("FAULT", res.get("why", "failed"))
                self.faults = 0
            elif kind == "recover":
                self._set("STACK", "recover attempted")
            self.log.write("step", **self.last_step)
            return out
        self.faults = 0
        grew = False if kind == "recover" else self._confirm(res, before)
        if grew:
            self._fault_streak = 0
        if grew and isinstance(res.get("tower"), list) and res.get("verified"):
            # the scripted backend keeps the tower bottom first (a base already on the table, an adopted tower): that list is the count
            self.tower = [{"id": b, "t": round(self.clock(), 2), "level": i + 1} for i, b in enumerate(res["tower"])]
        elif grew:
            self.tower.append({"id": res.get("placed_id"), "t": round(self.clock(), 2), "level": len(self.tower) + 1})
            self._set("SETTLE_CHECK", "placed")
            self._settle_and_latch()
        elif kind == "recover":
            n = res.get("tower_blocks")
            if isinstance(n, int) and n >= 0:
                self.tower = self.tower[:n] if n <= len(self.tower) else self.tower + [{"id": None, "level": i + 1} for i in range(len(self.tower), n)]
            self._set("STACK", "recovered")
        else:
            self._set("STACK", "no growth")
        self.last_step = {**res, "counted": grew}
        self.log.write("step", **self.last_step, tower_count=len(self.tower))
        out.update(grew=grew, tower_blocks=len(self.tower))
        return out

    def _confirm(self, res: dict, before: dict) -> bool:
        """A placement counts only when the backend saw it AND, unless the backend verified it itself, vision agrees."""
        if not res.get("grew", res.get("ok")):
            return False
        if res.get("verified") or not self.settings.vision_confirm:
            return True
        after = self._frames()
        if not before.get("overhead") or not after.get("overhead"):
            return False
        ans = vision.grew_and_stands(before, after, len(self.tower))
        res["vision"] = ans
        return bool(ans and ans.get("grew_by_one") and ans.get("standing") and not ans.get("something_fell"))

    def _settle_and_latch(self) -> None:
        """Wait settle_s after the release, then latch stage 2 when the tower is tall enough and still standing."""
        time.sleep(0)  # the settle wait itself happens on the next ticks: SETTLE_CHECK holds until settle_s passes
        self._settle_until = self.clock() + self.settings.settle_s
        threading.Thread(target=self._settle_worker, daemon=True).start()

    def _settle_worker(self) -> None:
        while self.clock() < getattr(self, "_settle_until", 0):
            time.sleep(0.1)
        with self.lock:
            if self.state != "SETTLE_CHECK":
                return
            standing = True
            if self.settings.vision_confirm and self.tower_in() >= self.settings.stage2_in and not self.stage2_passed:
                ans = vision.tower_standing(self._frames(), len(self.tower))
                standing = bool(ans and ans.get("standing")) if ans is not None else True
            if standing and self.tower_in() >= self.settings.stage2_in:
                if not self.stage2_passed:
                    self.log.write("stage2_passed", tower=len(self.tower), inches=self.tower_in())
                self.stage2_passed = True
            self._set("REST" if len(self.tower) >= self.active_goal() else "STACK", "settled")

    def _check_fall(self) -> None:
        if not self.settings.vision_confirm:
            return
        ans = vision.tower_standing(self._frames(), len(self.tower))
        if ans is None:
            return
        fell = bool(ans.get("fell")) or not ans.get("standing", True) or int(ans.get("tower_blocks", len(self.tower))) < len(self.tower)
        self.fell_reads = self.fell_reads + 1 if fell and float(ans.get("confidence", 0)) >= 0.6 else 0
        if self.fell_reads >= 2:  # two looks agree before acting: never restack on one bad read
            n = max(0, int(ans.get("tower_blocks") or 0))
            self.log.write("fall", was=len(self.tower), now=n, vision=ans)
            self.tower = self.tower[:n]
            self.fell_reads = 0
            self._set("RECOVER", "tower fell")

    # ---------------------------------------------------------------- helpers

    def _refresh_gate(self) -> None:
        try:
            g = self.gate_fn() or {}
            self.gate_cache = g
            self.health["gate"] = "ok" if g.get("known", True) else "unknown"
        except Exception as e:  # noqa: BLE001
            self.gate_cache = {"known": False, "open": self.gate_cache.get("open") and self.gate_cache.get("latched", False)}
            self.health["gate"] = f"error: {e}"[:200]

    def _frames(self) -> dict:
        try:
            ov = self.overhead.fresh_frame("overhead", 1.0)
            wr = self.arm.fresh_frame("wrist", 1.0)
            self.health["cams"] = "ok" if ov and wr else "stale"
            return {"overhead": ov, "wrist": wr}
        except Exception:  # noqa: BLE001
            return {}

    def _scene(self) -> dict:
        return {"tower": list(self.tower), "tower_blocks": len(self.tower), "active_goal": self.active_goal(),
                "time_left_s": self.time_left(), "checkpoint": self.settings.checkpoint, "wording": self.settings.wording}

    def _make_backend(self):
        try:
            if self.backend_factory:
                return self.backend_factory(self.settings, self.arm)
            name = {"scripted": "hillclimb.runner", "dream": "hillclimb.dream", "fake": "hillclimb.sim.fakes"}[self.settings.backend]
            mod = importlib.import_module(name)
            b = mod.make_backend(dataclasses.asdict(self.settings), self.arm)
            self.health.pop("backend", None)
            return b
        except Exception as e:  # noqa: BLE001
            self.health["backend"] = f"cannot load {self.settings.backend}: {type(e).__name__}: {e}"[:300]
            self.log.write("backend_load_error", error=self.health["backend"])
            return None

    def _safe(self, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self.log.write("safe_call_error", error=repr(e)[:200])

    def _power_on(self) -> dict:
        st = self.arm.power_on()
        if st is None:
            return {"ok": False, "why": f"power-on failed: {self.arm.last_error}"}
        ok = bool(st.get("arm_open")) and not st.get("problems")
        self.log.write("power_on", ok=ok, message=st.get("message"), problems=st.get("problems"))
        return {"ok": ok, "why": "" if ok else ("power-on: " + "; ".join(st.get("problems") or [st.get("message") or "arm still off"]))[:200]}

    def _arm_off_reason(self) -> str | None:
        """None while the left arm answers with joints and is enabled; else a plain reason. Cached 1 s."""
        now = time.time()
        if now - getattr(self, "_arm_chk_t", 0.0) < 1.0:
            return getattr(self, "_arm_chk", None)
        st = self.arm.state()
        if st is None:
            why = "LEFT ARM SERVER NOT ANSWERING (" + str(getattr(self.arm, "last_error", ""))[:80] + ")"
        elif st.get("q") is None or st.get("enabled") is False:
            why = "LEFT ARM IS OFF (server says enabled false, no joints): power it on; the round waits, nothing moves"
        else:
            why = None
            if self.go_wall is not None:
                self._arm_on_round = self.go_wall  # seen on during this round
        self._arm_chk_t, self._arm_chk = now, why
        self.health["left_arm"] = why or "ok"
        return why

    def _set(self, st: str, why: str) -> None:
        if st != self.state:
            self.log.write("state", frm=self.state, to=st, why=why, tower=len(self.tower), time_left=self.time_left())
            self.state, self.state_since = st, self.clock()
        self.reason = why

    def _save(self, force: bool = False) -> None:
        if not force and self.clock() - self._last_save < 1.0:
            return
        self._last_save = self.clock()
        data = {"settings": dataclasses.asdict(self.settings), "state": self.state, "reason": self.reason, "go_wall": self.go_wall,
                "tower": self.tower, "stage2_passed": self.stage2_passed, "estop": self.estop, "saved": self.clock()}
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, default=str))
            tmp.replace(self.state_file)
        except Exception as e:  # noqa: BLE001
            self.health["save"] = repr(e)[:200]
        self._publish()

    def _publish(self) -> None:
        """left-status.json where the right worker and the team launcher look: /tmp/relay/ and the newest relay run folder."""
        try:
            relay = pathlib.Path(os.environ.get("RELAY_DIR", "/tmp/relay"))
            if not relay.is_dir():
                return
            g = self.gate_cache or {}
            body = json.dumps({"version": 1, "side": "left", "state": self.state, "reason": self.reason,
                               "tower_blocks": len(self.tower), "active_goal": self.active_goal() if self.go_wall else None,
                               "time_left_s": self.time_left(), "gate_open": self.gate_open(),
                               "gate_bypassed": bool(self.settings.gate_bypass and self.settings.mode == "practice"),
                               "gate_known": g.get("known"), "time": self.clock()}, default=str)
            from hillclimb.common import relay as R
            rr = R.right_run()  # the right worker's own run (holds right-status.json), never just the newest folder
            for d in [relay] + ([rr] if rr is not None else []):
                tmp = d / "left-status.json.tmp"
                tmp.write_text(body)
                tmp.replace(d / "left-status.json")
        except Exception as e:  # noqa: BLE001
            self.health["publish"] = repr(e)[:160]

    def _load(self) -> None:
        try:
            d = json.loads(self.state_file.read_text())
        except Exception:  # noqa: BLE001  no saved round
            return
        try:
            self.settings = Settings(**{k: v for k, v in d.get("settings", {}).items() if hasattr(Settings, k)})
            self.go_wall, self.tower = d.get("go_wall"), d.get("tower", [])
            self.stage2_passed, self.estop = bool(d.get("stage2_passed")), bool(d.get("estop"))
            st = d.get("state", "IDLE")
            if self.go_wall is not None and st not in ("IDLE", "END") and (self.time_left() or 0) > 0:
                # resume: a step that was in flight is unknown, so look again before stacking (RECOVER verifies first)
                self.state = "LOCKED" if st == "LOCKED" else "RECOVER"
                self.reason = f"resumed after restart (was {st})"
            else:
                self.state, self.reason = ("END" if self.go_wall is not None else "IDLE"), "loaded"
            self.backend = self._make_backend() if self.state != "IDLE" else None
            self._prepared = self.state != "LOCKED"
            self.log.write("resume", state=self.state, tower=len(self.tower), time_left=self.time_left())
        except Exception as e:  # noqa: BLE001
            self.health["load"] = repr(e)[:200]


def default_gate() -> dict:
    """The gate's state: the in-process gate module when present, else GATE_URL, else unknown (shut)."""
    try:
        gate = importlib.import_module("hillclimb.gate")
        if hasattr(gate, "http_handler"):
            code, obj = gate.http_handler("GET", "/gate/state", None)
            return obj if code == 200 else {"known": False, "open": False}
        if hasattr(gate, "state"):
            return gate.state()
    except ModuleNotFoundError:
        pass
    if CFG["GATE_URL"]:
        import urllib.request
        with urllib.request.urlopen(CFG["GATE_URL"].rstrip("/") + "/gate/state", timeout=1.5) as r:
            return json.loads(r.read())
    return {"known": False, "open": False}
