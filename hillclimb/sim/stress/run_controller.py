"""Controller fault injection: arm off, arm server faults, raising backends, time edges, restarts, races.

    uv run --no-project --python 3.12 --with numpy --with opencv-python-headless python -m hillclimb.sim.stress.run_controller

Fakes only (see jail.py). Prints OK / BUG / NOTE per check.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

from hillclimb.sim.stress import jail  # noqa: F401  (must be first)
from hillclimb.sim.stress.jail import LEFT, record

from hillclimb.common.arm import ArmClient  # noqa: E402
from hillclimb.left.controller import LeftController  # noqa: E402
from hillclimb.sim.fakes import FakeBackend  # noqa: E402
from hillclimb.sim.stress.backends import ScriptLike, SwitchGate  # noqa: E402

BASE = {"backend": "fake", "vision_confirm": False, "round_s": 30, "max_goal": 4, "crunch_at_s": 5, "crunch_goal": 3,
        "freeze_s": 4, "fault_wait_s": 3.0, "settle_s": 0.3, "rest_check_s": 100}


def wait(pred, timeout, step=0.05):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if pred():
            return True
        time.sleep(step)
    return pred()


class Trace:
    """Samples (t, state, reason, tower) every 50 ms."""

    def __init__(self, ctl):
        self.ctl, self.rows, self.stop = ctl, [], False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            c = self.ctl
            row = (round(time.time(), 2), c.state, c.reason, len(c.tower))
            if not self.rows or self.rows[-1][1:] != row[1:]:
                self.rows.append(row)
            time.sleep(0.05)

    def states(self):
        return [r[1] for r in self.rows]

    def text(self, n=12):
        t0 = self.rows[0][0] if self.rows else 0
        return "; ".join(f"{r[0] - t0:.1f}s {r[1]}({r[2][:40]}) t{r[3]}" for r in self.rows[:n])


def mk(backend="script", gate=None, state_dir=None, **settings):
    tmp = state_dir or tempfile.mkdtemp(prefix="ctl_", dir=jail.TMP)
    os.environ["GATE_FILE"] = os.path.join(tmp, "usb_gate_open")
    arm = ArmClient(LEFT.url)
    holder = {}

    def factory(_s, a):
        b = ScriptLike(LEFT.url, **holder.get("kw", {})) if backend == "script" else FakeBackend(a, outcomes=holder.get("outcomes", ["ok"]), step_s=0.4)
        holder.setdefault("all", []).append(b)
        holder["b"] = b
        return b

    gate = gate or SwitchGate(False)
    ctl = LeftController(gate=gate, arm=arm, overhead_arm=arm, state_file=os.path.join(tmp, "state.json"), log_dir=tmp,
                         backend_factory=factory)
    cfg = dict(BASE)
    cfg.update(settings)
    holder["kw"] = cfg.pop("_kw", {})
    holder["outcomes"] = cfg.pop("_outcomes", ["ok"])
    ok, why = ctl.arm_round(cfg)
    assert ok, why
    return ctl, holder, gate, tmp


def reset_arm():
    LEFT.mode.update({"state": "ok", "move": "ok", "overhead": "ok", "grip": 0.8, "slow_s": 3.0})


# ------------------------------------------------------------------ 1. arm off at go and mid-round
def t_arm_off_at_go():
    reset_arm()
    LEFT.mode["state"] = "q_null"
    ctl, h, gate, _ = mk()
    tr = Trace(ctl)
    ctl.start()
    ctl.go()
    time.sleep(2.0)
    snap = ctl.snapshot()
    shows = "ARM IS OFF" in (snap["reason"] or "") or any("ARM IS OFF" in str(v) for v in snap["health"].values())
    record("OK" if shows else "BUG", "1a arm off at go: dashboard says the arm is off while LOCKED",
           f"state={snap['state']} reason={snap['reason']!r} last_step.why={str(snap['last_step'].get('why'))[:50]!r} health={list(snap['health'])}")
    gate.open = True
    t0 = time.time()
    time.sleep(10)
    b = h["b"]
    n_calls = len([c for c in b.calls if c[0] >= t0])
    record("NOTE", "1b arm off at go, gate opens: backend calls in 10 s",
           f"calls={n_calls} moves={b.moves_sent} /state GETs={LEFT.count('GET /state', t0)} states={sorted(set(tr.states()))}")
    LEFT.mode["state"] = "ok"
    grew = wait(lambda: len(ctl.tower) >= 1, 8)
    record("OK" if grew else "BUG", "1c arm powered on again: stacking resumes without a new Start", f"tower={len(ctl.tower)} state={ctl.state}")
    tr.stop = True
    ctl.shutdown()


def t_arm_off_mid_round():
    reset_arm()
    ctl, h, gate, _ = mk(max_goal=9, round_s=40)
    gate.open = True
    tr = Trace(ctl)
    ctl.start()
    ctl.go()
    wait(lambda: len(ctl.tower) >= 2, 10)
    n_before = len(ctl.tower)
    LEFT.mode["state"] = "disabled"
    t0 = time.time()
    time.sleep(10)
    b = h["b"]
    seq = [r[1] for r in tr.rows if r[0] >= t0]
    flips = sum(1 for a, c in zip(seq, seq[1:]) if a != c)
    record("NOTE", "1d arm off mid-round: state cycle in 10 s",
           f"{flips} state changes, calls={len([c for c in b.calls if c[0] >= t0])}, moves={LEFT.motion_calls(t0)}, "
           f"recover calls={b.count('recover', t0)}, states={sorted(set(seq))}")
    LEFT.mode["state"] = "ok"
    t1 = time.time()
    wait(lambda: b.count("recover", t1) >= 1 or len(b.tower) > n_before, 8)
    time.sleep(1.0)
    torn = b.count("recover", t1) >= 1
    record("BUG" if torn else "OK", "1e arm back on after a FAULT: RECOVER runs recover(), which (scripted) separates every raised block",
           f"recover calls after power-on={b.count('recover', t1)}, backend tower={len(b.tower)}, controller tower={len(ctl.tower)} (was {n_before})")
    tr.stop = True
    ctl.shutdown()
    reset_arm()


# ------------------------------------------------------------------ 2. arm server unreachable, slow, HTTP 400
def t_arm_server_faults():
    reset_arm()
    ctl, h, gate, _ = mk(max_goal=9)
    gate.open = True
    tr = Trace(ctl)
    ctl.start()
    ctl.go()
    wait(lambda: len(ctl.tower) >= 1, 8)
    LEFT.mode["move"] = "http400"
    t0 = time.time()
    time.sleep(8)
    b = h["b"]
    record("NOTE", "2a /move returns HTTP 400 every time (8 s)",
           f"place calls={b.count('place_next', t0)}, recover calls={b.count('recover', t0)}, /move posts={LEFT.motion_calls(t0)}, states={sorted(set(r[1] for r in tr.rows if r[0] >= t0))}")
    reset_arm()
    LEFT.mode.update(state="slow", slow_s=4.0)
    t0 = time.time()
    time.sleep(8)
    snap = ctl.snapshot()
    record("NOTE", "2b /state slower than the timeout (8 s)", f"state={snap['state']} reason={snap['reason'][:80]!r} busy={snap['busy']}")
    reset_arm()
    # unreachable: point the backend at a dead port
    b = h["b"]
    b.url = "http://127.0.0.1:1"
    t0 = time.time()
    time.sleep(8)
    record("NOTE", "2c arm server unreachable (8 s)",
           f"place={b.count('place_next', t0)} recover={b.count('recover', t0)} state={ctl.state} reason={ctl.reason[:70]!r}")
    b.url = LEFT.url
    tr.stop = True
    ctl.shutdown()
    # the repo's FakeBackend and the guarded ArmClient: a 400 is reported as "refused: motion locked"?
    reset_arm()
    LEFT.mode["move"] = "http400"
    a = ArmClient(LEFT.url)
    ok = a.target([0] * 6, 0.8)
    record("NOTE", "2d ArmClient.target on HTTP 400", f"returned {ok}, last_error={a.last_error!r} (FakeBackend counts this as refused_motion)")
    reset_arm()


# ------------------------------------------------------------------ 9. backend raising every call
def t_backend_raises_always():
    reset_arm()
    ctl, h, gate, _ = mk(_kw={"raise_on": ("all",)}, round_s=60)
    gate.open = True
    tr = Trace(ctl)
    ctl.start()
    ctl.go()
    time.sleep(1.0)
    t0 = time.time()
    time.sleep(15)
    b = h["b"]
    n = len([c for c in b.calls if c[0] >= t0])
    rate = n / 15 * 60
    record("BUG" if rate > 10 else "OK", "9a backend raises on every call: no cap or backoff on FAULT -> RECOVER",
           f"{n} backend calls in 15 s (~{rate:.0f}/min forever), recover calls={b.count('recover', t0)}, states={sorted(set(r[1] for r in tr.rows if r[0] >= t0))}")
    tr.stop = True
    ctl.shutdown()


def t_backend_fails_quietly():
    """ok False without raising (the scripted 'nothing left that plans'): faults counter 3 -> FAULT, then RECOVER."""
    reset_arm()
    ctl, h, gate, _ = mk(backend="fake", _outcomes=["fail"], round_s=60, max_goal=9)
    gate.open = True
    ctl.start()
    ctl.go()
    time.sleep(1.0)
    t0 = time.time()
    time.sleep(10)
    b = h["b"]
    n = b.calls.count("place_next")
    record("NOTE", "9b backend says ok but no growth every time (10 s)", f"place_next calls={n} (never FAULTs: 'no growth' resets faults to 0), state={ctl.state}")
    ctl.shutdown()


# ------------------------------------------------------------------ 7. time edges
def t_time_runs_out_mid_place():
    reset_arm()
    ctl, h, gate, _ = mk(round_s=6, freeze_s=1, max_goal=9, _kw={"step_s": 7.0, "moves": 7})
    gate.open = True
    tr = Trace(ctl)
    ctl.start()
    ctl.go()
    time.sleep(7.5)
    snap = ctl.snapshot()
    record("BUG" if snap["state"] not in ("END", "FREEZE") else "OK", "7a clock hits 0 while a placement is in flight: state shown",
           f"time_left={snap['time_left_s']} state={snap['state']} busy={snap['busy']} moves after 0 s: {LEFT.motion_calls(ctl.go_wall + 6)}")
    wait(lambda: ctl.state == "END", 10)
    record("NOTE", "7b when the late placement lands", f"{tr.text(8)}; tower={len(ctl.tower)}")
    tr.stop = True
    ctl.shutdown()


def t_freeze_from_recover():
    reset_arm()
    ctl, h, gate, _ = mk(round_s=12, freeze_s=6, max_goal=9, _kw={"raise_on": ("place_next",)}, fault_wait_s=0.5)
    gate.open = True
    tr = Trace(ctl)
    ctl.start()
    ctl.go()
    wait(lambda: ctl.state == "FREEZE", 12)
    time.sleep(1.0)
    b = h["b"]
    retreats = b.count("retreat")
    record("NOTE", "7c FREEZE reached from RECOVER: retreat issued?", f"retreat calls={retreats}; {tr.text(10)}")
    tr.stop = True
    ctl.shutdown()


def t_none_paths():
    ctl, h, gate, _ = mk()
    errs = []
    for name, fn in [("snapshot", ctl.snapshot), ("score", ctl.score), ("active_goal", ctl.active_goal), ("_scene", ctl._scene),
                     ("call_stop_now", ctl.call_stop_now), ("_publish", ctl._publish), ("tick", ctl.tick),
                     ("stack_next(auto)", ctl.stack_next), ("verify_tower", ctl.verify_tower)]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            errs.append(f"{name}: {e!r}")
    record("OK" if not errs else "BUG", "7d time_left None paths (IDLE, no go)", "; ".join(errs) or "no exception")
    ok, why = ctl.arm_round({"gate_bypass": "false"})
    record("BUG" if ok and ctl.settings.gate_bypass else "OK", "7e arm_round coerces 'false' to True (bool('false'))",
           f"ok={ok} why={why} gate_bypass={ctl.settings.gate_bypass}")
    try:
        r = ctl.arm_round({"round_s": None})
        record("NOTE", "7f arm_round with a null value", str(r))
    except Exception as e:  # noqa: BLE001
        record("NOTE", "7f arm_round with a null value raises out (server answers 500)", repr(e)[:100])


# ------------------------------------------------------------------ 6. bypass
def t_bypass():
    ctl, h, gate, _ = mk(mode="match")
    ok, why = ctl.arm_round({"gate_bypass": True})
    record("OK" if not ok else "BUG", "6a bypass while mode=match is refused", f"ok={ok} why={why}")
    ok, why = ctl.arm_round({"gate_bypass": True, "mode": "practice"})
    record("BUG" if ok else "OK", "6b status page Start with bypass ticked sends mode=practice: a match quietly becomes practice+bypass",
           f"ok={ok} mode={ctl.settings.mode} gate_open with gate shut={ctl.gate_open()}")
    ctl.arm_round({"gate_bypass": False, "mode": "match"})
    ctl.start()
    ctl.go()
    ok, why = ctl.arm_round({"gate_bypass": True, "mode": "practice"})
    record("OK" if not ok else "BUG", "6c bypass mid match round is refused", f"ok={ok} why={why}")
    ctl.shutdown()
    # practice mid-round arm_round is allowed while a job runs
    reset_arm()
    ctl, h, gate, _ = mk(max_goal=9, _kw={"step_s": 3.0, "moves": 6})
    gate.open = True
    ctl.start()
    ctl.go()
    wait(lambda: ctl._busy() and h["b"].count("place_next") >= 1, 6)
    old = h["b"]
    ok, why = ctl.arm_round({"gate_bypass": False})
    gate.open = False
    ctl.go()
    t0 = time.time()
    time.sleep(3.5)
    record("BUG" if LEFT.motion_calls(t0) > 0 or len(ctl.tower) > 0 else "OK",
           "6d practice arm_round mid-placement: old job keeps moving and its result lands in the new round",
           f"arm_round ok={ok}; /move after re-arm (gate shut, LOCKED)={LEFT.motion_calls(t0)}; new-round tower={len(ctl.tower)}; old backend tower={len(old.tower)}")
    ctl.shutdown()


# ------------------------------------------------------------------ 5. resume after restart with the gate shut
def t_resume_gate_shut():
    reset_arm()
    gate = SwitchGate(True)
    ctl, h, _, tmp = mk(gate=gate, max_goal=9, round_s=60)
    ctl.start()
    ctl.go()
    wait(lambda: len(ctl.tower) >= 2, 10)
    ctl.shutdown()
    time.sleep(0.6)
    ctl._save(force=True)
    gate2 = SwitchGate(False)  # e.g. gate_state.json lost, or the gate restored without a reference
    t0 = time.time()
    holder = {}

    def factory(_s, a):
        holder["b"] = ScriptLike(LEFT.url)
        return holder["b"]

    ctl2 = LeftController(gate=gate2, arm=ArmClient(LEFT.url), overhead_arm=ArmClient(LEFT.url), state_file=os.path.join(tmp, "state.json"),
                          log_dir=tmp, backend_factory=factory)
    ctl2.start()
    time.sleep(3)
    b = holder["b"]
    record("BUG" if b.count("recover") or LEFT.motion_calls(t0) else "OK", "5a server restart mid-round with the gate now shut: RECOVER moves anyway",
           f"resumed state={ctl2.state}, recover calls={b.count('recover')}, /move={LEFT.motion_calls(t0)}, gate_open={ctl2.gate_open()}")
    ctl2.shutdown()


# ------------------------------------------------------------------ stage 2 with the scripted contract
def t_stage2_scripted():
    reset_arm()
    ctl, h, gate, _ = mk(max_goal=4)
    gate.open = True
    ctl.start()
    ctl.go()
    wait(lambda: len(ctl.tower) >= 4 and ctl.state == "REST", 15)
    s = ctl.snapshot()
    record("BUG" if not s["stage2_passed"] else "OK", "S1 scripted results (verified + tower list) never latch stage 2",
           f"tower={s['tower_blocks']} ({s['tower_in']} in) stage2={s['stage2_passed']} score.tower={s['score']['tower']} state={s['state']} call_stop={s['call_stop_why']}")
    ctl.shutdown()


# ------------------------------------------------------------------ 10. lock held across network calls: E-stop latency
def t_estop_latency():
    reset_arm()
    ctl, h, gate, _ = mk(max_goal=9, _kw={"step_s": 0.2})
    gate.open = True
    ctl.start()
    ctl.go()
    wait(lambda: len(ctl.tower) >= 1, 6)
    LEFT.mode.update(overhead="slow", slow_s=5.0)  # _frames() under the lock in _start_job: 1.5 s timeout per frame
    worst = 0.0
    for _ in range(3):
        wait(lambda: not ctl._busy(), 5)
        time.sleep(0.05)
        t0 = time.time()
        got = threading.Event()

        def grab():
            with ctl.lock:
                got.set()
        threading.Thread(target=grab, daemon=True).start()
        got.wait(10)
        worst = max(worst, time.time() - t0)
    record("BUG" if worst > 1.0 else "OK", "10a controller lock held across camera HTTP calls (blocks E-stop, Stop, /api/state)",
           f"worst wait for the lock {worst:.1f} s with a slow overhead camera")
    t0 = time.time()
    n0 = LEFT.count("POST /stop")
    th = threading.Thread(target=ctl.emergency_stop, daemon=True)
    th.start()
    wait(lambda: LEFT.count("POST /stop") > n0, 10, 0.01)
    record("NOTE", "10b E-stop to the arm's /stop", f"{time.time() - t0:.2f} s; the job thread still alive after E-stop: {ctl._busy()}")
    ctl.shutdown()
    reset_arm()


def t_gate_read_raises():
    ctl, h, gate, _ = mk()
    gate.open = True
    ctl._refresh_gate()
    gate.raise_ = True
    ctl._refresh_gate()
    record("OK" if not ctl.gate_open() else "BUG", "10c gate read raises after an open read: controller treats it as shut",
           f"gate_cache={ctl.gate_cache} gate_open={ctl.gate_open()}")
    gate.raise_ = False
    gate.open, gate.known = True, False
    ctl._refresh_gate()
    record("OK" if not ctl.gate_open() else "BUG", "10d open but known=False", f"gate_open={ctl.gate_open()}")
    gate.known = True
    # gate drops back to shut mid-STACK: does the controller stop?
    reset_arm()
    ctl, h, gate, _ = mk(max_goal=9)
    gate.open = True
    ctl.start()
    ctl.go()
    wait(lambda: len(ctl.tower) >= 1, 6)
    gate.open = False
    time.sleep(0.5)
    t0 = time.time()
    time.sleep(3)
    record("NOTE", "10e gate reads shut again mid-STACK", f"place calls in 3 s={h['b'].count('place_next', t0)} state={ctl.state} (the gate latches, so this only happens on a gate error)")
    ctl.shutdown()


def t_go_after_end():
    reset_arm()
    ctl, h, gate, tmp = mk(max_goal=9)
    gate.open = True
    ctl.start()
    ctl.go()
    wait(lambda: len(ctl.tower) >= 2, 8)
    ctl.stop_round()
    gate.open = False
    gf = os.path.exists(os.environ["GATE_FILE"])
    ok, why = ctl.go()  # POST /left/go without /left/arm (or stack_next's own go())
    time.sleep(1.0)
    record("BUG" if ok and (len(ctl.tower) or gf) else "OK", "5b Go after END without arm: old tower, stage 2, _prepared and GATE_FILE carry over",
           f"go ok={ok}; tower={len(ctl.tower)} stage2={ctl.stage2_passed} prepared={ctl._prepared} GATE_FILE still present={gf}")
    ctl.shutdown()


def main():
    tests = [t_none_paths, t_bypass, t_arm_off_at_go, t_arm_off_mid_round, t_arm_server_faults, t_backend_raises_always,
             t_backend_fails_quietly, t_time_runs_out_mid_place, t_freeze_from_recover, t_resume_gate_shut, t_stage2_scripted,
             t_estop_latency, t_gate_read_raises, t_go_after_end]
    only = os.environ.get("ONLY")
    for t in tests:
        if only and only not in t.__name__:
            continue
        try:
            t()
        except Exception as e:  # noqa: BLE001
            import traceback
            record("BUG", t.__name__ + " crashed", traceback.format_exc()[-400:])
    jail.summary()


if __name__ == "__main__":
    main()
