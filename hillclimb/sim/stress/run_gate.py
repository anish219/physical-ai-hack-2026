"""USB gate fault injection: camera faults while armed, VLM faults and flicker, latch leaks across rounds.

    uv run --no-project --python 3.12 --with numpy --with opencv-python-headless python -m hillclimb.sim.stress.run_gate

The VLM is a fake (vlm._post replaced: no OpenAI call, no key). The frame source and jaw watch are the real classes,
polling the fake arm servers. The worst-case VLM answers YES to everything, so any "open" under a camera fault is a
hole in the health checks, not in the prompt.
"""
from __future__ import annotations

import collections
import json
import os
import time

from hillclimb.sim.stress import jail  # noqa: F401  (must be first)
from hillclimb.sim.stress.jail import LEFT, RIGHT, record

from hillclimb.gate import service, settings, vlm  # noqa: E402

for k, v in {"FAST_PERIOD_S": 0.3, "AGENTIC_MIN_GAP_S": 0.6, "AGENTIC_BUDGET_S": 3.0, "FAST_TIMEOUT_S": 2.0,
             "GATE_FRAME_POLL_S": 0.1}.items():
    settings.set_override(k, v)


class FakeVLM:
    def __init__(self):
        self.fast, self.agentic, self.mode, self.i = "yes", ["yes"], None, 0
        self.calls = collections.Counter()

    def post(self, body, timeout_s):
        if timeout_s <= 0.05:
            raise vlm.VlmError("no time left")
        agentic = "tools" in body
        self.calls["agentic" if agentic else "fast"] += 1
        if self.mode == "error":
            raise vlm.VlmError("HTTP 500: boom")
        if self.mode == "timeout":
            time.sleep(min(timeout_s, 0.8))
            raise vlm.VlmError("TimeoutError: timed out")
        if not agentic:
            if self.mode == "malformed":
                content = "Sure, a cable is out {not json"
            elif self.mode == "string_true":
                content = json.dumps({"new_cable_unplugged_and_resting_on_table": "true", "plugged_now": 3})
            elif self.mode == "two_objects":
                content = '{"new_cable_unplugged_and_resting_on_table": true} {"x": 1}'
            else:
                content = json.dumps({"new_cable_unplugged_and_resting_on_table": self.fast == "yes", "plugged_ref": 4, "plugged_now": 3})
            return {"choices": [{"message": {"content": content}}], "usage": {}}
        v = self.agentic[self.i % len(self.agentic)]
        self.i += 1
        if self.mode == "malformed":
            args = "{unplugged_and_on_table: true"
        elif self.mode == "string_true":
            args = json.dumps({"unplugged_and_on_table": "true", "confidence": 0.9, "evidence": "x"})
        elif self.mode == "no_verdict":
            return {"choices": [{"message": {"role": "assistant", "content": "I think yes"}}], "usage": {}}
        else:
            args = json.dumps({"unplugged_and_on_table": v == "yes", "confidence": 0.9, "evidence": "fake"})
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{self.i}", "type": "function", "function": {"name": "submit_verdict", "arguments": args}}]}}], "usage": {}}


FAKE = FakeVLM()
vlm._post = FAKE.post


def cams(mode, left=None, right=None):
    LEFT.mode["overhead"] = left or mode
    RIGHT.mode["overhead"] = right or mode


def wait(pred, timeout, step=0.05):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if pred():
            return True
        time.sleep(step)
    return pred()


N = [0]


def mk_gate(log_dir=None):
    N[0] += 1
    d = log_dir or os.path.join(jail.TMP, f"gate_{N[0]}")
    os.makedirs(d, exist_ok=True)
    settings.set_override("LOG_DIR", d)
    g = service.Gate()
    wait(lambda: g.source.latest() is not None and g.source.fresh() and g.jaw.fresh(), 5)
    return g, d


def close(g):
    g.stop()
    g.source.close()
    g.jaw.close()


def armed_gate():
    cams("ok")
    FAKE.mode, FAKE.fast, FAKE.agentic = None, "no", ["no"]
    g, d = mk_gate()
    r = g.arm({})
    assert r.get("ok"), r
    return g, d


# ------------------------------------------------------------------ 3. camera faults while armed (VLM says yes to all)
def t_camera_faults():
    for name, left, right, extra in [("frozen, stamp kept", "frozen", "frozen", {}),
                                     ("missing (404) on both servers", "missing", "missing", {}),
                                     ("garbage bytes after a JPEG header", "garbage", "garbage", {}),
                                     ("slow (beyond the 2 s timeout)", "slow", "slow", {"slow_s": 4.0}),
                                     ("frozen, no X-Stamp header", "frozen_nostamp", "frozen_nostamp", {}),
                                     ("left flaky, right frozen 60 s ago", "flaky", "frozen", {"frozen_age_s": 60.0}),
                                     ("left frozen, right live (fallback)", "frozen", "ok", {})]:
        g, _ = armed_gate()
        LEFT.mode.update(extra)
        RIGHT.mode.update(extra)
        cams(None, left, right)
        time.sleep(2.5)  # past GATE_FRAME_MAX_AGE_S
        FAKE.fast, FAKE.agentic = "yes", ["yes"]
        opened = wait(lambda: g.core.open, 8)
        st = g.state()
        expect_open = name.startswith("left frozen, right live")
        verdict = "OK" if opened == expect_open else ("BUG" if opened else "NOTE")
        record(verdict, f"3 overhead {name}: gate {'OPENED' if opened else 'stayed shut'}",
               f"fresh={g.source.fresh()} known={st['known']} vlm calls={dict(FAKE.calls)} source={g.source.url[-30:]}")
        FAKE.calls.clear()
        for k in ("slow_s", "frozen_age_s"):
            LEFT.mode.pop(k, None)
            RIGHT.mode.pop(k, None)
        LEFT.mode["slow_s"] = RIGHT.mode["slow_s"] = 3.0
        close(g)
    cams("ok")


def t_frozen_after_first_yes():
    g, _ = armed_gate()
    FAKE.fast, FAKE.agentic = "yes", ["yes"]
    wait(lambda: g.core.streak >= 1, 5)
    s1 = g.core.streak
    cams("frozen")
    time.sleep(5.0)  # max frame age 2 s plus one agentic wait
    record("OK" if not g.core.open and g.core.streak == 0 else "BUG", "3h camera freezes after one agentic yes: streak resets, gate shut",
           f"streak {s1} -> {g.core.streak}, open={g.core.open}")
    cams("ok")
    close(g)


# ------------------------------------------------------------------ 4. VLM faults
def t_vlm_faults():
    cams("ok")
    for mode in ("error", "timeout", "malformed", "string_true", "two_objects", "no_verdict"):
        g, _ = armed_gate()
        FAKE.mode, FAKE.fast, FAKE.agentic = mode, "yes", ["yes"]
        opened = wait(lambda: g.core.open, 6)
        record("BUG" if opened else "OK", f"4 VLM {mode}: gate {'OPENED' if opened else 'stayed shut'}",
               f"streak={g.core.streak} calls={dict(FAKE.calls)} last={str((g.last_reads or [{}])[-1].get('error'))[:60]}")
        FAKE.calls.clear()
        FAKE.mode = None
        close(g)
    # flicker: agentic yes, no, yes, no ...
    g, _ = armed_gate()
    FAKE.fast, FAKE.agentic, FAKE.i = "yes", ["yes", "no"], 0
    opened = wait(lambda: g.core.open, 8)
    record("BUG" if opened else "OK", "4 VLM agentic yes/no flicker", f"open={opened} agentic calls={FAKE.calls['agentic']} streak={g.core.streak}")
    close(g)
    # one yes then no forever
    g, _ = armed_gate()
    FAKE.fast, FAKE.agentic, FAKE.i = "yes", ["yes"] + ["no"] * 1000, 0
    opened = wait(lambda: g.core.open, 6)
    record("BUG" if opened else "OK", "4 VLM one agentic yes then no", f"open={opened} streak={g.core.streak}")
    close(g)
    # fast yes / agentic yes on alternate frames, fast no in between: fast no must reset
    g, _ = armed_gate()
    FAKE.fast, FAKE.agentic, FAKE.i = "yes", ["yes", "yes"], 0
    t0 = time.time()
    opened = wait(lambda: g.core.open, 8)
    hist = [h for h in g.core.history if h.get("kind") in ("agentic", "open")]
    gaps = [round(b["now_t"] - a["now_t"], 2) for a, b in zip(hist, hist[1:]) if "now_t" in a and "now_t" in b]
    record("OK" if opened else "BUG", "4 sanity: two agentic yeses on frames >= min gap apart open it",
           f"opened after {time.time() - t0:.1f} s, now_t gaps {gaps}")
    close(g)


# ------------------------------------------------------------------ 8. latch leaks into the next round
def t_latch_leak():
    g, d = armed_gate()
    FAKE.fast, FAKE.agentic = "yes", ["yes"]
    wait(lambda: g.core.open, 8)
    record("NOTE", "8 round 1 gate opened", f"open={g.core.open}")
    FAKE.fast, FAKE.agentic = "no", ["no"]
    cams("missing")
    time.sleep(2.5)
    r = g.arm({})
    st = g.state()
    record("BUG" if st["open"] else "OK", "8a new round's /gate/arm fails (no fresh frame): last round's latch stays OPEN",
           f"arm={r} state.open={st['open']} known={st['known']} armed={st['armed']}")
    # restart of the process: a new Gate restores the open latch from disk regardless of age
    close(g)
    cams("ok")
    g2, _ = mk_gate(d)
    record("NOTE" if g2.core.open else "OK", "8b process restart restores the open latch from gate_state.json (no age limit)",
           f"open={g2.core.open} latched_at age={time.time() - (g2.core.latched_at or time.time()):.0f} s")
    os.environ["GATE_MAX_ARMED_S"] = "1"
    time.sleep(2.0)
    record("NOTE" if g2.core.open else "OK", "8c GATE_MAX_ARMED_S auto-disarm never touches an OPEN gate", f"open={g2.core.open} armed={g2.core.armed}")
    os.environ.pop("GATE_MAX_ARMED_S", None)
    r = g2.arm({})
    record("OK" if r.get("ok") and not g2.core.open else "BUG", "8d a successful /gate/arm resets the latch", f"arm ok={r.get('ok')} open={g2.core.open}")
    close(g2)


def main():
    only = os.environ.get("ONLY")
    for t in (t_camera_faults, t_frozen_after_first_yes, t_vlm_faults, t_latch_leak):
        if only and only not in t.__name__:
            continue
        try:
            t()
        except Exception:  # noqa: BLE001
            import traceback
            record("BUG", t.__name__ + " crashed", traceback.format_exc()[-500:])
    jail.summary()


if __name__ == "__main__":
    main()
