"""End-to-end checks of the left controller with a fake gate and fake backends, optionally against a station sim.

    LEFT_URL=http://127.0.0.1:9131 ~/dream-env/bin/python -m hillclimb.sim.test_round [--server-port 9132]

Each check prints PASS or FAIL with what it saw; the script exits non-zero on any FAIL. No pytest needed.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
import urllib.request

from hillclimb.common.arm import ArmClient
from hillclimb.common.config import CFG
from hillclimb.left.controller import LeftController
from hillclimb.sim.fakes import FakeBackend, FakeGate

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)


def wait(pred, timeout: float, step: float = 0.1) -> bool:
    t_end = time.time() + timeout
    while time.time() < t_end:
        if pred():
            return True
        time.sleep(step)
    return pred()


def make(tmp: str, gate, outcomes, sim_url: str | None, **settings):
    arm = ArmClient(CFG["LEFT_URL"])
    fb = {}

    def factory(_s, a):
        fb["b"] = FakeBackend(a, outcomes=outcomes, step_s=0.4, sim_url=sim_url)
        return fb["b"]

    ctl = LeftController(gate=gate, arm=arm, overhead_arm=arm, state_file=os.path.join(tmp, "state.json"), log_dir=tmp, backend_factory=factory)
    base = {"backend": "fake", "vision_confirm": False, "round_s": 40, "max_goal": 4, "crunch_at_s": 10, "crunch_goal": 3,
            "freeze_s": 4, "fault_wait_s": 1.0, "settle_s": 0.5, "rest_check_s": 100}
    base.update(settings)
    ok, why = ctl.arm_round(base)
    assert ok, why
    return ctl, fb


def test_flow(sim_url):
    tmp = tempfile.mkdtemp(prefix="hc_flow_")
    ctl, fb = make(tmp, FakeGate(open_after_s=3), ["ok"], sim_url)
    ctl.start()
    ctl.go()
    check("go enters LOCKED", wait(lambda: ctl.state == "LOCKED", 2), ctl.state)
    sent = ctl.arm.target([0, -1.8, 1.7, -1.4, 0, 0], 0.8)
    check("LOCKED refuses motion", sent is False and "locked" in (ctl.arm.last_error or ""), ctl.arm.last_error or "")
    check("prepare runs during the gate wait", wait(lambda: "prepare" in fb["b"].calls, 2), str(fb["b"].calls))
    check("gate open -> STACK", wait(lambda: ctl.state in ("STACK", "SETTLE_CHECK"), 6), ctl.state)
    check("reaches goal 4 and RESTs", wait(lambda: ctl.state == "REST" and len(ctl.tower) == 4, 20), f"{ctl.state} tower={len(ctl.tower)}")
    check("stage 2 latched at >= 4 in", ctl.stage2_passed, f"{ctl.tower_in():.2f} in")
    check("no motion refused once the gate opened", fb["b"].refused_motion == 0, f"refused={fb['b'].refused_motion}")
    s = ctl.snapshot()
    check("score counts usb + tower", s["score"]["total"] == 10 + 20 + 3 * int(ctl.tower_in()), json.dumps(s["score"]))
    check("ENDs at the clock", wait(lambda: ctl.state == "END", 45), ctl.state)
    ctl.shutdown()


def test_crunch_and_freeze(sim_url):
    tmp = tempfile.mkdtemp(prefix="hc_crunch_")
    ctl, fb = make(tmp, FakeGate(open_after_s=1), ["fail"], sim_url, round_s=16, crunch_at_s=12, crunch_goal=2, freeze_s=6, max_goal=5)
    ctl.start()
    ctl.go()
    check("crunch goal takes over", wait(lambda: ctl.active_goal() == 2, 8), f"goal={ctl.active_goal()}")
    check("FREEZE in the last freeze_s", wait(lambda: ctl.state == "FREEZE", 12), ctl.state)
    n = len(fb["b"].calls)
    time.sleep(2)
    check("no new placement in FREEZE", fb["b"].calls[n:].count("place_next") == 0, str(fb["b"].calls[n:]))
    ctl.shutdown()


def test_fault(sim_url):
    tmp = tempfile.mkdtemp(prefix="hc_fault_")
    ctl, fb = make(tmp, FakeGate(open_after_s=0.5), ["ok", "raise", "ok", "ok", "ok"], sim_url, max_goal=3)
    ctl.start()
    ctl.go()
    check("backend exception -> FAULT", wait(lambda: ctl.state == "FAULT", 10), ctl.state)
    check("FAULT -> RECOVER -> stacking resumes", wait(lambda: "recover" in fb["b"].calls and len(ctl.tower) >= 2, 15),
          f"tower={len(ctl.tower)} calls={fb['b'].calls[-4:]}")
    check("controller loop still alive", time.time() - ctl.health.get("heartbeat", 0) < 2, f"heartbeat age {time.time() - ctl.health.get('heartbeat', 0):.1f}s")
    ctl.shutdown()


def test_restart(sim_url):
    tmp = tempfile.mkdtemp(prefix="hc_restart_")
    gate = FakeGate(open_after_s=0.5)
    ctl, fb = make(tmp, gate, ["ok"], sim_url, max_goal=9, round_s=60)
    ctl.start()
    ctl.go()
    wait(lambda: len(ctl.tower) >= 2, 10)
    ctl.shutdown()
    time.sleep(0.5)
    ctl._save(force=True)
    n, go_wall = len(ctl.tower), ctl.go_wall
    ctl2 = LeftController(gate=gate, arm=ctl.arm, overhead_arm=ctl.arm, state_file=os.path.join(tmp, "state.json"), log_dir=tmp,
                          backend_factory=lambda s, a: FakeBackend(a, ["ok"], 0.4, sim_url))
    check("restart keeps the round clock", ctl2.go_wall == go_wall and (ctl2.time_left() or 0) > 0, f"time_left={ctl2.time_left():.1f}")
    check("restart keeps the tower count", len(ctl2.tower) == n, f"tower={len(ctl2.tower)}")
    check("restart re-checks before stacking (RECOVER)", ctl2.state == "RECOVER", ctl2.state)
    ctl2.start()
    check("resumed round keeps stacking", wait(lambda: len(ctl2.tower) > n, 10), f"tower={len(ctl2.tower)}")
    ctl2.shutdown()


def test_external(sim_url):
    tmp = tempfile.mkdtemp(prefix="hc_ext_")
    gate = FakeGate(open_after_s=2)
    ctl, fb = make(tmp, gate, ["ok"], sim_url, drive="external", max_goal=9)
    ctl.start()
    ctl.go()
    r = ctl.stack_next()
    check("stack_next refused while the gate is shut", r["ok"] is False and r["why"] == "gate shut", json.dumps(r))
    wait(ctl.gate_open, 4)
    r = ctl.stack_next()
    check("stack_next places one block once open", r.get("grew") is True and r["tower_blocks"] == 1, json.dumps({k: r[k] for k in ("ok", "grew", "tower_blocks", "why")}))
    v = ctl.verify_tower()
    check("verify_tower reports the count", v["known"] and v["block_count"] == 1, json.dumps(v)[:160])
    ctl.shutdown()


def test_server(sim_url, port):
    from hillclimb.left import server as S
    tmp = tempfile.mkdtemp(prefix="hc_srv_")
    ctl, fb = make(tmp, FakeGate(open_after_s=1), ["ok"], sim_url, mode="match")
    srv, _ = S.build(port, ctl)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ctl.start()
    base = f"http://127.0.0.1:{port}"
    get = lambda p: json.loads(urllib.request.urlopen(base + p, timeout=5).read())  # noqa: E731
    page = urllib.request.urlopen(base + "/", timeout=5).read()
    check("status page serves", b"Hill Climb" in page, f"{len(page)} bytes")
    req = urllib.request.Request(base + "/left/go", b"{}", {"Content-Type": "application/json"})
    check("POST /left/go", json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"] is True)
    st = get("/api/state")
    check("GET /api/state", "left" in st and "gate" in st, st["left"]["state"])
    if sim_url:
        img = urllib.request.urlopen(base + "/cam/overhead.jpg", timeout=5).read()
        check("camera proxy returns a JPEG", img[:2] == b"\xff\xd8", f"{len(img)} bytes")
    req = urllib.request.Request(base + "/left/arm", json.dumps({"max_goal": 3, "mode": "match"}).encode(), {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        locked = False
    except urllib.error.HTTPError as e:
        locked = e.code == 409
    check("match mode refuses settings mid-round (409)", locked and ctl.settings.max_goal == 4, f"max_goal={ctl.settings.max_goal}")
    gate = urllib.request.urlopen(base + "/gate/state", timeout=5) if S.gate_module() else None
    check("gate routes mounted or reported missing", gate is not None or bool(S._gate_err), S._gate_err or "mounted")
    ctl.shutdown()
    srv.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-port", type=int, default=9132)
    a = ap.parse_args()
    if any(f":{p}" in CFG["LEFT_URL"] or f":{p}" in CFG["RIGHT_URL"] for p in (8691, 8692, 8081, 8084)):
        raise SystemExit("refusing: LEFT_URL/RIGHT_URL point at a real arm. Set them to a sim, e.g. LEFT_URL=http://127.0.0.1:9131")
    sim_url = CFG["LEFT_URL"] if "127.0.0.1:91" in CFG["LEFT_URL"] else None
    for t in (test_flow, test_crunch_and_freeze, test_fault, test_restart, test_external):
        try:
            t(sim_url)
        except Exception as e:  # noqa: BLE001
            check(t.__name__ + " ran", False, repr(e)[:200])
    try:
        test_server(sim_url, a.server_port)
    except Exception as e:  # noqa: BLE001
        check("test_server ran", False, repr(e)[:200])
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)} of {len(RESULTS)} passed", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
