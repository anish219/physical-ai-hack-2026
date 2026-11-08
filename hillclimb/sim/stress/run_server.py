"""HTTP route fault injection with the real gate module in-process (fake VLM, fake cameras, fake arm).

    uv run --no-project --python 3.12 --with numpy --with opencv-python-headless python -m hillclimb.sim.stress.run_server

Go twice, Start with a dead overhead after an opened round (latch leak), keep_gate in match, Restart mid-job, Stop
during prepare, E-stop then Start.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from hillclimb.sim.stress import jail  # noqa: F401  (must be first)
from hillclimb.sim.stress.jail import LEFT, record
from hillclimb.sim.stress import run_gate as G  # noqa: E402  (fake VLM patched in, gate settings sped up)

from hillclimb.common.arm import ArmClient  # noqa: E402
from hillclimb.gate import settings  # noqa: E402
from hillclimb.left import server as S  # noqa: E402
from hillclimb.left.controller import LeftController  # noqa: E402
from hillclimb.sim.stress.backends import ScriptLike  # noqa: E402

settings.set_override("LOG_DIR", os.path.join(jail.TMP, "srv_gate"))
os.makedirs(os.path.join(jail.TMP, "srv_gate"), exist_ok=True)
os.environ["GATE_FILE"] = os.path.join(jail.TMP, "srv_usb_gate_open")
HOLD = {}


def factory(_s, _a):
    HOLD["b"] = ScriptLike(LEFT.url, **HOLD.get("kw", {}))
    HOLD.setdefault("all", []).append(HOLD["b"])
    return HOLD["b"]


ctl = LeftController(arm=ArmClient(LEFT.url), overhead_arm=ArmClient(LEFT.url), state_file=os.path.join(jail.TMP, "srv_state.json"),
                     log_dir=os.path.join(jail.TMP, "srv_logs"), backend_factory=factory)  # default_gate: the in-process module
srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(ctl, {"overhead": (ctl.arm, "overhead")}))
srv.daemon_threads = True
jail.allow(srv.server_address[1])
BASE_URL = f"http://127.0.0.1:{srv.server_address[1]}"
threading.Thread(target=srv.serve_forever, daemon=True).start()
S.gate_module().start()
ctl.start()

MATCH = {"backend": "fake", "mode": "match", "vision_confirm": False, "round_s": 60, "max_goal": 9, "freeze_s": 4,
         "fault_wait_s": 3.0, "settle_s": 0.3, "rest_check_s": 100, "gate_bypass": False}


def post(path, body=None, timeout=30):
    req = urllib.request.Request(BASE_URL + path, json.dumps(body or {}).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(path):
    with urllib.request.urlopen(BASE_URL + path, timeout=10) as r:
        return json.loads(r.read())


def wait(pred, timeout, step=0.05):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if pred():
            return True
        time.sleep(step)
    return pred()


def start(settings_=None, go_body=None):
    """What the status page's Start does: /left/arm, then /left/go."""
    c1, a = post("/left/arm", settings_ or MATCH)
    c2, g = post("/left/go", go_body or {})
    return c1, a, c2, g


def stop_all():
    post("/left/stop")
    wait(lambda: not ctl._busy(), 15)


def t_go_twice():
    G.cams("ok")
    G.FAKE.mode, G.FAKE.fast, G.FAKE.agentic = None, "no", ["no"]
    HOLD["kw"] = {}
    c1, a, c2, g = start()
    ref1 = (g.get("gate") or {}).get("ref_t")
    c3, g2 = post("/left/go")
    record("OK" if c3 == 409 and ctl.state == "LOCKED" else "BUG", "5a Go pressed twice", f"first {c2}, second {c3} {g2.get('why')}; gate re-armed by the second: {bool(g2.get('gate'))}")
    stop_all()


def t_latch_leak_via_start():
    G.cams("ok")
    G.FAKE.mode, G.FAKE.fast, G.FAKE.agentic = None, "yes", ["yes"]
    HOLD["kw"] = {}
    start()
    opened = wait(lambda: ctl.state in ("STACK", "SETTLE_CHECK") and len(ctl.tower) >= 1, 15)
    record("NOTE", "8 round 1: gate opened and left stacked", f"opened={opened} tower={len(ctl.tower)}")
    stop_all()
    # round 2: the overhead camera is down at Start, the VLM now says no to everything
    G.FAKE.fast, G.FAKE.agentic = "no", ["no"]
    G.cams("missing")
    time.sleep(2.5)
    t0 = time.time()
    c1, a, c2, g = start()
    time.sleep(4)
    b = HOLD["b"]
    st = get("/api/state")
    leaked = b.count("place_next", t0) > 0 or ctl.state in ("STACK", "SETTLE_CHECK")
    record("BUG" if leaked else "OK", "8e Start with no fresh overhead after an opened round: left STACKS on last round's latch (DQ)",
           f"go={c2} gate arm={(g.get('gate') or {}).get('why', g.get('gate'))!r}; state={ctl.state} place_next calls={b.count('place_next', t0)} "
           f"/move={LEFT.motion_calls(t0)} gate.open={st['gate'].get('open')} gate.armed={st['gate'].get('armed')}")
    stop_all()
    G.cams("ok")
    time.sleep(1.5)
    # keep_gate in match: the gate is not re-armed at all
    G.FAKE.fast, G.FAKE.agentic = "yes", ["yes"]
    start()
    wait(lambda: ctl.state in ("STACK", "SETTLE_CHECK"), 15)
    stop_all()
    G.FAKE.fast, G.FAKE.agentic = "no", ["no"]
    t0 = time.time()
    c1, a, c2, g = start(go_body={"keep_gate": True})
    time.sleep(3)
    record("BUG" if HOLD["b"].count("place_next", t0) else "OK", "8f /left/go {keep_gate:true} accepted in match mode: no new reference, old latch reused",
           f"go={c2} state={ctl.state} place_next calls={HOLD['b'].count('place_next', t0)}")
    stop_all()


def fresh_round_open():
    """A round whose gate opens now (fresh reference)."""
    G.cams("ok")
    G.FAKE.fast, G.FAKE.agentic = "no", ["no"]
    start()
    time.sleep(0.8)
    G.FAKE.fast, G.FAKE.agentic = "yes", ["yes"]
    return wait(lambda: ctl.state in ("STACK", "SETTLE_CHECK"), 15)


def t_restart_mid_job():
    HOLD["kw"] = {"step_s": 6.0, "moves": 12}
    fresh_round_open()
    wait(lambda: ctl._busy() and HOLD["b"].count("place_next") >= 1, 8)
    old = HOLD["b"]
    G.FAKE.fast, G.FAKE.agentic = "no", ["no"]  # new round: the right arm has not unplugged anything yet
    t0 = time.time()
    c, r = post("/left/restart", {"settings": MATCH})
    dt = time.time() - t0
    m0 = old.moves_sent
    time.sleep(4)
    new = HOLD["b"]
    record("BUG" if old.moves_sent > m0 else "OK", "5b Restart mid-placement: the old job stops moving (hold sets ABORT)",
           f"restart {c} in {dt:.1f}s; old job /move after restart={old.moves_sent - m0}; new round state={ctl.state}, "
           f"new backend look moves={new.moves_sent if new is not old else 0}, new-round tower={len(ctl.tower)}, faults={ctl.faults}")
    stop_all()


def t_stop_during_prepare():
    HOLD["kw"] = {"prepare_s": 3.0}
    G.cams("ok")
    G.FAKE.fast, G.FAKE.agentic = "no", ["no"]
    start()
    wait(lambda: ctl._busy(), 3)
    time.sleep(0.5)
    t0 = time.time()
    post("/left/stop")
    time.sleep(4)
    record("OK" if LEFT.motion_calls(t0) == 0 else "BUG", "5c Stop during prepare: no motion after Stop", f"/move after stop={LEFT.motion_calls(t0)} state={ctl.state}")
    pend = ctl._job_result
    record("NOTE", "5d the stopped prepare's result is left pending for the next round", f"pending={bool(pend)} kind={(pend or {}).get('kind')}")
    HOLD["kw"] = {}


def t_estop_then_start():
    HOLD["kw"] = {"step_s": 4.0, "moves": 8}
    fresh_round_open()
    wait(lambda: ctl._busy() and HOLD["b"].count("place_next") >= 1, 8)
    t0 = time.time()
    post("/left/estop")
    time.sleep(1.0)
    m = LEFT.motion_calls(t0 + 0.3)
    G.FAKE.fast, G.FAKE.agentic = "no", ["no"]
    t1 = time.time()
    c1, a, c2, g = start()
    time.sleep(4)
    record("OK" if m == 0 and LEFT.motion_calls(t1 + 1.0) == 0 else "BUG", "5e E-stop mid-placement, then Start (gate shut)",
           f"/move after estop={m}, /move after Start={LEFT.motion_calls(t1 + 1.0)}, state={ctl.state} estop={ctl.estop} faults={ctl.faults} reason={ctl.reason[:50]!r}")
    stop_all()
    HOLD["kw"] = {}


def main():
    only = os.environ.get("ONLY")
    for t in (t_go_twice, t_latch_leak_via_start, t_restart_mid_job, t_stop_during_prepare, t_estop_then_start):
        if only and only not in t.__name__:
            continue
        try:
            t()
        except Exception:  # noqa: BLE001
            import traceback
            record("BUG", t.__name__ + " crashed", traceback.format_exc()[-500:])
    ctl.shutdown()
    srv.shutdown()
    jail.summary()


if __name__ == "__main__":
    main()
