"""Live-loop test of the gate service against a station sim (no USB strip there, so the gate must stay shut).

    python -m hillclimb.gate.tests.live_sim --sim-url http://127.0.0.1:19101 --port 18771 [--kill-cmd "..."] [--seconds 40]

Starts the standalone gate server as a child process pointed at the sim (overhead frames and /state), then checks:
  1. health: frames and jaw state fresh, key present
  2. calibrate (auto strip finder on the sim's photo background)
  3. arm, then the loop runs: reads arrive, the gate stays shut, nothing crashes
  4. the sim stops (kill-cmd): inputs go stale, reads become "no", the gate stays shut, the server still answers
  5. restart of the gate server: the armed round is restored from disk, still shut
  6. latch persistence: a saved open gate comes back open after a restart
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request


def call(url, method="GET", body=None, timeout=5):
    data = json.dumps(body or {}).encode() if method == "POST" else None
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except Exception as e:  # noqa: BLE001
        return 0, {"error": f"{type(e).__name__}: {e}"}


def start_server(port, sim_url, env):
    return subprocess.Popen([sys.executable, "-m", "hillclimb.gate.serve", "--port", str(port), "--left-url", sim_url, "--right-url", sim_url],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_up(base, secs=20):
    t = time.time()
    while time.time() - t < secs:
        code, _ = call(base + "/gate/health", timeout=2)
        if code == 200:
            return True
        time.sleep(0.5)
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-url", required=True)
    ap.add_argument("--port", type=int, default=18771)
    ap.add_argument("--kill-cmd", default="")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--log-dir", default="")
    a = ap.parse_args(argv)
    log_dir = pathlib.Path(a.log_dir or f"/tmp/gate_live_{int(time.time())}")
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LOG_DIR=str(log_dir), GATE_FRAME_MAX_AGE_S="2.0")
    base = f"http://127.0.0.1:{a.port}"
    results = []

    def ok(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(("ok   " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)

    srv = start_server(a.port, a.sim_url, env)
    try:
        ok("server up", wait_up(base))
        time.sleep(3)
        code, h = call(base + "/gate/health")
        ok("health: overhead fresh", h.get("overhead_fresh"), json.dumps(h)[:200])
        ok("health: jaw state fresh", h.get("jaw_fresh"))
        ok("health: key present", h.get("key_present"))
        code, c = call(base + "/gate/calibrate", "POST", {})
        ok("calibrate answers", code == 200, json.dumps(c)[:200])
        code, r = call(base + "/gate/arm", "POST", {})
        ok("arm", code == 200 and r.get("ok"), json.dumps(r)[:200])
        t_end = time.time() + a.seconds
        opened, reads = False, 0
        while time.time() < t_end:
            code, s = call(base + "/gate/state")
            if code != 200:
                ok("state answers during loop", False, json.dumps(s)[:200])
                break
            opened |= bool(s.get("open"))
            reads = len(s.get("last_reads") or [])
            time.sleep(2)
        code, s = call(base + "/gate/state")
        ok("reads arrive", reads > 0, f"{reads} reads; last {json.dumps((s.get('last_reads') or [{}])[-1])[:240]}")
        ok("gate stays shut on a table with no unplug", not opened and not s.get("open"))
        ok("no loop errors", s.get("loop_errors", 1) == 0, f"loop_errors={s.get('loop_errors')}")
        if a.kill_cmd:
            subprocess.run(a.kill_cmd, shell=True)
            time.sleep(8)
            code, s = call(base + "/gate/state")
            ok("server answers with the sim gone", code == 200)
            ok("inputs stale: unhealthy, not known", s.get("checks", {}).get("healthy") == "no" and not s.get("known"), json.dumps(s.get("checks"))[:200])
            ok("gate still shut", not s.get("open"))
            last = (s.get("last_reads") or [{}])[-1]
            ok("reads become stale no", last.get("yes") is False, json.dumps(last)[:200])
    finally:
        srv.terminate()
        try:
            srv.wait(5)
        except Exception:  # noqa: BLE001
            srv.kill()
    # restart: the armed round comes back from disk, still shut
    srv = start_server(a.port, a.sim_url, env)
    try:
        ok("server restarts", wait_up(base))
        code, s = call(base + "/gate/state")
        ok("restart restores the armed round", s.get("armed") and not s.get("open"), json.dumps({k: s.get(k) for k in ("armed", "open", "latched_at")}))
    finally:
        srv.terminate()
        srv.wait(5)
    # latch persistence: a saved open gate comes back open
    st = log_dir / "gate_state.json"
    data = json.loads(st.read_text())
    data.update(open=True, latched_at=time.time())
    st.write_text(json.dumps(data))
    srv = start_server(a.port, a.sim_url, env)
    try:
        ok("server restarts again", wait_up(base))
        code, s = call(base + "/gate/state")
        ok("a latched open gate stays open after a restart", s.get("open") is True)
    finally:
        srv.terminate()
        srv.wait(5)
    good = all(r for _, r in results)
    print("ALL PASS" if good else "SOME FAILED", flush=True)
    print(f"logs in {log_dir}")
    return good


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
