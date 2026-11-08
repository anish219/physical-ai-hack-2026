"""Read-only view of the right arm for the status page: the newest relay run folder and the right arm server's state.

Never posts anything. Every read has a short timeout and a failure becomes a field saying so.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.request

from hillclimb.common.config import CFG

RELAY = pathlib.Path(os.environ.get("RELAY_DIR", "/tmp/relay"))


def _tail(path: pathlib.Path, n: int) -> list[dict]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64000))
            lines = f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _arm_state() -> dict:
    try:
        with urllib.request.urlopen(CFG["RIGHT_URL"].rstrip("/") + "/state", timeout=1.0) as r:
            s = json.loads(r.read())
        return {"ok": True, "moving": s.get("moving"), "grip": s.get("grip"), "holding": s.get("grip_holding"),
                "enabled": s.get("enabled"), "error": s.get("error")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:120]}


def status() -> dict:
    now = time.time()
    out: dict = {"t": now, "arm": _arm_state(), "run": None}
    try:
        runs = [p for p in RELAY.iterdir() if p.is_dir()]
    except OSError:
        runs = []
    if not runs:
        out["summary"] = "RIGHT: no run folder in " + str(RELAY)
        return out
    from hillclimb.common import relay as R
    run = R.right_run()
    if run is None:
        out["summary"] = "RIGHT: no right-status.json in any run folder of " + str(RELAY)
        return out
    st = {}
    try:
        st = json.loads((run / "right-status.json").read_text())
    except (OSError, ValueError):
        pass
    events = _tail(run / "events.jsonl", 5)
    fr = None
    try:
        fr = json.loads((run / "first-removal.json").read_text())
    except (OSError, ValueError):
        pass
    last_t = st.get("time") or (events[-1].get("time") if events else None) or run.stat().st_mtime
    age = now - float(last_t)
    phase = st.get("phase") or (events[-1].get("phase") if events else "unknown")
    running = phase not in ("stopped", "done", "finished", "error") and age < 15
    ev = []
    for e in events:
        ports = e.get("ports")
        ev.append({"time": e.get("time"), "phase": e.get("phase"), "error": e.get("error"),
                   "ports": [p.get("state") for p in ports] if isinstance(ports, list) else None})
    out["run"] = {"dir": run.name, "run_id": st.get("run_id") or run.name, "phase": phase, "age_s": round(age, 1),
                  "running": running, "events": ev,
                  "first_removal": ({"unplugged_count": fr.get("unplugged_count"), "observed_at": fr.get("observed_at")} if fr else None)}
    when = time.strftime("%H:%M:%S", time.localtime(float(last_t)))
    if running:
        out["summary"] = f"RIGHT: running, phase {phase}, last update {age:.0f} s ago"
    else:
        out["summary"] = f"RIGHT: not running (last run {when}, {phase})"
    return out
