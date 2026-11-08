"""Replays the recorded USB episodes through the gate's own fast and agentic checks and scores every decision.

    python -m hillclimb.gate.tests.replay_stress VIDS LABELS.json GRIP.json OUT.jsonl [--workers 8] [--limit N]

VIDS holds unplug/ and insert/ LeRobot video trees (overhead "scene" camera). LABELS is the certain-label set
(labels4_certain.json) and GRIP the 10 Hz jaw traces (grip10hz.json) from the experiments. For each case, the gate
logic runs as it would live: the fast check at "now"; on a yes, an agentic check; on a yes, a second agentic check on
a frame 2 s later. The gate's "go" is scored against the truth at the moment of the second check:
  unplug: "no" up to 0.5 s before the jaw releases, "go" from 1 s after, unsure in between (left out)
  insert: "no" while the jaw holds the cable, unsure otherwise (left out)
Reference frames are the episode start (0.3 s). Needs ffmpeg on PATH. Keys from the environment or KEYS_FILE.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from hillclimb.gate import imaging, settings, vlm
from hillclimb.gate.frames import ListFrameSource


def hold_run(g):
    best, s = (0, -1), -1
    for i in range(len(g) + 1):
        on = i < len(g) and 3 <= g[i] <= 8.5
        if on and s < 0:
            s = i
        if not on and s >= 0:
            if i - s > best[1] - best[0] + 1:
                best = (s, i - 1)
            s = -1
    return best


def truth(ds, g, t):
    if ds == "unplug":
        a, b = hold_run(g)
        release = (b + 1) / 10
        if t >= release + 1.0:
            return "go"
        if t <= release - 0.5:
            return "no"
        return "unsure"
    i = min(len(g) - 1, round(t * 10))
    w = g[max(0, i - 2):i + 3]
    return "no" if w and all(3 <= x <= 8.5 for x in w) else "unsure"


class VideoFrames:
    """Frames from one episode's overhead video by time, cached, via ffmpeg."""

    _cache: dict = {}
    _lock = threading.Lock()

    def __init__(self, path: pathlib.Path, dur: float):
        self.path, self.dur = path, dur

    def __call__(self, t: float):
        t = max(0.05, min(self.dur - 0.05, t))
        key = (str(self.path), round(t, 1))
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(self.path), "-frames:v", "1", "-q:v", "3",
                            "-f", "image2pipe", "-vcodec", "mjpeg", "-"], capture_output=True)
        data = r.stdout if r.returncode == 0 and r.stdout else None
        with self._lock:
            self._cache[key] = data
        return data


def run_case(vids, case_id, grip):
    ds, ep, _, ts = case_id.split("_")
    g = grip[f"{ds}/{ep}"]
    dur = len(g) / 10
    t = float(ts)
    video = vids / ds / "chunk-000" / "observation.images.scene" / f"episode_{ep}.mp4"
    frames = VideoFrames(video, dur)
    box = settings.get("STRIP_BOX")
    ref = frames(0.3)
    row = {"case": case_id, "t": t, "truth_now": truth(ds, g, t)}
    # the trigger (fast, then agentic) is at t; the confirming agentic check looks at a frame 2 s later when the
    # recording goes on, else 1.5 s earlier (a live gate's second check is newer; for scoring, only the pair matters).
    # The gate's decision happens at the later of the two moments and is scored against the truth there.
    t_second = t + 2.0 if t + 2.0 <= dur - 0.1 else (t - 1.5 if t - 1.5 >= 0.5 else None)
    src = ListFrameSource(frames, now_t=t, start_t=0.05, end_t=dur - 0.05)
    comp = imaging.composite(ref, src.latest().jpeg, box) if (ref and src.latest()) else None
    f = vlm.fast_check(comp)
    row["fast"] = {"yes": f["yes"], "ms": f["ms"], "error": f["error"]}
    if not f["yes"]:
        row["gate_go"] = False
        row["decision_t"] = t
        row["truth_decision"] = row["truth_now"]
        return row
    a1 = vlm.agentic_check(src, ref, t, box)
    row["agentic1"] = {k: a1[k] for k in ("yes", "ms", "error", "tools", "asked", "evidence")}
    if not a1["yes"]:
        row["gate_go"] = False
        row["decision_t"] = t
        row["truth_decision"] = row["truth_now"]
        return row
    t2 = t_second
    if t2 is None:
        row["gate_go"] = None  # the recording is too short to confirm on another frame
        row["decision_t"] = t
        row["truth_decision"] = row["truth_now"]
        return row
    src2 = ListFrameSource(frames, now_t=t2, start_t=0.05, end_t=dur - 0.05)
    a2 = vlm.agentic_check(src2, ref, t2, box)
    row["agentic2"] = {k: a2[k] for k in ("yes", "ms", "error", "tools", "asked", "evidence")}
    row["gate_go"] = bool(a2["yes"])
    row["decision_t"] = max(t, t2)
    row["truth_decision"] = truth(ds, g, max(t, t2)) if t2 > t else (row["truth_now"] if truth(ds, g, t2) == row["truth_now"] else "unsure")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("vids")
    ap.add_argument("labels")
    ap.add_argument("grip")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter", default="")
    a = ap.parse_args(argv)
    vids = pathlib.Path(a.vids)
    labels = json.loads(pathlib.Path(a.labels).read_text())
    grip = json.loads(pathlib.Path(a.grip).read_text())
    out = pathlib.Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()}
    cases = [c for c in labels if c not in done and a.filter in c]
    if a.limit:
        cases = cases[:a.limit]
    print(f"{len(cases)} cases to run ({len(done)} done)", flush=True)
    lock = threading.Lock()
    t0 = time.time()

    def work(c):
        try:
            row = run_case(vids, c, grip)
        except Exception as e:  # noqa: BLE001
            row = {"case": c, "error": f"{type(e).__name__}: {e}", "gate_go": False}
        with lock, open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, cases))
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    summarize(out)


def summarize(out):
    rows = [json.loads(l) for l in pathlib.Path(out).read_text().splitlines() if l.strip()]
    fast_yes = sum(1 for r in rows if r.get("fast", {}).get("yes"))
    false_go = [r["case"] for r in rows if r.get("gate_go") and r.get("truth_decision") == "no"]
    go_cases = [r for r in rows if r.get("truth_now") == "go"]
    caught = sum(1 for r in go_cases if r.get("gate_go"))
    no_cases = [r for r in rows if r.get("truth_decision") == "no" or (r.get("truth_now") == "no" and r.get("gate_go") is False)]
    ag1_false = sum(1 for r in rows if r.get("truth_now") == "no" and r.get("agentic1", {}).get("yes"))
    fast_false = sum(1 for r in rows if r.get("truth_now") == "no" and r.get("fast", {}).get("yes"))
    errors = sum(1 for r in rows if r.get("error") or (r.get("fast") or {}).get("error"))
    ms = sorted(r["fast"]["ms"] for r in rows if r.get("fast") and not r["fast"]["error"])
    ams = sorted(r[k]["ms"] for r in rows for k in ("agentic1", "agentic2") if r.get(k) and not r[k]["error"])
    unsure_go = [r["case"] for r in rows if r.get("gate_go") and r.get("truth_decision") == "unsure"]
    print(json.dumps({
        "cases": len(rows), "errors": errors,
        "gate_false_go": len(false_go), "false_go_cases": false_go,
        "certain_no_decisions": len(no_cases),
        "go_cases": len(go_cases), "gate_caught": caught,
        "fast_yes_on_no": fast_false, "agentic1_yes_on_no": ag1_false, "fast_yes_total": fast_yes,
        "go_on_unsure_moment": unsure_go,
        "fast_ms_p50": ms[len(ms) // 2] if ms else None, "fast_ms_p90": ms[int(len(ms) * 0.9)] if ms else None,
        "agentic_ms_p50": ams[len(ams) // 2] if ams else None, "agentic_ms_max": ams[-1] if ams else None,
    }, indent=1))


if __name__ == "__main__":
    main()
