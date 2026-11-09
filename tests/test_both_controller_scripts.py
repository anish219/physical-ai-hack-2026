import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from dream_hackathon import sequences


def test_controller_starts_and_stops_both_loaded_scripts(monkeypatch, tmp_path):
    left, left_start, _ = sequences.load_script("hillclimb/left/match.py:start")
    right, right_start, _ = sequences.load_script("scripts/start_right_arm_simple.py:start")
    calls = []
    monkeypatch.setattr(left, "ensure_server", lambda: None)
    monkeypatch.setattr(left, "server_up", lambda: True)
    monkeypatch.setattr(left, "_request", lambda path, *a, **kw: calls.append(path) or {"ok": True})
    worker = Mock()
    worker.stop = None
    worker.poll.return_value = None
    monkeypatch.setattr(right.subprocess, "Popen", lambda argv, **kw: calls.append(argv) or worker)
    monkeypatch.setattr(right.Path, "mkdir", lambda *a, **kw: None)
    loaded = {
        "hillclimb/left/match.py:start": (left, left_start, "start"),
        "scripts/start_right_arm_simple.py:start": (right, right_start, "start"),
    }
    monkeypatch.setattr(sequences, "load_script", lambda ref: loaded[ref])
    nodes = {side: SimpleNamespace(snapshot=lambda: {"arm_open": False}) for side in ("left", "right")}
    callers = {side: SimpleNamespace(sides=(side,)) for side in ("left", "right", "both")}
    manager = sequences.SequenceManager(callers, nodes, lambda event: None)
    for side, ref in zip(("left", "right"), loaded, strict=False):
        manager.update(side, {"mode": "script", "scriptRef": ref, "phases": []})
    try:
        assert manager.start_match() == ["left", "right"]
        deadline = time.monotonic() + 3
        while len(calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "/left/arm" in calls and "/left/go" in calls
        argv = next(item for item in calls if isinstance(item, list))
        assert Path(argv[1]).name == "run_right_arm_simple.py"
        assert "--run-id" in argv and "--go-time" in argv
    finally:
        manager.close()
        deadline = time.monotonic() + 3
        while any(run.snapshot()["busy"] for run in manager.runs.values()) and time.monotonic() < deadline:
            time.sleep(0.01)
    assert "/left/stop" in calls
    worker.send_signal.assert_called_once()
