"""The right worker's relay folders (/tmp/relay/<run>/), read-only, shared by the gate, the controller and the page.

A run folder is the right arm's only when it holds right-status.json. Folder mtimes are not used to pick the newest
run: other tools (and our own left-status.json) write into these folders and would make an old folder look new.
"""
from __future__ import annotations

import json
import os
import pathlib


def relay_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("RELAY_DIR", "/tmp/relay"))


def right_runs() -> list[pathlib.Path]:
    """Run folders holding right-status.json, newest right-status.json first."""
    try:
        runs = [p for p in relay_dir().iterdir() if p.is_dir() and (p / "right-status.json").is_file()]
    except OSError:
        return []
    return sorted(runs, key=lambda p: (p / "right-status.json").stat().st_mtime, reverse=True)


def right_run() -> pathlib.Path | None:
    runs = right_runs()
    return runs[0] if runs else None


def _first_event_time(run: pathlib.Path) -> float | None:
    try:
        with open(run / "events.jsonl", encoding="utf-8", errors="replace") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                try:
                    t = json.loads(line).get("time")
                except ValueError:
                    continue
                if isinstance(t, (int, float)):
                    return float(t)
    except OSError:
        pass
    return None


def run_window(run: pathlib.Path) -> tuple[float | None, float | None]:
    """(start, last update) wall times of a right run: its first event, and its right-status.json write."""
    start = _first_event_time(run)
    try:
        last = (run / "right-status.json").stat().st_mtime
    except OSError:
        last = None
    if start is None:
        try:
            start = min(p.stat().st_mtime for p in run.iterdir())
        except (OSError, ValueError):
            start = None
    return start, last
