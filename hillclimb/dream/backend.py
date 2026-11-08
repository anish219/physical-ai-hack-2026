"""The Dream stacking backend: the StackBackend interface (hillclimb/INTERFACES.md) run by a Dream-1 fine-tune.

The rules come from the station-sim experiments of Nov 8 (docs/SPIKES.md, section 3):
- the policy never stops by itself, so a step ends on the jaw: HOLDING for at least 1.5 s, then released with the
  jaws open, then 2 s more (the policy backs off as in training), then stop;
- every prompt starts from the COMMANDED training start pose;
- while the jaws hold nothing, rows that would put the hand into the tower's column are refused; the jaws are never
  opened inside the column; the retreat goes up first, then across;
- the policy stays open and warm between prompts (inspect-only calls, rows discarded);
- a step reports what the jaw saw; whether the tower grew is judged by the left controller ("grew by one and
  stands", any colour).

Nothing here raises to the caller: every failure comes back as {"ok": False, "why": ...}.
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np

from . import geometry as G
from .arm import ArmClient
from .loop import DreamLoop
from .wording import stack_prompt

DEFAULT_CHECKPOINT = "dream-1-ft:f9ed26ed-blocks-left-fresh500-20261108"
HOLD_MIN_S = 1.5
AFTER_RELEASE_S = 2.0
OPEN_IS = 0.70  # normalized jaw opening that counts as open
AVOID_COLOURS = ("purple", "yellow")  # B misses these most often in the sim; used only to order candidates


class DreamBackend:
    def __init__(self, left_url: str, checkpoint: str = DEFAULT_CHECKPOINT, *, hz: float = 20.0,
                 keep_out: bool = True, log_dir: str | None = None, client_factory=None) -> None:
        self.arm = ArmClient(left_url)
        self.checkpoint = checkpoint
        self.keep_out = keep_out
        self.column: G.Column | None = None
        self.log_path = pathlib.Path(log_dir) / "dream.jsonl" if log_dir else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        kw = {"client_factory": client_factory} if client_factory else {}
        self.loop = DreamLoop(self.arm, checkpoint, hz=hz, log=self._log, **kw)
        self.loop.row_filter = self._row_ok

    # ---------------------------------------------------------------- logging
    def _log(self, rec: dict) -> None:
        rec = {"t": round(time.time(), 3), "part": "dream", **rec}
        if self.log_path:
            try:
                with self.log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass

    # ---------------------------------------------------------------- the interface
    def prepare(self, scene: dict | None = None) -> None:
        """Open the policy and start warming it. Touches no block; the arm does not move."""
        self.loop.open()

    def status(self) -> dict:
        return {"checkpoint": self.checkpoint, **self.loop.status,
                "column": self.column.as_dict() if self.column else None}

    def hold(self) -> None:
        self.loop.stop()

    def retreat(self) -> None:
        top = self.column.top_z if self.column else None
        self._log({"ev": "retreat", "log": G.retreat(self.arm, top, self.column)})

    def recover(self, scene: dict, deadline_s: float) -> dict:
        return {"ok": False, "why": "the Dream backend does not recover a fallen tower; use the scripted backend"}

    def place_next(self, scene: dict, deadline_s: float = 75.0) -> dict:
        """One block onto the tower. `scene`: {"tower": {"x", "y", "top_z", "top": block, "height": n},
        "loose": [block, ...], "tower_count": n}; blocks are {id, colour, x, y, level, lying}. Returns
        {"ok", "placed_id", "prompt", "why", "timing"}: ok means the jaw held a block for 1.5 s and then let go; the
        caller still checks that the tower grew."""
        t0 = time.time()
        res = {"ok": False, "placed_id": None, "prompt": None, "why": "", "timing": {}}
        try:
            if not self.loop.ready():
                res["why"] = f"policy {self.loop.status['mode']}: {self.loop.status['error']}".strip(": ")
                return res
            tower = scene.get("tower") or {}
            top = tower.get("top")
            if not top:
                res["why"] = "no tower top in the scene (place the base with the scripted backend first)"
                return res
            pick, prompt = self._choose(scene)
            if prompt is None:
                res["why"] = "no loose block can be named unambiguously"
                return res
            res.update(placed_id=pick["id"], prompt=prompt)
            self.column = G.Column(tower["x"], tower["y"], tower["top_z"]) if self.keep_out else None
            if not self._at_start():
                G.retreat(self.arm, tower.get("top_z"), self.column)
            t_go = time.time()
            self.loop.start(prompt)
            ev = self._watch(t_go, deadline_s)
            self.loop.stop()
            ev["policy_s"] = round(time.time() - t_go, 1)
            res["timing"] = ev
            res["ok"] = ev.get("why") == "released"
            res["why"] = ev.get("why", "")
            log = G.retreat(self.arm, tower.get("top_z"), self.column)
            res["timing"]["retreat"] = log
            res["timing"]["wall_s"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            self.loop.stop()
            res["why"] = f"error: {e}"[:300]
        self._log({"ev": "place_next", **{k: v for k, v in res.items() if k != "timing"}, "timing": res["timing"]})
        return res

    # ---------------------------------------------------------------- helpers
    def _choose(self, scene: dict):
        """The loose block to name: nameable, then nearest the start hand position, avoiding the colours B misses
        most. Any colour counts for height."""
        loose = [b for b in scene.get("loose", []) if b.get("level", 0) == 0]
        geo = G.geometry()
        hx, hy = (geo.fk_command(G.START_Q)[0][:2] if geo else (0.25, 0.0))
        order = sorted(loose, key=lambda b: (b["colour"] in AVOID_COLOURS, math.hypot(b["x"] - hx, b["y"] - hy)))
        top = scene["tower"]["top"]
        for b in order:
            p = stack_prompt(self.checkpoint, b, {**top, "level": scene["tower"].get("height", 1) - 1}, loose,
                             scene.get("tower_count", 1))
            if p:
                return b, p
        return None, None

    def _at_start(self) -> bool:
        try:
            q = np.asarray(self.arm.state()["goal"] or [], float)
        except Exception:  # noqa: BLE001
            return False
        return q.shape == (6,) and float(np.abs(q - np.asarray(G.START_Q)).max()) < 0.05

    def _row_ok(self, row: list[float], st: dict) -> bool:
        """Refuse a row that would put the hand into the tower column while the jaws hold nothing."""
        if self.column is None or st.get("grip_holding"):
            return True
        geo = G.geometry()
        if geo is None:
            return True
        p, R = geo.fk_command(row[:6])
        return not self.column.hand_inside(geo, p, R)

    def _watch(self, t_go: float, deadline_s: float) -> dict:
        """Watch the jaw until held >= 1.5 s, released open, + 2 s; or the deadline; or a loop failure."""
        ev = {"why": "deadline"}
        hold_since = None
        held_long = False
        let_go = None
        while True:
            t = time.time() - t_go
            if t > deadline_s:
                break
            if self.loop.status["mode"] in ("error", "warming", "opening"):
                ev["why"] = f"policy {self.loop.status['mode']}: {self.loop.status['error']}"[:200]
                break
            st = self.loop.last_state or {}
            holding = bool(st.get("grip_holding"))
            g = float(st.get("grip", 1.0))
            if holding:
                hold_since = hold_since if hold_since is not None else t
                if not held_long and t - hold_since >= HOLD_MIN_S:
                    held_long = True
                    ev["grasp_s"] = round(hold_since, 1)
            else:
                hold_since = None
            if held_long and not holding and g > OPEN_IS and let_go is None:
                let_go = t
                ev["release_s"] = round(t, 1)
                geo = G.geometry()
                if self.column is not None and geo is not None:
                    p, R = geo.fk_measured(st["q"])
                    ev["released_in_column"] = self.column.hand_inside(geo, p, R)
            if let_go is not None and t - let_go >= AFTER_RELEASE_S:
                ev["why"] = "released"
                break
            time.sleep(0.1)
        ev["rows_refused"] = self.loop.status["rows_refused"]
        return ev
