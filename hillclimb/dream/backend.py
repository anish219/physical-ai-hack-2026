"""The Dream stacking backend: the StackBackend interface (hillclimb/INTERFACES.md) run by a Dream-1 fine-tune.

The rules come from the station-sim experiments of Nov 8 (docs/SPIKES.md, section 3):
- the policy never stops by itself, so a step ends on the jaw: HOLDING for at least 1.5 s, then released with the
  jaws open, then 2 s more (the policy backs off as in training), then stop;
- every prompt starts from the COMMANDED training start pose;
- while the jaws hold nothing, rows that would put the hand into the tower's column are refused; the jaws are never
  opened by the harness inside the column; the retreat goes up first, then across;
- the policy stays open and warm between prompts (inspect-only calls every 30 s, rows discarded);
- the controller counts height by "grew by one and stands", any colour; this backend reports what the jaw saw and
  leaves `verified` false, so the controller asks Terra.

Nothing here raises to the caller: every failure comes back as {"ok": False, "why": ...}.
"""

from __future__ import annotations

import time

from hillclimb.common.log import JsonlLog
from hillclimb.common.wording import table_for

from . import geometry as G
from .arm import Arm
from .describe import prompt_from_positions, prompt_from_read, read_scene
from .loop import DreamLoop

DEFAULT_CHECKPOINT = "dream-1-ft:f9ed26ed-blocks-left-fresh500-20261108"
HOLD_MIN_S = 1.5
AFTER_RELEASE_S = 2.0
OPEN_MARGIN = 0.05  # released: no longer holding and opened this much past the width it held at
BLOCK_H = 0.0345
TABLE_Z = -0.008


class DreamBackend:
    def __init__(self, arm, checkpoint: str = DEFAULT_CHECKPOINT, *, wording: str = "", hz: float = 20.0,
                 keep_out: bool = True, tower_xy=None, playback: str = "row0", log_dir: str | None = None, client_factory=None) -> None:
        self.arm = arm if isinstance(arm, Arm) else (Arm.from_url(arm) if isinstance(arm, str) else Arm(arm))
        self.checkpoint = checkpoint
        self.table = wording or table_for(checkpoint)
        self.keep_out = keep_out
        self.tower_xy = tuple(tower_xy) if tower_xy else None  # known tower spot (m); else learned from releases
        self.column: G.Column | None = None
        self.log = JsonlLog("dream", log_dir)
        kw = {"client_factory": client_factory} if client_factory else {}
        self.loop = DreamLoop(self.arm, checkpoint, hz=hz, playback=playback, log=lambda rec: self.log.write(rec.pop("ev", "loop"), **rec), **kw)
        self.loop.row_filter = self._row_ok

    # ---------------------------------------------------------------- the interface
    def prepare(self, scene: dict | None = None) -> None:
        """Open the policy and start warming it. Touches no block; the arm does not move."""
        self.loop.open()

    def status(self) -> dict:
        return {"backend": "dream", "checkpoint": self.checkpoint, "wording": self.table, **self.loop.status,
                "tower_xy": self.tower_xy, "column": self.column.as_dict() if self.column else None}

    def hold(self) -> None:
        self.loop.stop()

    def retreat(self) -> None:
        top = self.column.top_z if self.column else None
        self.log.write("retreat", steps=G.retreat(self.arm, top, self.column))

    def recover(self, scene: dict, deadline_s: float = 60.0) -> dict:
        return {"ok": False, "grew": False, "verified": False, "placed_id": None,
                "why": "the Dream backend does not rebuild a fallen tower; use the scripted backend"}

    def close(self) -> None:
        self.loop.close()

    def place_next(self, scene: dict | None = None, deadline_s: float = 75.0) -> dict:
        """One block onto the tower.

        `scene` from the controller: {"tower_blocks": n, ...}; optional "tower_xy": [x, y] (m) and "loose": [{id,
        colour, x, y, lying}] with "top": {colour, x, y} (sim tests). Without positions, Terra reads the frames for
        the tower top's colour and the nameable loose blocks.
        Returns {"ok", "grew", "verified", "placed_id", "prompt", "why", "timing"}: ok and grew mean the jaw held a
        block for 1.5 s and then let it go (over the tower when the tower's position is known); verified is always
        false here, so the controller confirms the growth with vision."""
        scene = scene or {}
        t0 = time.time()
        res = {"ok": False, "grew": False, "verified": False, "placed_id": None, "prompt": None, "why": "", "timing": {}}
        try:
            if not self.loop.ready():
                st = self.loop.status
                res["why"] = f"policy {st['mode']}" + (f": {st['error']}" if st.get("error") else "")
                return self._done(res)
            n = int(scene.get("tower_blocks") or len(scene.get("tower") or []) or 0)
            if n < 1:
                res["why"] = "no tower yet: place the base with the scripted backend first"
                return self._done(res)
            chosen = self._choose(scene, n)
            if chosen is None:
                res["why"] = "no loose block could be named unambiguously"
                return self._done(res)
            prompt, block = chosen
            res.update(prompt=prompt, placed_id=block.get("id") or block["colour"])
            xy = scene.get("tower_xy") or self.tower_xy
            top_z = TABLE_Z + n * BLOCK_H
            self.column = G.Column(xy[0], xy[1], top_z) if (self.keep_out and xy) else None
            if not self._at_start():
                G.retreat(self.arm, top_z, self.column)
            t_go = time.time()
            self.loop.start(prompt)
            ev = self._watch(t_go, deadline_s)
            self.loop.stop()
            ev["policy_s"] = round(time.time() - t_go, 1)
            res["timing"] = ev
            released = ev.get("why") == "released"
            res["ok"] = released or bool(ev.get("maybe_placed"))
            res["grew"] = bool((released and ev.get("released_in_column", True)) or ev.get("maybe_placed"))
            res["why"] = ev.get("why", "")
            if released and ev.get("release_xy") and not self.tower_xy and ev.get("released_in_column", True):
                self.tower_xy = tuple(ev["release_xy"])  # learned: where the first good place happened
            res["timing"]["retreat"] = G.retreat(self.arm, top_z + (BLOCK_H if released else 0.0), self.column)
            res["timing"]["wall_s"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            self.loop.stop()
            res["why"] = f"error: {type(e).__name__}: {e}"[:300]
        return self._done(res)

    # ---------------------------------------------------------------- helpers
    def _done(self, res: dict) -> dict:
        self.log.write("place_next", **res)
        return res

    def _choose(self, scene: dict, n: int):
        if scene.get("loose") and scene.get("top"):
            geo = G.geometry()
            hand = tuple(geo.fk_command(G.START_Q)[0][:2]) if geo else (0.277, 0.0)
            return prompt_from_positions(scene["loose"], scene["top"], self.table, hand)
        frames = {}
        for name in ("overhead", "wrist"):
            try:
                frames[name] = self.arm.frame(name)
            except Exception:  # noqa: BLE001
                frames[name] = None
        if not frames.get("overhead"):
            return None
        read = read_scene(frames, n, self.table)
        self.log.write("scene_read", read=read)
        return prompt_from_read(read, self.table, n) if read else None

    def _at_start(self) -> bool:
        try:
            goal = self.arm.state().get("goal") or []
        except Exception:  # noqa: BLE001
            return False
        return len(goal) == 6 and max(abs(a - b) for a, b in zip(goal, G.START_Q)) < 0.05

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
        g_held = None
        refused0 = self.loop.status["rows_refused"]
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
                g_held = g
                if not held_long and t - hold_since >= HOLD_MIN_S:
                    held_long = True
                    ev["grasp_s"] = round(hold_since, 1)
            else:
                hold_since = None
            if held_long and not holding and g_held is not None and g > g_held + OPEN_MARGIN and let_go is None:
                let_go = t
                ev["release_s"] = round(t, 1)
                geo = G.geometry()
                if geo is not None and st.get("q"):
                    p, R = geo.fk_measured(st["q"])
                    ev["release_xy"] = [round(float(p[0]), 4), round(float(p[1]), 4)]
                    if self.column is not None:
                        ev["released_in_column"] = bool(self.column.hand_inside(geo, p, R) or (
                            abs(float(p[0]) - self.column.x) < 0.05 and abs(float(p[1]) - self.column.y) < 0.05))
            if let_go is not None and t - let_go >= AFTER_RELEASE_S:
                ev["why"] = "released"
                break
            time.sleep(0.1)
        if ev["why"] == "deadline" and held_long and let_go is None:
            st = self.loop.last_state or {}
            if not st.get("grip_holding"):
                ev["why"] = "deadline after a grasp, jaw now empty"  # it may have placed: the controller's vision decides
                ev["maybe_placed"] = True
        ev["rows_refused"] = self.loop.status["rows_refused"] - refused0
        return ev
