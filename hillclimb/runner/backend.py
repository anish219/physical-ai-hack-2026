"""The scripted stacking backend: the tall-tower runner behind the StackBackend interface (hillclimb/INTERFACES.md).

prepare() looks at the table and plans the whole tower (spot, order, clearing moves) without touching a block.
place_next() runs the planned moves up to and including the next stack place (one more block on the tower), re-planning
from the belief when a move fails; levels 8 and 9 go through the tilted grip from a staged helper stack when helper
spots are given. Every call returns a dict; nothing raises out of this module.

Environment:
    RUNNER_SIM_PORT   run against calib/station_sim on 127.0.0.1:PORT instead of the real arm server
    LEFT_URL, TEAM_STATE_URL, DREAM_STATE_URL, REC_URL   the real station's URLs (defaults: 8691, 8081, 8084, 8697)
    GATE_FILE         the first pick or push waits until this file exists (the USB gate)
    TALL_KNOWN_BIG    piece sizes, e.g. '{"green": false, "orange": false}' for all-small cubes
    HILLCLIMB_RUNS    where sessions, records and status go (default ~/hillclimb_runs)
    HILLCLIMB_RECORD  1: episodes go to the recorder at REC_URL (a recorder error is only logged); anything else
                      (the match default): the recorder is never called
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import time as _wall
import traceback
import urllib.request

from hillclimb.runner.calib import arm_runner as AR
from hillclimb.runner.calib import data_loop as D
from hillclimb.runner.calib import opstack as O
from hillclimb.runner.calib import tall_tower as T

AI_PLANNER = os.environ.get("PLANNER", "ai") == "ai"  # an AI picks each next move; the planner checks only that move
TILT_LEVEL = 7  # tower height at which the next block needs the tilted grip (fingers down reaches 7)


class RunnerBackend:
    def __init__(self, sim_port=None, helper_spots=(), tilt_theta_deg=40.0, tilt_up_mm=25.0, tilt_sag_mm=25.0, log=None):
        self.sim_port = int(sim_port or os.environ.get("RUNNER_SIM_PORT") or 0) or None
        self.jlog = log  # hillclimb.common.log.JsonlLog, when the controller gives one
        self.helper_spots = list(helper_spots)  # [(x, y) m]: helper stacks staged for levels 8 and 9
        self.tilt = (math.radians(tilt_theta_deg), tilt_up_mm / 1000, tilt_sag_mm / 1000)
        self.t = None  # the runner (tall_tower.Tall)
        self.tower: list[str] = []
        self.moves: list = []
        self.spot = None
        self.replans = 0
        self.avoid: set[str] = set()  # blocks the AI or a failed move ruled out for this round
        self.refusals: list[str] = []
        self.used_helpers: set[int] = set()
        self.log: list[dict] = []
        self.prepared = False
        if self.sim_port:
            T.use_sim(self.sim_port, str(T._RUNS_ROOT / "sim_station" / f"p{self.sim_port}"))
        else:  # the real station: its own local URLs (LEFT_URL, TEAM_STATE_URL, DREAM_STATE_URL, REC_URL), the wall clock
            AR.use_backend(dict(AR.REAL), _wall)

    # ---- helpers
    def _note(self, **row):
        row["t"] = round(_wall.time(), 2)
        self.log.append(row)
        if self.jlog is not None:
            try:
                self.jlog.write(str(row.get("what", "note")), **{k: v for k, v in row.items() if k != "tb"})
            except Exception:  # noqa: BLE001
                pass
        try:
            D.status("backend: " + json.dumps(row, default=str)[:400])
        except Exception:  # noqa: BLE001
            pass
        return row

    def _fail(self, what, e):
        return self._note(ok=False, what=what, why=f"{type(e).__name__}: {e}"[:400], tb=traceback.format_exc()[-1200:])

    def table_reset(self, why="table reset"):
        """Records before now no longer describe the table (a new match on a reset table): world_override.json's
        "since" moves to now, so neither the tower adoption nor the keep-out sees earlier towers. The old file is kept
        beside it as world_override.<time>.json."""
        ov = pathlib.Path(T.TALL) / "world_override.json"
        ov.parent.mkdir(parents=True, exist_ok=True)
        stamp = _wall.strftime("%Y-%m-%d %H:%M:%S")
        if ov.exists():
            ov.replace(ov.with_name(f"world_override.{_wall.strftime('%Y%m%d-%H%M%S')}.json"))
        ov.write_text(json.dumps({"since": stamp, "why": why}))
        self.tower, self.moves, self.prepared = [], [], False
        return self._note(ok=True, what="table_reset", since=stamp, why=why)

    ARM_OFF_WHY = "LEFT ARM IS OFF: its server reports no joint readings or not enabled; power it on, then press Start"

    @staticmethod
    def arm_off(st: dict | None) -> bool:
        """True when the arm server answers but the arm itself is closed or unpowered (q null, or enabled false)."""
        return not st or st.get("q") is None or st.get("grip") is None or st.get("enabled") is False

    def _arm_off_result(self, what, T0):
        """None while the arm is up; else a not-ok result (no motion attempted, no traceback spam)."""
        try:
            st = O.arm_state()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "grew": False, "verified": False, "placed_id": None, "arm_off": True,
                    "why": f"left arm server unreachable: {type(e).__name__}: {e}"[:200], "timing": {"s": round(_wall.time() - T0, 1)}}
        if self.arm_off(st):
            return {"ok": False, "grew": False, "verified": False, "placed_id": None, "arm_off": True, "why": self.ARM_OFF_WHY,
                    "timing": {"s": round(_wall.time() - T0, 1)}}
        return None

    def _ensure_runner(self):
        if self.t is None:
            for p in (D.TOKEN.parent, D.LOOP, T.TALL, O.RECORDS.parent):  # a fresh station has none of the run folders
                pathlib.Path(p).mkdir(parents=True, exist_ok=True)
            if not D.TOKEN.exists():
                D.take_token()
            self.t = T.Tall(T.TALL / _wall.strftime("%m%d-%H%M%S"))
            self.t.op.st["target"] = None
            self.t.no_perception = True
            self.t.no_vet = True

    def _stacks(self):
        """Standing stacks in the belief (blocks within 3 cm in xy), bottom first, tallest first."""
        bel = {k: b for k, b in self.t.belief.items() if not k.startswith("pile")}
        groups: list[list[str]] = []
        for k in sorted(bel, key=lambda k: T.bot(bel[k])):
            for g in groups:
                if math.hypot(bel[k]["x"] - bel[g[0]]["x"], bel[k]["y"] - bel[g[0]]["y"]) < 0.03:
                    g.append(k)
                    break
            else:
                groups.append([k])
        return sorted([g for g in groups if len(g) >= 2], key=len, reverse=True)

    tower_spot = None  # set per round from the controller settings ("x,y" cm)

    def _plan(self, budget=20.0):  # 40 s held the first pick 44 s after the looks (Nov 8 13:56)
        t = self.t
        stacks = self._stacks()
        # the runner's own records of raised places (the looks read tall tops poorly): those columns are towers too
        try:
            rec = T.world_from_records()
        except Exception:  # noqa: BLE001
            rec = {}
        cols: list[list] = []
        for b in rec.values():
            if (b.get("level") or 0) >= 1:
                for c in cols:
                    if math.hypot(b["x"] - c[0], b["y"] - c[1]) < 0.03:
                        c[2] += 1
                        break
                else:
                    cols.append([b["x"], b["y"], 1])
        near_col = {k for k, b in t.belief.items() for c in cols if math.hypot(b["x"] - c[0], b["y"] - c[1]) < 0.03}
        if not self.tower and (stacks or cols):  # a tower already stands (an earlier round, a restart): build on it
            if cols:
                c = max(cols, key=lambda c: c[2])
                self.tower = sorted([k for k, b in t.belief.items() if math.hypot(b["x"] - c[0], b["y"] - c[1]) < 0.03],
                                    key=lambda k: T.bot(t.belief[k]))
                if not self.tower:
                    raise D.Fatal(f"the records show a tower at ({c[0] * 100:.1f}, {c[1] * 100:.1f}) cm but no look finds it: "
                                  "not building elsewhere and not taking blocks from it")
            else:
                self.tower = list(stacks[0])
            self._note(ok=True, what="adopt_tower", tower=list(self.tower))
        covered = {k for g in stacks for k in g[:-1]}  # a block with another on it is never picked
        in_stacks = {k for g in stacks for k in g} | near_col
        left = [k for k, b in t.belief.items() if k not in self.tower and not k.startswith("pile") and k not in covered
                and k not in in_stacks and b.get("iou", 0) >= 0.5 and b["x"] >= 0.15]
        order = sorted(left, key=lambda k: (bool(t.belief[k].get("lying")), -T.bot(t.belief[k])))
        t.dr = {"carry_z": 0.15, "over_dz": 0.01}
        # the tower spot: chosen by best_sequence on its left-preference arc (30-55 deg left first, straight ahead as the
        # fallback), the spot that plans the most blocks winning, "more left" only breaking ties. TOWER_SPOT / tower_spot
        # ("x,y" cm) is only the first spot tried, never the only one (a pinned spot planned 0-1 blocks on Nov 8).
        spot_env = os.environ.get("TOWER_SPOT") or self.tower_spot
        t.spot_pin = tuple(float(v) / 100.0 for v in str(spot_env).split(",")) if spot_env else None
        res = t.best_sequence(order, tuple(self.tower), budget=budget)
        if res is None:
            raise D.Fatal(f"no tower spot plans: {dict(t.why)}")
        self.spot, base, moves, _stop = res
        if not self.tower:
            self.tower = list(base)
        self.moves = list(moves)
        return len(self.moves)

    FOLDED_REST = (0.0, -1.89, 1.73, -1.47, 0.0, 0.0)  # the left arm's power-on rest (Nov 7-8 logs), joints in rad

    def _unstick_start(self):
        """A stop mid-move can leave the arm where the pose check itself fails (Nov 8 14:46: "camera rolled under" at 0%
        of every path), so every move out is refused. When the fingertips are high (over 12 cm, clear of blocks), a joint
        move to the folded rest first; the unfold then starts from the known-good rest."""
        import numpy as np
        s = D.arm_state()
        q = s.get("q") or []
        if len(q) < 6 or max(abs(a - b) for a, b in zip(q[1:4], self.FOLDED_REST[1:4])) < 0.30:
            return
        t = self.t
        qm = np.array(q) + t.op.dq
        ok, why = t.op.safe.check_pose(qm, 0.07)
        if ok:
            return
        tip = t.op.tip(qm)[0]
        if tip[2] < 0.03 and math.hypot(tip[0], tip[1]) > 0.15:
            # low out on the table (a block may sit under it): straight up first, then fold (Isaiah: recover on its own)
            s_up = dict(s)
            up = list(q)
            up[1] = max(up[1] - 0.25, -2.6)
            body = {"q": [float(v) for v in up], "grip": float(s.get("goal_grip") or 0.8), "rate": 0.2}
            req = urllib.request.Request(O.URL + "/move", json.dumps(body).encode(), {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3).read()
            _wall.sleep(2.0)
        body = {"q": [float(v) for v in self.FOLDED_REST], "grip": float(s.get("goal_grip") or 0.8), "rate": 0.3}
        req = urllib.request.Request(O.URL + "/move", json.dumps(body).encode(), {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3).read()
        t_end = _wall.time() + 12.0
        while _wall.time() < t_end:
            _wall.sleep(0.3)
            q = D.arm_state().get("q") or q
            if max(abs(a - b) for a, b in zip(q[:4], self.FOLDED_REST[:4])) < 0.05:
                break
        self._note(ok=True, what="unstick", why=f"start pose failed ({why}); folded to rest from {tip[2] * 1000:.0f} mm")

    def _unfold_from_rest(self):
        """From the folded rest, go to the look pose first through the runner's home rules (to_pose). The arm server's
        start/zero joints can be anywhere after a reboot, so the unfold exception is given the present pose as the start,
        only when the arm really sits in the folded rest. Every route and sweep check still applies."""
        s = D.arm_state()
        q = s.get("q") or []
        if len(q) < 4 or max(abs(a - b) for a, b in zip(q[1:4], self.FOLDED_REST[1:4])) > 0.30:  # 0.15 missed a slumped rest (Nov 8 14:26)
            return
        if max(abs(a - b) for a, b in zip(q[1:4], self.FOLDED_REST[1:4])) > 0.05:
            # a slumped rest (Nov 8 14:29: shoulder -2.16, elbow 1.55): every check refuses the first move out of it
            # ("camera rolled under" at 0%), so first a small joint move back to the exact folded rest, curled at the base
            body = {"q": [float(v) for v in self.FOLDED_REST], "grip": float(s.get("goal_grip") or s.get("grip") or 0.8), "rate": 0.3}
            req = urllib.request.Request(O.URL + "/move", json.dumps(body).encode(), {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3).read()
            t_end = _wall.time() + 8.0
            while _wall.time() < t_end:
                _wall.sleep(0.3)
                q = D.arm_state().get("q") or q
                if max(abs(a - b) for a, b in zip(q[:4], self.FOLDED_REST[:4])) < 0.04:
                    break
            self._note(ok=True, what="rest_reset", why="slumped rest moved back to the folded rest", q=[round(v, 3) for v in q])
        real = D.arm_state

        def with_start(*a, **k):
            st = dict(real(*a, **k))
            st["start_joints"] = list(q)
            return st

        D.arm_state = with_start
        try:
            if hasattr(T, "arm_state"):
                T.arm_state, saved_t = with_start, T.arm_state
            self.t.to_pose(T.LOOK_TIP, "look")
            self._note(ok=True, what="unfold", why="folded rest to the look pose")
        finally:
            D.arm_state = real
            if hasattr(T, "arm_state"):
                T.arm_state = saved_t

    # ---- the interface
    def prepare(self, scene=None):
        """Looks and the whole plan; no block is touched (the runner's first pick or push also waits on GATE_FILE)."""
        ab = getattr(O, "ABORT", None)
        if ab is not None:
            ab.clear()
        off = self._arm_off_result("prepare", _wall.time())
        if off is not None:
            return off
        try:
            self._ensure_runner()
            t = self.t
            held = self.set_down_held()  # a block in the jaws at the start (a stop mid-lift): set it down before anything else
            if held is not None and not held.get("ok"):
                return self._note(ok=False, what="prepare", why=held["why"])
            self._unstick_start()
            self._unfold_from_rest()
            # a re-look renames the blocks: the tower is found again by where it stands (this round's placement records,
            # or a standing stack), never by the old names (Nov 8 13:59: "blue#1" became another block 30 cm away)
            self.tower, self.moves = [], []
            t.multi_look(None)
            self._tower_from_records(t)
            T.do_scatter_setup(t)
            t.tower_ids = set(sum(t.scatter_list, []))
            if AI_PLANNER:
                n = 0
                stacks = self._stacks()
                if stacks and not self.tower:  # a tower already standing: build on it, never pick from it
                    self.tower = list(stacks[0])
                    self._note(ok=True, what="adopt_tower", tower=list(self.tower))
            else:
                n = self._plan()
            self.prepared = True
            return self._note(ok=True, what="prepare", planned_moves=n, spot=self.spot, blocks=sorted(t.belief))
        except BaseException as e:  # noqa: BLE001
            return self._fail("prepare", e)

    def _tower_from_records(self, t):
        """After a look, the blocks this round placed come back from the placement records at their exact spots: the
        looks read a tall tower's upper blocks as ghosts about 10 cm off at table level. Sightings within 4 cm of a
        recorded column are dropped (the records win); same-colour sightings within 13 cm of a column taller than one
        block are dropped as ghosts (losing a real loose block there is the safe side)."""
        try:
            known = T.world_from_records()
        except Exception:  # noqa: BLE001
            return
        up = {k: b for k, b in known.items() if T.bot(b) > T.TABLE_Z + 0.005}
        if not up:
            return
        under = {k: b for k, b in known.items() if k not in up and any(math.hypot(b["x"] - o["x"], b["y"] - o["y"]) < 0.03 for o in up.values())}
        tower = {**under, **up}
        cols = []
        for b in tower.values():
            for c in cols:
                if math.hypot(b["x"] - c[0], b["y"] - c[1]) < 0.03:
                    c[2].add(T.colour_of(next(k for k, v in tower.items() if v is b)))
                    break
            else:
                cols.append([b["x"], b["y"], {T.colour_of(next(k for k, v in tower.items() if v is b))}])
        dropped = []
        for k, b in list(t.belief.items()):
            for cx, cy, colours in cols:
                d = math.hypot(b["x"] - cx, b["y"] - cy)
                if d < 0.04 or (d < 0.13 and T.colour_of(k) in colours):
                    dropped.append(k)
                    t.belief.pop(k)
                    break
        for i, (k, b) in enumerate(sorted(tower.items(), key=lambda kv: T.bot(kv[1]))):
            t.belief[f"{T.colour_of(k)}#t{i + 1}"] = dict(b)
        t.put_blocks()
        self._note(ok=True, what="tower_from_records", tower=[f"{T.colour_of(k)}#t{i + 1}" for i, k in enumerate(sorted(tower, key=lambda kk: T.bot(tower[kk])))],
                   dropped=dropped)

    def set_down_held(self, dry=False):
        """A block held at the start (grip_holding, or grip and goal_grip both under 0.75): the runner's own set-back
        (night_setback): straight down at a table spot with the place's legality checks, open, back off 6 cm up, then
        the look pose. The spot: straight below the hand when the records show it clear (a stop mid-lift is over the
        block's own source), else the nearest free spot. None when nothing is held; else a dict (ok False: no legal
        set-down). dry=True plans only and moves nothing."""
        import numpy as np
        st = O.arm_state()
        if not (st.get("grip_holding") or (st["grip"] < 0.75 and st["goal_grip"] < 0.75)):
            return None
        torque = st.get("grip_measured_torque_nm")
        if not st.get("grip_holding") and torque is not None and abs(float(torque)) < 0.3:
            # closed jaws but the grip force sensor feels nothing (Nov 8 14:18: jaws at 0.39, empty, round refused):
            # open the jaws where they are, nothing else moves
            if not dry:
                req = urllib.request.Request(O.URL + "/move", json.dumps({"q": [float(v) for v in st["q"]], "grip": 0.85, "rate": 0.3}).encode(),
                                             {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=3).read()
                _wall.sleep(1.0)
            self._note(ok=True, what="set_down_held", why=f"jaws closed but empty (grip torque {float(torque):.2f} N m): opened in place")
            return None
        t = self.t
        bid = "held#1"
        q0 = np.array(st["q"]) + t.op.dq
        t.op.st["target"] = list(q0)
        tip, yaw_deg = t.op.tip(q0)
        known = T.world_from_records()  # everything placed and standing, as obstacles
        t.belief = {k: dict(b) for k, b in known.items()}
        gh = O.GRIP_TIP - O.TABLE_Z  # the normal grasp: fingertips 15 mm over the block's bottom
        t.belief[bid] = {"x": float(tip[0]), "y": float(tip[1]), "yaw": math.radians(yaw_deg), "level": 0, "z0": O.TABLE_Z,
                         "lying": False, "iou": 1.0, "edge": False, "t_seen": _wall.time()}
        t.op.held = (bid, gh)
        t.held_ext = D.ext(t.belief[bid])
        R = t.op.kin.fk(q0)[1]
        others = [(b["x"], b["y"]) for k, b in t.belief.items() if k != bid]
        rings = [(float(tip[0] + r * math.cos(a)), float(tip[1] + r * math.sin(a))) for r in (0.03, 0.06, 0.09, 0.12, 0.15)
                 for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
        cands = [(float(tip[0]), float(tip[1]))] + [c for c in rings if 0.17 <= c[0] <= 0.40 and abs(c[1]) <= 0.145]
        tried, pl, spot = [], None, None
        for x, y in cands:
            if any(math.hypot(x - ox, y - oy) < 0.07 for ox, oy in others):
                continue
            t.belief[bid].update(x=x, y=y)
            pl, _k, why = t.legal_place(bid, q0, R, (x, y, math.radians(yaw_deg), O.TABLE_Z, None), gh)
            tried.append(f"({x * 100:.1f}, {y * 100:.1f}): {'ok' if pl is not None else str(why)[:60]}")
            if pl is not None:
                spot = (x, y)
                break
        if pl is None:
            t.op.held = None
            return self._note(ok=False, what="set_down_held", why="holding a block at start: no legal set-down", tried=tried[:8])
        if dry:
            t.op.held = None
            return self._note(ok=True, what="set_down_held", dry=True, spot=[round(spot[0] * 100, 1), round(spot[1] * 100, 1)],
                              nodes=len(pl.nodes), tip_z_mm=round(float(t.op.tip(pl.nodes[-1][0])[0][2]) * 1000), tried=tried[:8])
        t.go(pl, D.GRIP_CLOSE, "set back", "down")
        t.release(O.arm_state())
        t.op.held = None
        t.held_ext = 0.0
        back = O.Plan(t.op, np.array(t.op.st["target"]), 0.08)
        back.R = pl.R
        back.to(t.op.tip(pl.nodes[-1][0])[0] + [0, 0, 0.06], yaw_deg, 0.3, "back off", line=0.01, keep=True)
        back.check(skip=bid)
        t.go(back, 0.8, "back off", "up")
        t.to_pose(O.LOOK_TIP, "look")
        t.belief = {}  # the looks that follow see the table afresh
        return self._note(ok=True, what="set_down_held", spot=[round(spot[0] * 100, 1), round(spot[1] * 100, 1)])

    def place_next(self, scene=None, deadline_s=120.0):
        """One more block on the tower: the planned clearing moves, then the next stack place. dict, never raises."""
        T0 = _wall.time()
        gate = os.environ.get("GATE_FILE")
        if gate and not os.path.exists(gate):  # never touch a block before the USB gate (and never wait on it here)
            return {"ok": False, "grew": False, "verified": False, "placed_id": None, "why": "USB gate shut", "tower": list(self.tower), "timing": {"s": 0.0}}
        ab = getattr(O, "ABORT", None)
        if ab is not None:
            ab.clear()
        off = self._arm_off_result("place_next", T0)
        if off is not None:
            return off
        try:
            if not self.prepared:
                r = self.prepare(scene)
                if not r.get("ok"):
                    return {"ok": False, "grew": False, "verified": False, "placed_id": None, "why": r.get("why"), "timing": {"s": round(_wall.time() - T0, 1)}}
            if len(self.tower) >= TILT_LEVEL and self.helper_spots:
                return self._place_tilted(T0)
            if len(self.tower) >= int(os.environ.get("TOP_TILT_FROM", "4")):
                # past fingers-down's reach over the top: the remote metal arm's night_top recipe (6 high REAL, Nov 8
                # 11:50 and 12:01): a 20 deg tilted hand, 9 mm sag, one block at a time
                r = self._place_top_tilted(T0)
                if r.get("ok") or r.get("raised"):
                    return r
            if AI_PLANNER:
                return self._place_next_ai(T0, deadline_s)
            t = self.t
            while _wall.time() - T0 < deadline_s:
                if not self.moves:
                    if self.replans >= 4:
                        return self._note(ok=False, grew=False, verified=False, what="place_next", placed_id=None, why="nothing left that plans",
                                          tower=list(self.tower), timing={"s": round(_wall.time() - T0, 1)})
                    self.replans += 1
                    self._plan(budget=15.0)
                    if not self.moves:
                        return self._note(ok=False, grew=False, verified=False, what="place_next", placed_id=None, why="no moves planned",
                                          tower=list(self.tower), timing={"s": round(_wall.time() - T0, 1)})
                kind, bid, tgt = self.moves.pop(0)
                try:
                    if kind == "push":
                        s, e, inline = tgt
                        objs = t.objects()
                        ex = {p["block"] for p in objs if math.hypot(p["x"] - s[0], p["y"] - s[1]) < 0.15}
                        t.push(T.np.asarray(s), T.np.asarray(e), ex, inline, f" (planned, {bid})")
                        continue
                    if kind == "clear":
                        t.block(bid, tgt, 1, 0, "clear")
                        continue
                    if self.tower:
                        s_ = t.belief[self.tower[-1]]
                        tgt = (s_["x"], s_["y"], s_["yaw"], len(self.tower), self.tower[-1])
                    t.block(bid, tgt, 1, len(self.tower) + 1, "stack")
                    self.tower.append(bid)
                    # block() returns only after the runner's own after-place and standing checks passed (a place that
                    # is off or not standing is re-placed or raises), so a normal return is a verified place
                    return self._note(ok=True, grew=True, verified=True, what="place_next", placed_id=bid, why="",
                                      tower=list(self.tower), timing={"s": round(_wall.time() - T0, 1)})
                except (D.NoPlan, D.NoSpot, D.CycleFail, SystemExit) as e:
                    st = O.arm_state()
                    if st["grip"] < 0.75 and st["goal_grip"] < 0.75:
                        return self._note(ok=False, grew=False, verified=False, what="place_next", placed_id=None, tower=list(self.tower),
                                          why=f"holding a block after a failed {kind} {bid}: {str(e)[:200]}",
                                          timing={"s": round(_wall.time() - T0, 1)})
                    t.op.st["target"] = None
                    self._note(ok=False, what="move", kind=kind, block=bid, why=str(e)[:300])
                    self.moves = []  # re-plan from the belief as it is now
            return self._note(ok=False, grew=False, verified=False, what="place_next", placed_id=None, why="deadline", tower=list(self.tower),
                              timing={"s": round(_wall.time() - T0, 1)})
        except BaseException as e:  # noqa: BLE001
            r = self._fail("place_next", e)
            return {"ok": False, "grew": False, "verified": False, "placed_id": None, "why": r["why"], "tower": list(self.tower), "timing": {"s": round(_wall.time() - T0, 1)}}

    AI_SPOTS = ((0.214, 0.180), (0.190, 0.200), (0.235, 0.140), (0.250, 0.090), (0.260, 0.040), (0.280, -0.010), (0.240, -0.060))

    def _overhead_jpeg(self):
        for base in ("http://127.0.0.1:8691", "http://127.0.0.1:8692"):
            try:
                with urllib.request.urlopen(base + "/overhead.jpg", timeout=1.5) as r:
                    data = r.read()
                if data[:2] == b"\xff\xd8":
                    return data
            except Exception:  # noqa: BLE001
                continue
        return None

    def _ai_choose(self, cands, spots):
        """Terra picks the next block (and the tower spot on the first block). Returns its JSON or None."""
        from hillclimb.common import vision as V
        t = self.t
        obst = [(round(float(o[0]) * 100, 1), round(float(o[1]) * 100, 1)) for o in (t.obstacles if len(getattr(t, "obstacles", [])) else [])][:30]
        balls = [(round(x * 100, 1), round(y * 100, 1)) for x, y in getattr(t, "balls", [])]
        rows = []
        for k in cands:
            b = t.belief[k]
            near = min([math.hypot(b["x"] * 100 - ox, b["y"] * 100 - oy) for ox, oy in obst + balls] or [99.0])
            rows.append({"id": k, "x_cm": round(b["x"] * 100, 1), "y_cm": round(b["y"] * 100, 1), "reach_cm": round(math.hypot(b["x"], b["y"]) * 100, 1),
                         "fit": round(float(b.get("iou", 0)), 2), "nearest_ball_or_stick_cm": round(near, 1)})
        tower = None
        if self.tower:
            tb = t.belief.get(self.tower[-1], {})
            tower = {"blocks": len(self.tower), "x_cm": round(tb.get("x", 0) * 100, 1), "y_cm": round(tb.get("y", 0) * 100, 1)}
        prompt = (
            "You plan a robot arm stacking foam cubes into ONE tall tower, one move at a time. Coordinates: arm frame in cm, x forward "
            "from the left arm's base, +y to the LEFT. The arm reaches about 46 cm at table height; closer is easier.\n"
            "RULES: never choose a block that is really a ball, stick or cable (balls of the same colour are common: a candidate "
            "touching or within 5 cm of a ball, or with a weak fit under 0.6 that looks round in the image, must be avoided). Touching a "
            "ball is an instant loss. Prefer blocks that are easy: close, clear of neighbours, good fit.\n"
            f"Tower now: {json.dumps(tower)}.\n"
            + ("" if self.tower else "Choose the tower spot too, from SPOTS (index): prefer one to the left and out of the way of the loose blocks, "
               "at least 6 cm from every block, ball and stick.\nSPOTS: " + json.dumps([[round(x * 100, 1), round(y * 100, 1)] for x, y in spots]) + "\n")
            + f"CANDIDATE blocks: {json.dumps(rows)}\nBalls/sticks seen (cm): {json.dumps(balls + obst)}\n"
            + f"Already refused this round (do not pick again): {json.dumps(sorted(self.avoid))}\n"
            + (f"Recent refusals: {json.dumps(self.refusals[-5:])}\n" if self.refusals else "")
            + "The image is the overhead camera. Answer only JSON: {\"block\": id or null, \"spot\": index or null, "
              "\"avoid\": [ids you believe are really a ball, stick or phantom, ONLY those; reach and planning are checked by the planner, not you], \"why\": short}")
        ans, ms = V.call(prompt, [self._overhead_jpeg()], timeout=12.0)
        D.status(f"ai planner ({ms / 1000:.1f} s): {json.dumps(ans)[:300] if ans else 'no answer'}")
        return ans

    def _simple_choose(self, cands):
        """Blocks further left (+y) or further out (+x) than the base first, farthest first; then the rest, farthest first."""
        t = self.t
        b0 = t.belief[self.tower[0]] if self.tower and self.tower[0] in t.belief else {"x": 0.25, "y": 0.0}
        d = lambda k: math.hypot(t.belief[k]["x"] - b0["x"], t.belief[k]["y"] - b0["y"])
        first = [k for k in cands if t.belief[k]["y"] > b0["y"] + 0.02 or t.belief[k]["x"] > b0["x"] + 0.02]
        rest = [k for k in cands if k not in first]
        order = sorted(first, key=d, reverse=True) + sorted(rest, key=d, reverse=True)
        D.status(f"simple order: {order}")
        return {"block": order[0] if order else None, "spot": 0, "avoid": []}

    def _place_next_ai(self, T0, deadline_s):
        """One move chosen by the AI, checked by the planner (that move only), then done. A refused or failed move is
        reported back to the AI, which picks again; no re-look and no whole-sequence re-plan."""
        t = self.t
        self.skip_now = set()  # refused for this placement only; back in the pool for the next one
        for attempt in range(6):
            if _wall.time() - T0 > deadline_s:
                break
            cands = [k for k, b in t.belief.items() if k not in self.tower and k not in self.avoid and k not in self.skip_now and not k.startswith("pile")
                     and not b.get("lying") and abs(D.bot(b) - T.TABLE_Z) < 0.008 and b.get("iou", 0) >= 0.45 and b["x"] >= 0.08
                     and math.hypot(b["x"], b["y"]) <= 0.55]
            if not cands:
                return self._note(ok=False, grew=False, verified=False, what="place_next", placed_id=None, why="nothing left that plans",
                                  tower=list(self.tower), timing={"s": round(_wall.time() - T0, 1)})
            if not self.tower and os.environ.get("PLANNER", "simple") == "simple":
                # Isaiah's simple order (Nov 8 15:16): the orange block nearest the arm is the base where it lies
                oranges = [k for k in cands if T.colour_of(k) == "orange"]
                if oranges:
                    base = min(oranges, key=lambda k: math.hypot(t.belief[k]["x"], t.belief[k]["y"]))
                    self.tower = [base]
                    D.status(f"simple: {base} at ({t.belief[base]['x'] * 100:.1f}, {t.belief[base]['y'] * 100:.1f}) is the tower base where it lies")
                    continue
            if not self.tower:
                # the first motion stacks: a block already lying clear (6 cm from every other block, ball and stick) is
                # the base where it is, the leftmost such block in the easy band (Isaiah, Nov 8 15:08)
                obst = [(float(o[0]), float(o[1])) for o in (t.obstacles if len(getattr(t, "obstacles", [])) else [])] + list(getattr(t, "balls", []))
                bases = [k for k in cands if 0.17 <= t.belief[k]["x"] <= 0.32 and abs(t.belief[k]["y"]) <= 0.22
                         and all(math.hypot(t.belief[k]["x"] - b["x"], t.belief[k]["y"] - b["y"]) >= 0.06 for o, b in t.belief.items() if o != k)
                         and all(math.hypot(t.belief[k]["x"] - ox, t.belief[k]["y"] - oy) >= 0.06 for ox, oy in obst)]
                if bases:
                    base = max(bases, key=lambda k: t.belief[k]["y"])
                    self.tower = [base]
                    D.status(f"ai: {base} at ({t.belief[base]['x'] * 100:.1f}, {t.belief[base]['y'] * 100:.1f}) is the tower base where it lies")
                    continue
            spots = []
            if not self.tower:
                for x, y in self.AI_SPOTS:
                    if all(math.hypot(x - b["x"], y - b["y"]) >= 0.06 for b in t.belief.values()):
                        spots.append((x, y))
                spots = spots or list(self.AI_SPOTS)
            if os.environ.get("PLANNER", "simple") == "simple":
                ans = self._simple_choose(cands)
            else:
                ans = self._ai_choose(cands, spots) or {}
            for a in ans.get("avoid") or []:
                if a in t.belief:
                    self.avoid.add(a)
            bid = ans.get("block")
            if bid not in cands or bid in self.avoid:
                bid = min(cands, key=lambda k: math.hypot(t.belief[k]["x"] - 0.2, t.belief[k]["y"]))  # fallback: the closest
            if self.tower:
                s_ = t.belief[self.tower[-1]]
                tgt = (s_["x"], s_["y"], s_["yaw"], len(self.tower), self.tower[-1])
            else:
                i = ans.get("spot")
                sx, sy = spots[i] if isinstance(i, int) and 0 <= i < len(spots) else spots[0]
                tgt = (sx, sy, math.atan2(sy, sx) if abs(math.atan2(sy, sx)) > math.radians(30) else 0.0, 0, None)
            t1 = _wall.time()
            ok = t.feasible(bid, tgt)
            D.status(f"ai move {bid} -> ({tgt[0] * 100:.1f}, {tgt[1] * 100:.1f}) level {len(self.tower) + 1}: plan check {'ok' if ok else 'REFUSED: ' + str(getattr(t, 'last_refusal', ''))[:220]} in {_wall.time() - t1:.1f} s")
            if not ok and self.tower and ("no safe pose" in str(getattr(t, "last_refusal", "")) or "no rule route" in str(getattr(t, "last_refusal", ""))):
                # out of fingers-down reach (about 46 cm at the table): the tilted hand reaches 55-60 cm (night_top, the
                # remote metal arm's 6-high recipe), onto the tower top
                r = self._place_top_tilted(T0, only=[bid])
                if r.get("ok"):
                    return r
                D.status(f"ai move {bid}: tilted too: {str(r.get('why'))[:160]}")
            if not ok:
                self.skip_now.add(bid)
                self.refusals.append(f"{bid}: its move does not plan ({getattr(t, 'last_refusal', '')[:110]})")
                continue
            try:
                t.block(bid, tgt, 1, len(self.tower) + 1, "stack")
            except (D.NoPlan, D.NoSpot, D.CycleFail, SystemExit) as e:
                st = O.arm_state()
                (self.avoid if "ball" in str(e) else self.skip_now).add(bid)
                self.refusals.append(f"{bid}: {str(e)[:120]}")
                if st["grip"] < 0.75 and st["goal_grip"] < 0.75 and st.get("grip_holding"):
                    return self._note(ok=False, grew=False, verified=False, what="place_next", placed_id=None, tower=list(self.tower),
                                      why=f"holding a block after a failed move {bid}: {str(e)[:200]}", timing={"s": round(_wall.time() - T0, 1)})
                t.op.st["target"] = None
                continue
            self.tower.append(bid)
            return self._note(ok=True, grew=True, verified=True, what="place_next", placed_id=bid, why="ai move",
                              tower=list(self.tower), timing={"s": round(_wall.time() - T0, 1)})
        return self._note(ok=False, grew=False, verified=False, what="place_next", placed_id=None, why="no moves planned",
                          tower=list(self.tower), timing={"s": round(_wall.time() - T0, 1)})

    def _place_top_tilted(self, T0, only=None):
        """One loose block onto the tower top with the hand tilted (calib/night_top.plan_top): tilted pick, carry level
        over the top, set down, slide out, look pose. Tries the loose blocks nearest the tower first until one plans."""
        import numpy as np
        from hillclimb.runner.calib import keepout as KO
        from hillclimb.runner.calib import night_top as NT
        t = self.t
        theta = math.radians(float(os.environ.get("TOP_TILT_DEG", "20")))
        sag = float(os.environ.get("TOP_SAG_MM", "9")) / 1000
        if not self.tower or self.tower[-1] not in t.belief:
            return {"ok": False, "why": "tilted top: no tower top in the belief"}
        s_b = dict(t.belief[self.tower[-1]])
        top = D.top_of(s_b)
        tx0, ty0 = s_b["x"], s_b["y"]
        loose = [k for k, b in t.belief.items() if k not in self.tower and not b.get("lying") and abs(D.bot(b) - T.TABLE_Z) < 0.008
                 and b.get("iou", 0) >= 0.5 and not k.startswith("pile")
                 and math.hypot(b["x"] - tx0, b["y"] - ty0) > 0.06]
        loose.sort(key=lambda k: math.hypot(t.belief[k]["x"] - tx0, t.belief[k]["y"] - ty0))
        if only is not None:
            loose = [k for k in only if k in t.belief and k not in self.tower]
        whys = []
        for bid in loose:
            try:
                dpsi, pick, carry, out, gh, z_set, tx, ty = NT.plan_top(t, bid, s_b, top, theta, sag, quiet=True)
            except SystemExit as e:
                whys.append(f"{bid}: {str(e)[:70]}")
                continue
            try:  # the close-up ball check before every pick, tilted too (never touch a ball)
                t.to_pose(O.LOOK_TIP, "look")
                t.ball_check(bid)
            except (D.CycleFail, SystemExit) as e:
                whys.append(f"{bid}: {str(e)[:70]}")
                if "ball" in str(e):
                    self.avoid.add(bid)
                continue
            D.status(f"tilted top: {bid} onto the tower top ({(top - T.TABLE_Z) * 1000:.0f} mm), {math.degrees(theta):.0f} deg tilt")
            op = t.op
            try:
                t.go(pick, 0.8, "top pick", "down")
                t.grip_to(O.arm_state(), D.GRIP_CLOSE, wait_close=True)
                st = O.arm_state()
                if st["grip"] < D.GRIP_HELD:
                    t.grip_to(st, 0.8)
                    return {"ok": False, "grew": False, "verified": False, "placed_id": None, "why": f"tilted top: {bid} grasp closed empty",
                            "tower": list(self.tower), "timing": {"s": round(_wall.time() - T0, 1)}}
                op.held = (bid, gh)
                KO.ACTIVE[:] = [(tx, ty)]
                t.go(carry, D.GRIP_CLOSE, "top carry", "down")
                t.release(O.arm_state())
                op.held = None
                tip_rel = op.tip(np.array(O.arm_state()["q"]) + op.dq)[0]
                KO.add_place(float(tip_rel[0]), float(tip_rel[1]), top, T.H, T.TABLE_Z, note=f"{bid} tilted top place")
                t.go(out, 0.8, "slide out", "level")
            finally:
                KO.ACTIVE.clear()
                op.held = None
            b = dict(t.belief.pop(bid))
            b.update(x=tx, y=ty, z0=top, level=len(self.tower))
            nid = f"{T.colour_of(bid)}#t{len(self.tower) + 1}"
            t.belief[nid] = b
            self.tower.append(nid)
            t.put_blocks()
            try:
                t.to_pose(O.LOOK_TIP, "look")
            except SystemExit as e:
                D.status(f"tilted top: look pose after the place refused ({str(e)[:80]})")
            return self._note(ok=True, grew=True, verified=True, what="place_next", placed_id=nid, why="tilted top place",
                              tower=list(self.tower), timing={"s": round(_wall.time() - T0, 1)})
        return {"ok": False, "why": "tilted top: no loose block plans: " + "; ".join(whys)[:300]}

    def _place_tilted(self, T0):
        """Levels 8 and 9: the next staged helper's top block, tilted onto the tower top (calib/tilt_helper.py)."""
        from hillclimb.runner.calib import tilt_helper as TH
        t = self.t
        theta, up, sag = self.tilt
        top = t.belief[self.tower[-1]]
        for k, (sx, sy) in enumerate(self.helper_spots):
            if k in self.used_helpers:
                continue
            stack = TH.column(t.belief, sx, sy, r=0.035)
            if not stack:
                continue
            bid = stack[-1]
            res = TH.tilt_place(t, bid, D.top_of(top), top["x"], top["y"], theta, up, sag, top["yaw"])
            self.used_helpers.add(k)
            self.tower.append(bid)
            return self._note(ok=True, grew=True, verified=False, what="place_next", placed_id=bid, why="tilted",
                              tower=list(self.tower), tilt=res,
                              timing={"s": round(_wall.time() - T0, 1)})
        return self._note(ok=False, grew=False, verified=False, what="place_next", placed_id=None, why="no staged helper with a block on top",
                          tower=list(self.tower), timing={"s": round(_wall.time() - T0, 1)})

    def recover(self, scene=None, deadline_s=180.0):
        """Every piled or touching block apart on the table (a tumbled tower); the tower is then re-planned."""
        T0 = _wall.time()
        gate = os.environ.get("GATE_FILE")
        if gate and not os.path.exists(gate):  # a recover touches blocks: never before the USB gate
            return {"ok": False, "why": "USB gate shut", "timing": {"s": 0.0}}
        ab = getattr(O, "ABORT", None)
        if ab is not None:
            ab.clear()
        off = self._arm_off_result("recover", T0)
        if off is not None:
            return off
        try:
            self._ensure_runner()
            self.t.recover()
            self.tower, self.moves, self.prepared = [], [], False
            return self._note(ok=True, what="recover", timing={"s": round(_wall.time() - T0, 1)})
        except BaseException as e:  # noqa: BLE001
            r = self._fail("recover", e)
            return {"ok": False, "why": r["why"], "timing": {"s": round(_wall.time() - T0, 1)}}

    def retreat(self):
        """To the look pose under the runner's home rules (clear of every stack it knows)."""
        try:
            self._ensure_runner()
            self.t.op.st["target"] = None
            self.t.to_pose(O.LOOK_TIP, "retreat")
            return self._note(ok=True, what="retreat")
        except BaseException as e:  # noqa: BLE001
            return self._fail("retreat", e)

    def hold(self):
        """Stop where the arm is: the streaming loop aborts before its next goal (O.ABORT), then the arm server's own stop."""
        ab = getattr(O, "ABORT", None)
        if ab is not None:
            ab.set()
        try:
            req = urllib.request.Request(O.URL + "/stop", b"{}", {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
            return self._note(ok=True, what="hold")
        except BaseException as e:  # noqa: BLE001
            return self._fail("hold", e)

    def close(self, clean=True):
        try:
            if self.t is not None:
                T.finish(self.t, clean)
        except BaseException as e:  # noqa: BLE001
            self._fail("close", e)
