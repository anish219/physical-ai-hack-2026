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
        # the tower area, biased left (+y, image-left on the overhead): the blocks refused on Nov 8 sat at y < -15 cm,
        # toward the strip, and a base there put the tower in their routes. TOWER_Y_MIN (m) overrides the left bias.
        t.spot_area = ((0.19, 0.29), (float(os.environ.get("TOWER_Y_MIN", "-0.03")), 0.15))
        res = t.best_sequence(order, tuple(self.tower), budget=budget)
        if res is None:
            raise D.Fatal(f"no tower spot plans: {dict(t.why)}")
        self.spot, base, moves, _stop = res
        if not self.tower:
            self.tower = list(base)
        self.moves = list(moves)
        return len(self.moves)

    FOLDED_REST = (0.0, -1.89, 1.73, -1.47, 0.0, 0.0)  # the left arm's power-on rest (Nov 7-8 logs), joints in rad

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
            self._unfold_from_rest()
            # a re-look renames the blocks: the tower is found again by where it stands (this round's placement records,
            # or a standing stack), never by the old names (Nov 8 13:59: "blue#1" became another block 30 cm away)
            self.tower, self.moves = [], []
            t.multi_look(None)
            self._tower_from_records(t)
            T.do_scatter_setup(t)
            t.tower_ids = set(sum(t.scatter_list, []))
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
