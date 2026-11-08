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

    def _plan(self, budget=40.0):
        t = self.t
        left = [k for k, b in t.belief.items() if k not in self.tower and not k.startswith("pile")
                and b.get("iou", 0) >= 0.5 and b["x"] >= 0.15]
        order = sorted(left, key=lambda k: (bool(t.belief[k].get("lying")), -T.bot(t.belief[k])))
        t.dr = {"carry_z": 0.15, "over_dz": 0.01}
        t.spot_area = ((0.19, 0.29), (-0.12, 0.15))
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
        if len(q) < 4 or max(abs(a - b) for a, b in zip(q[1:4], self.FOLDED_REST[1:4])) > 0.15:
            return
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
        try:
            self._ensure_runner()
            t = self.t
            self._unfold_from_rest()
            t.multi_look(None)
            T.do_scatter_setup(t)
            t.tower_ids = set(sum(t.scatter_list, []))
            n = self._plan()
            self.prepared = True
            return self._note(ok=True, what="prepare", planned_moves=n, spot=self.spot, blocks=sorted(t.belief))
        except BaseException as e:  # noqa: BLE001
            return self._fail("prepare", e)

    def place_next(self, scene=None, deadline_s=120.0):
        """One more block on the tower: the planned clearing moves, then the next stack place. dict, never raises."""
        T0 = _wall.time()
        gate = os.environ.get("GATE_FILE")
        if gate and not os.path.exists(gate):  # never touch a block before the USB gate (and never wait on it here)
            return {"ok": False, "grew": False, "verified": False, "placed_id": None, "why": "USB gate shut", "tower": list(self.tower), "timing": {"s": 0.0}}
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
        """Stop where the arm is (the arm server's own stop: it holds the current position)."""
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
