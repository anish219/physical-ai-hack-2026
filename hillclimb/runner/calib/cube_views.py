"""Wrist-camera views of the BIG70 tag cube from 12-16 arm poses, for the joint fit (calib/fit_cube.py). No contact.

Through the team's left arm server (127.0.0.1:8691), only /move and /stop. Before every move: the team page
(127.0.0.1:18081) must show assigned_side not left and recording false; the Dream caller (127.0.0.1:18084) must not
run the left arm (the right arm may be busy: every path stays on the left, fingertips y > -10 cm, every arm point 25 cm
from the right arm's base); the server's goal must be the last one we sent. Any of those fails: /stop, back
to the start joints slowly, exit.

Poses: the fingertips 15-25 cm from the cube's centre, the fingers pointing at it (the wrist camera looks along them),
from several sides and tilts, the jaws turned differently from pose to pose; fingertips at least 6 cm over the table,
y above -15 cm, every arm point at least 3 cm over the table and 6 cm from the cube's centre along each path (sim model).
At each pose: hold 1 s, save both frames with the joints the server stamped on them.

    python -m calib.cube_views [--dry]
"""

import json
import math
import pathlib
import sys
import time
import urllib.request

import mujoco
import numpy as np

from hillclimb.runner.calib.fit_overhead import arm_points
from hillclimb.runner.calib.safety import check_path, check_pose, _mount
from hillclimb.runner.calib.touch_plan import tip_of
from hillclimb.runner.sim.grasp_check import reach
from hillclimb.runner.sim.physics import Scene

ROOT = pathlib.Path(__file__).resolve().parent.parent
_RUNS_ROOT = pathlib.Path(__import__("os").environ.get("HILLCLIMB_RUNS", str(pathlib.Path.home() / "hillclimb_runs")))
URL = "http://127.0.0.1:8691"
TEAM, CALLER = "http://127.0.0.1:18081/api/state", "http://127.0.0.1:18084/api/state"
RATE = 0.25
RUN_FIT = pathlib.Path(__file__).resolve().parent.parent / "runs" / "station" / "calib4" / "fit" / "fit_freepos.json"
RIGHT_BASE = np.array([0.0, -0.69])  # the right arm's base in the left arm's frame: keep every point 25 cm away
OUT = _RUNS_ROOT / "station" / "calib4"


def get(url, timeout=5):
    return urllib.request.urlopen(url, timeout=timeout).read()


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5).read()


class Abort(Exception):
    pass


class Guard:
    def __init__(self):
        self.goal = None

    def check(self, full=True):
        s = json.loads(get(URL + "/state"))
        if s.get("error"):
            raise Abort(f"arm server error: {s['error']}")
        if self.goal is not None and np.max(np.abs(np.array(s["goal"]) - self.goal)) > 1e-4:
            raise Abort("the left server's goal changed under us: someone else is driving the left arm")
        if full:  # the left arm taken by anyone else: stop. The right arm busy is fine (left-side paths only).
            team = json.loads(get(TEAM))
            if team.get("assigned_side") in ("left", "both") or team.get("recording"):
                raise Abort(f"team page: assigned {team.get('assigned_side')}, recording {team.get('recording')}")
            caller = json.loads(get(CALLER))
            if caller["left"].get("running") or caller["both"].get("running"):
                raise Abort("Dream caller is running the left arm")
        return s

    def move(self, q, rate=RATE):
        s0 = self.check()
        ok, why = check_path(s0["goal"], q, recover=True)  # calib/safety.py (the server steps from its command, not the sagging reading): camera upright, camera box clear, wrist limits
        if not ok:
            raise Abort(f"safety refused the move: {why}")
        post("/move", {"q": [float(v) for v in q], "grip": 0.0, "rate": min(rate, 0.3)})
        self.goal = np.array(q, float)
        t0 = time.time()
        time.sleep(0.2)
        while time.time() - t0 < 60:
            s = self.check(full=False)
            if not s["moving"]:
                return s
            time.sleep(0.15)
        raise Abort("move did not finish")


def ik_point(s, tip_target, look_at, q0, roll_dir=None, iters=800):
    """Fingertips at tip_target with the fingers (tool z) pointing at look_at; optionally the jaw travel (tool x)
    pushed toward roll_dir (projected), else free."""
    m, d = s.model, s.data
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
    dofs = s.arm_dadr["right"]
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"right_{n}") for n in s.joint_names]
    lo, hi = m.jnt_range[jids, 0] + 0.03, m.jnt_range[jids, 1] - 0.03
    tips = reach(s, "metal")
    q = np.array(q0, float)
    want = np.asarray(look_at) - np.asarray(tip_target)
    want /= np.linalg.norm(want)
    target_tcp = np.asarray(tip_target) - want * tips
    for _ in range(iters):
        s.set_arm("right", q, 1.0, snap=True)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        R = d.site_xmat[sid].reshape(3, 3)
        e_pos = target_tcp - d.site_xpos[sid]
        e_rot = np.cross(R[:, 2], want)
        if roll_dir is not None:
            rd = np.asarray(roll_dir, float) - want * (want @ roll_dir)
            e_rot = e_rot + 0.5 * np.cross(R[:, 0], rd / np.linalg.norm(rd))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, dofs], 0.3 * jr[:, dofs]])
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([e_pos, 0.3 * e_rot])), lo, hi)
    s.set_arm("right", q, 1.0, snap=True)
    mujoco.mj_kinematics(m, d)
    R = d.site_xmat[sid].reshape(3, 3)
    return q, float(np.linalg.norm(target_tcp - d.site_xpos[sid])), math.degrees(math.acos(np.clip(R[:, 2] @ want, -1, 1)))


def path_ok(s, q0, q1, cube):
    """Every arm point over 3 cm above the table and 6 cm from the cube's centre, fingertips 6 cm up, along the path."""
    tips = reach(s, "metal")
    worst = (1.0, 1.0, 1.0, 1.0, 1.0)
    for t in np.linspace(0, 1, 16):
        q = q0 + (q1 - q0) * t
        p = arm_points(s, q, per_mesh=60)
        tip = tip_of(s, q, tips)
        worst = (min(worst[0], p[:, 2].min()), min(worst[1], np.linalg.norm(p - cube, axis=1).min()), min(worst[2], tip[2]),
                 min(worst[3], tip[1]), min(worst[4], np.linalg.norm(p[:, :2] - RIGHT_BASE, axis=1).min()))
    return (worst[0] > 0.03 - 0.008 and worst[1] > 0.06 and worst[2] > 0.06 - 0.008 and worst[3] > -0.10 and worst[4] > 0.25), worst


def plan(cube, q_start):
    s = Scene("metal", "right", props=[], draw=True)
    tips = reach(s, "metal")
    cands = []
    seeds = [np.array(q_start, float), np.array([0.6, -0.9, 1.4, -0.6, 0, 0]), np.array([0.9, -0.6, 1.0, -0.4, 0, 0]),
             np.array([0.3, -1.2, 1.8, -0.8, 0, 0])]
    rng = np.random.default_rng(1)
    base_az = math.atan2(-cube[1], -cube[0])  # from the cube toward the arm's base
    for az in np.radians(np.linspace(-80, 80, 9)):
        for el in np.radians([25, 40, 55, 70]):
            for r in (0.17, 0.22):
                a = base_az + az
                tip = cube + r * np.array([math.cos(el) * math.cos(a), math.cos(el) * math.sin(a), math.sin(el)])
                if tip[2] < 0.06 or tip[1] < -0.14 or tip[1] > 0.30 or tip[0] < 0.06:  # over the table, off the people's side
                    continue
                roll = rng.normal(size=3)
                best = None
                for sd in seeds:
                    q, ep, ea = ik_point(s, tip, cube, sd, roll_dir=roll)
                    if best is None or ep + 0.002 * ea < best[1] + 0.002 * best[2]:
                        best = (q, ep, ea)
                if best[1] < 0.004 and best[2] < 4 and -0.6 < best[0][0] < 1.6:  # the arm never swings back toward people
                    cands.append({"tip": tip, "q": best[0], "az": math.degrees(az), "el": math.degrees(el), "r": r})
    print(f"{len(cands)} reachable viewpoints", flush=True)
    # spread: greedy farthest-point in (az, el, r, wrist roll) space
    chosen = []
    feat = np.array([[c["az"] / 40, c["el"] / 20, c["r"] / 0.05, c["q"][5] / 0.8] for c in cands])
    if not len(cands):
        return s, []
    chosen.append(int(np.argmin(np.linalg.norm(np.array([c["q"] for c in cands]) - q_start, axis=1))))
    while len(chosen) < min(16, len(cands)):
        dmin = np.min(np.linalg.norm(feat[:, None] - feat[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(dmin)))
    # order as a short tour from the start, checking each path
    left = [cands[i] for i in chosen]
    tour, cur = [], np.array(q_start, float)
    while left:
        left.sort(key=lambda c: np.abs(c["q"] - cur).max())
        for i, c in enumerate(left):
            ok, worst = path_ok(s, cur, c["q"], cube)
            if ok:
                tour.append(c)
                cur = c["q"]
                left.pop(i)
                break
        else:
            print(f"  {len(left)} viewpoints dropped: no clear path", flush=True)
            break
    ok, worst = path_ok(s, cur, np.array(q_start, float), cube)
    while tour and not ok:  # the way home must be clear too
        tour.pop()
        cur = tour[-1]["q"] if tour else np.array(q_start, float)
        ok, worst = path_ok(s, cur, np.array(q_start, float), cube)
    for c in tour:
        print(f"  view az {c['az']:+4.0f} el {c['el']:2.0f} r {c['r']:.2f}: tip {np.round(c['tip'] * 1000).astype(int).tolist()} mm, "
              f"q {np.round(c['q'], 2).tolist()}", flush=True)
    return s, tour


def ik_camera(s, cam_pos, look_at, q0, iters=900):
    """The wrist camera (fitted mount, calib/safety._mount) at cam_pos looking at look_at, image-up as close to world
    up as the view allows (the camera upright)."""
    m, d = s.model, s.data
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
    dofs = s.arm_dadr["right"]
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"right_{n}") for n in s.joint_names]
    lo, hi = m.jnt_range[jids, 0] + 0.03, m.jnt_range[jids, 1] - 0.03
    Rm, tm = _mount()
    z = np.asarray(look_at, float) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])  # image right: horizontal; image down then points as far down as it can
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rc_want = np.stack([x, y, z], 1)
    Rt_want = Rc_want @ Rm.T  # tool axes that give that camera
    q = np.array(q0, float)
    for _ in range(iters):
        s.set_arm("right", q, 1.0, snap=True)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        Rt = d.site_xmat[sid].reshape(3, 3)
        pc = d.site_xpos[sid] + Rt @ tm
        e_pos = cam_pos - pc
        e_rot = 0.5 * sum(np.cross(Rt[:, k], Rt_want[:, k]) for k in range(3))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, pc, m.site_bodyid[sid])
        J = np.vstack([jp[:, dofs], 0.3 * jr[:, dofs]])
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([e_pos, 0.3 * e_rot])), lo, hi)
    s.set_arm("right", q, 1.0, snap=True)
    mujoco.mj_kinematics(m, d)
    Rt = d.site_xmat[sid].reshape(3, 3)
    pc = d.site_xpos[sid] + Rt @ tm
    ang = math.degrees(math.acos(np.clip((np.trace(Rt.T @ Rt_want) - 1) / 2, -1, 1)))
    return q, float(np.linalg.norm(cam_pos - pc)), ang


def plan_upright(cube, q_start, n=8):
    """New views under calib/safety.py: the camera 22-34 cm from the cube, upright, from several sides and heights."""
    s = Scene("metal", "right", props=[], draw=True)
    tips = reach(s, "metal")
    seeds = [np.array(q_start, float), np.array([0.9, -1.0, 1.2, -0.6, 0, 0]), np.array([0.6, -1.4, 1.6, -0.8, 0, 0]),
             np.array([1.2, -0.8, 1.0, -0.5, 0, 0]), np.array([0.3, -1.6, 1.9, -1.0, 0, 0])]
    cands = []
    base_az = math.atan2(-cube[1], -cube[0])
    for az in np.radians(np.linspace(-90, 110, 11)):
        for el in np.radians([20, 32, 45, 55]):
            for r in (0.22, 0.28, 0.34, 0.40):
                a = base_az + az
                cam = cube + r * np.array([math.cos(el) * math.cos(a), math.cos(el) * math.sin(a), math.sin(el)])
                best = None
                for sd in seeds:
                    q, ep, ea = ik_camera(s, cam, cube, sd)
                    if best is None or ep + 0.001 * ea < best[1] + 0.001 * best[2]:
                        best = (q, ep, ea)
                q, ep, ea = best
                if ep > 0.005 or ea > 6 or not (-0.6 < q[0] < 1.6):
                    continue
                tip = tip_of(s, q, tips)
                if tip[2] < 0.06 or tip[1] < -0.10 or tip[1] > 0.32 or tip[0] < 0.06 or not (0.15 <= np.linalg.norm(tip - cube) <= 0.36):
                    continue
                ok, why = check_pose(q)
                if not ok:
                    continue
                cands.append({"tip": tip, "q": q, "az": math.degrees(az), "el": math.degrees(el), "r": r})
    print(f"{len(cands)} upright viewpoints pass the pose checks", flush=True)
    if not cands:
        return s, []
    feat = np.array([[c["tip"][0] / 0.05, c["tip"][1] / 0.05, c["tip"][2] / 0.05, c["az"] / 30, c["el"] / 15] for c in cands])
    chosen = [int(np.argmax(feat[:, 0]))]
    while len(chosen) < min(3 * n, len(cands)):
        dmin = np.min(np.linalg.norm(feat[:, None] - feat[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(dmin)))
    left = [cands[i] for i in chosen]
    tour, cur = [], np.array(q_start, float)
    while left and len(tour) < n:
        left.sort(key=lambda c: np.abs(c["q"] - cur).max())
        for i, c in enumerate(left):
            if path_ok(s, cur, c["q"], cube)[0] and check_path(cur, c["q"])[0] and check_path(c["q"], q_start)[0]:
                tour.append(c)
                cur = c["q"]
                left.pop(i)
                break
        else:
            break
    for c in tour:
        print(f"  view az {c['az']:+4.0f} el {c['el']:2.0f} r {c['r']:.2f}: tip {np.round(c['tip'] * 1000).astype(int).tolist()} mm, "
              f"q {np.round(c['q'], 2).tolist()}", flush=True)
    return s, tour


def snap(tag):
    meta = {}
    for cam in ("wrist", "overhead"):
        r = urllib.request.urlopen(URL + f"/{cam}.jpg", timeout=5)
        (OUT / f"{tag}-{cam}.jpg").write_bytes(r.read())
        meta[cam] = {"joints": json.loads(r.headers.get("X-Joints") or "null"), "stamp": r.headers.get("X-Stamp")}
    meta["state"] = json.loads(get(URL + "/state"))
    (OUT / f"{tag}.json").write_text(json.dumps(meta))


def main():
    dry = "--dry" in sys.argv
    co = json.load(open(OUT / "cube_overhead.json"))
    cube = np.array([co["xy"][0], co["xy"][1], -0.008 + 0.035])
    g = Guard()
    st = g.check(full=not dry)
    if "--first" in sys.argv:  # resuming: home is where the first part of the run started
        q_start = np.array(json.load(open(OUT / "start_q.json")), float)
    else:
        q_start = np.array(st["q"], float)
        (OUT / "start_q.json").write_text(json.dumps(q_start.tolist()))
    saved = OUT / "views_plan.json"
    if saved.exists() and "--replan" not in sys.argv:
        s = Scene("metal", "right", props=[], draw=True)
        tour, cur = [], np.array(st["goal"], float)
        plan_ = json.load(open(saved))
        if "--first" in sys.argv:  # the views of this plan not yet taken
            plan_ = plan_[int(sys.argv[sys.argv.index("--skip") + 1]):] if "--skip" in sys.argv else plan_
        for c in plan_:
            c = {**c, "tip": np.array(c["tip"]), "q": np.array(c["q"])}
            ok, worst = path_ok(s, cur, c["q"], cube)
            ok = ok and check_path(cur, c["q"], recover=True)[0]
            if ok:
                tour.append(c); cur = c["q"]
            else:
                print(f"  dropped view az {c['az']:+.0f} el {c['el']:.0f}: {np.round(worst, 3).tolist()}")
        while tour and not path_ok(s, tour[-1]["q"], q_start, cube)[0]:
            tour.pop()
        print(f"{len(tour)} saved views pass the checks", flush=True)
    else:
        fit = RUN_FIT if RUN_FIT.exists() else None
        if fit is not None:  # aim with the cube where the wrist chain puts it (consistent with the arm model's FK)
            c = json.load(open(fit))["cubes"]["BIG70"]
            cube = np.array([c["x"], c["y"], -0.008 + 0.035])
        s, tour = plan_upright(cube, q_start)
    if "--first" not in sys.argv:  # a resumed run keeps the plan file as it was
        (OUT / "views_plan.json").write_text(json.dumps([{**c, "tip": c["tip"].tolist(), "q": c["q"].tolist()} for c in tour], indent=1))
    if dry or not tour:
        print("dry: nothing moved" if dry else "no views")
        return
    done = 0
    first = int(sys.argv[sys.argv.index("--first") + 1]) if "--first" in sys.argv else (
        len(list(OUT.glob("view[0-9][0-9].json"))) if "--replan" in sys.argv else 0)  # --first: resume a replanned tour
    try:
        for k, c in enumerate(tour):
            i = first + k
            if (OUT / f"view{i:02d}.json").exists() and "--redo" not in sys.argv:
                continue  # resume: views already saved by an earlier run
            (OUT / "STATUS.txt").write_text(f"moving to view {i} ({k + 1} of {len(tour)} new)\n")
            g.move(c["q"])
            time.sleep(1.0)
            g.check(full=False)
            snap(f"view{i:02d}")
            done += 1
            print(f"  view {i}: saved", flush=True)
    except Abort as e:
        print(f"ABORT: {e}", flush=True)
        try:
            post("/stop", {})
        except Exception:  # noqa: BLE001
            pass
        g.goal = None
    finally:
        # home slowly and directly (the last view's way home was checked; views sit high and away from everything)
        try:
            here = json.loads(get(URL + "/state"))["goal"]
            ok, why = check_path(here, q_start, recover=True)
            if not ok:
                (OUT / "STATUS.txt").write_text(f"NOT homed: safety refused the way home ({why}); arm holding where it is\n")
                raise SystemExit(f"safety refused the way home: {why}")
            g.goal = q_start
            post("/move", {"q": q_start.tolist(), "grip": 0.0, "rate": 0.2})
            time.sleep(0.3)
            t0 = time.time()
            while time.time() - t0 < 60 and json.loads(get(URL + "/state"))["moving"]:
                time.sleep(0.2)
            time.sleep(0.5)
            snap("end")
        finally:
            print(f"{done} views -> {OUT}")


if __name__ == "__main__":
    main()
