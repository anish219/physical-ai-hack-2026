"""Recorded USB approach, wrist CV alignment, retargeted grasp/pull, fixed shortcut drop."""

import argparse
import json
import math
import pathlib
import time
import urllib.request

import cv2
import numpy as np

from dream_hackathon.bundle_vision import BundleVision, servo


class HttpArm:
    """Borrow an already running per-arm server and its wrist stream."""

    def __init__(self, base):
        self.base = base.rstrip("/")
        self.last_stamp = None

    def state(self):
        with urllib.request.urlopen(self.base + "/state", timeout=10) as response:
            state = json.load(response)
        return np.asarray(state["q"], float), float(state["grip"])

    def command(self, q, grip):
        request = urllib.request.Request(
            self.base + "/target",
            data=json.dumps({"q": np.asarray(q).tolist(), "grip": float(grip)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()

    def image(self):
        # Wait for a distinct frame, rather than counting a cached JPEG as new evidence.
        while True:
            with urllib.request.urlopen(self.base + "/wrist.jpg", timeout=10) as response:
                stamp = response.headers.get("X-Stamp")
                encoded = response.read()
            if stamp is None or stamp != self.last_stamp:
                self.last_stamp = stamp
                frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Wrist server returned an invalid image")
                return frame
            time.sleep(0.01)


def sample(rows, t):
    times = np.array([row["t"] for row in rows])
    q = np.array([np.interp(t, times, [row["q"][j] for row in rows]) for j in range(6)])
    return q, float(np.interp(t, times, [row["grip"] for row in rows]))


def section(rows, start, end):
    times = [start, *[row["t"] for row in rows if start < row["t"] < end], end]
    result = []
    for t in sorted(set(times)):
        q, grip = sample(rows, t)
        result.append({"t": t - start, "q": q.tolist(), "grip": grip})
    return result


def play(arm, rows, hz=30, clock=time.monotonic, sleep=time.sleep):
    """Interpolate at wall-clock time; always send the exact final target."""
    start = clock()
    while True:
        tick = clock()
        elapsed = min(tick - start, rows[-1]["t"])
        arm.command(*sample(rows, elapsed))
        if elapsed >= rows[-1]["t"]:
            return
        sleep(max(0, 1 / hz - (clock() - tick)))


def move_to(arm, q, grip, speed, hz=30, settle=0, sleep=time.sleep):
    start_q, start_grip = arm.state()
    duration = max(float(np.max(np.abs(q - start_q))) / speed, abs(grip - start_grip) / 1.5)
    play(
        arm,
        [
            {"t": 0, "q": start_q.tolist(), "grip": start_grip},
            {"t": duration, "q": np.asarray(q).tolist(), "grip": grip},
        ],
        hz=hz,
    )
    sleep(settle)


def pose_error(target, current):
    rotation = cv2.Rodrigues(target[:3, :3] @ current[:3, :3].T)[0].ravel()
    return np.r_[target[:3, 3] - current[:3, 3], rotation]


class ToolKinematics:
    """FK and numerical pose IK from the station's existing calibrated arm model."""

    def __init__(self, fk):
        self.fk = fk

    def solve(self, target, seed):
        q = np.asarray(seed, float).copy()
        for _ in range(80):
            current = self.fk(q)
            error = pose_error(target, current)
            if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 1e-4:
                return q
            columns = []
            for j in range(len(q)):
                perturbed = q.copy()
                perturbed[j] += 1e-4
                columns.append(pose_error(self.fk(perturbed), current) / 1e-4)
            jacobian = np.column_stack(columns)
            q += np.linalg.solve(jacobian.T @ jacobian + 1e-8 * np.eye(len(q)), jacobian.T @ error)
        raise ValueError("Could not retarget the recorded pull to the CV-aligned tool pose")

    @classmethod
    def from_config(cls, path, side):
        # Only builds a model. Does not open motors, power, CAN, or cameras.
        from dream_hackathon import config, ik

        station = config.load(path)
        mm = station.makermods
        if mm is None:
            raise ValueError("This recording uses a MakerMods arm")
        tcp = station.arms[side].tcp_xyz_m
        if tcp is None:
            raise ValueError("Use the station's measured TCP in its existing config")
        import mujoco

        model = ik.urdf_model(mm.urdf.read_text(), mm.flange_link, tcp)
        data = mujoco.MjData(model)
        indices = [model.jnt_qposadr[model.joint(name).id] for name in mm.urdf_joints]
        site = model.site(ik.TCP_SITE).id

        def fk(q):
            data.qpos[indices] = np.asarray(q) * np.asarray(mm.joint_signs)
            mujoco.mj_kinematics(model, data)
            pose = np.eye(4)
            pose[:3, :3] = data.site_xmat[site].reshape(3, 3)
            pose[:3, 3] = data.site_xpos[site]
            return pose

        return cls(fk)


def retarget(rows, recorded_aim, aligned_aim, kinematics):
    """Apply one rigid base-frame correction to every recorded tool pose."""
    transform = kinematics.fk(aligned_aim) @ np.linalg.inv(kinematics.fk(recorded_aim))
    corrected = []
    seed = np.asarray(aligned_aim).copy()
    for row in rows:
        seed = kinematics.solve(transform @ kinematics.fk(np.asarray(row["q"])), seed)
        corrected.append({**row, "q": seed.tolist()})
    return corrected


def shortcut(rows, corrected_end, plan):
    """Reconnect to the absolute measured drop pose, holding the recorded grip."""
    transfer = [row for row in rows if row["segment"] == "planned direct transfer"]
    release = [row for row in rows if row["segment"] == "planned release at recorded pose"]
    end = transfer[-1]
    duration = max(
        plan["direct_transfer_duration_s"],
        float(np.max(np.abs(np.asarray(end["q"]) - corrected_end["q"])))
        / plan["initial_approach_max_joint_speed_rad_s"],
    )
    result = [
        {"t": 0, "q": corrected_end["q"], "grip": corrected_end["grip"]},
        {"t": duration, "q": end["q"], "grip": corrected_end["grip"]},
    ]
    for row in release:
        result.append({"t": duration + row["t"] - end["t"], "q": row["q"], "grip": row["grip"]})
    return result


def run(arm, rows, plan, vision, kinematics, report, *, play_fn=play, move_fn=move_to, servo_fn=servo):
    aim_time = plan["aim_time_s"]
    end_time = max(row["t"] for row in rows if row["segment"] == "recorded pull")
    hz = plan["control_hz"]
    report("start", {"aim_time_s": aim_time}, None)
    move_fn(arm, *sample(rows, 0), plan["initial_approach_max_joint_speed_rad_s"], hz=hz)
    report("approach", {}, None)
    play_fn(arm, section(rows, 0, aim_time), hz=hz)
    time.sleep(plan["servo"]["settle_s"])

    def align_move(q, grip):
        move_fn(arm, q, grip, plan["servo"]["speed_rad_s"], hz=hz, settle=plan["servo"]["settle_s"])

    aligned = servo_fn(arm, vision, align_move, plan["servo"], report)
    recorded_aim, _ = sample(rows, aim_time)
    pull = retarget(section(rows, aim_time, end_time), recorded_aim, aligned, kinematics)
    drop = shortcut(rows, pull[-1], plan)
    report("grasp-pull", {"corrected_targets": len(pull), "aligned_q": aligned.tolist()}, None)
    play_fn(arm, pull, hz=hz)
    report("shortcut-drop", {"duration_s": drop[-1]["t"]}, None)
    play_fn(arm, drop, hz=hz)
    report("complete", {"drop_q": drop[-1]["q"], "drop_grip": drop[-1]["grip"]}, None)


def load_plan(path):
    path = pathlib.Path(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (path.parent / plan["targets_file"]).read_text().splitlines()]
    times = np.array([row["t"] for row in rows])
    if not np.all(np.diff(times) > 0) or not np.isfinite(times).all():
        raise ValueError("Recorded timestamps must be finite and strictly increasing")
    for row in rows:
        if np.shape(row["q"]) != (6,) or not np.isfinite(row["q"]).all() or not math.isfinite(row["grip"]):
            raise ValueError("Each target needs six finite joint angles and a finite grip")
    if not 0 < plan["aim_time_s"] < max(r["t"] for r in rows if r["segment"] == "recorded pull"):
        raise ValueError("Aim checkpoint must be inside the recorded approach/pull")
    reference = cv2.imread(str(path.parent / plan["reference_image"]))
    if reference is None:
        raise ValueError("Cannot read the aim reference image")
    return plan, rows, BundleVision(reference, plan["vision"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=pathlib.Path)
    parser.add_argument("--execute", action="store_true", help="Run the full motion on the existing arm server")
    parser.add_argument("--config", type=pathlib.Path, help="Existing station config for Cartesian retargeting")
    parser.add_argument("--server", help="Override the plan's per-arm server URL")
    parser.add_argument("--check-image", type=pathlib.Path, help="Offline alignment check; no arm requests")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("inspection/bundle-pull-run"))
    args = parser.parse_args()
    if args.execute and args.check_image:
        parser.error("Choose --execute or --check-image")
    if args.execute and args.config is None:
        parser.error("--execute requires --config for the recorded arm's kinematics")
    plan, rows, vision = load_plan(args.plan)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "events.jsonl").open("w", encoding="utf-8") as events:

        def report(phase, status, image):
            record = {"phase": phase, "time": time.time(), **status}
            print(json.dumps(record), flush=True)
            events.write(json.dumps(record) + "\n")
            events.flush()
            if image is not None:
                cv2.imwrite(str(args.output / "alignment.jpg"), image)

        if args.check_image:
            image = cv2.imread(str(args.check_image))
            if image is None:
                raise ValueError(f"Cannot read {args.check_image}")
            _, status, overlay = vision.measure(image)
            report("offline-image", status, overlay)
        elif args.execute:
            kinematics = ToolKinematics.from_config(args.config, plan["side"])
            run(HttpArm(args.server or plan["server"]), rows, plan, vision, kinematics, report)
        else:
            _, status, overlay = vision.measure(vision.reference)
            report(
                "inspect",
                {
                    "targets": len(rows),
                    "aim_time_s": plan["aim_time_s"],
                    "recorded_pull_end_s": plan["retained_source_range_s"][1] - plan["retained_source_range_s"][0],
                    "source": plan["source_directory"],
                    "reference": status,
                },
                overlay,
            )


if __name__ == "__main__":
    main()
