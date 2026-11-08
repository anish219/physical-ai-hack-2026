"""Drive one Metal arm with an SO-101 leader: python -m dream_hackathon.so101 --help."""

import argparse
import dataclasses
import math
import pathlib
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack

import numpy as np

LEADER_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LEADER_URDF = pathlib.Path(__file__).with_name("models") / "so101_leader.urdf"


class LeaderFK:
    """SO-101 handle-frame pose from its five calibrated arm angles.

    The trigger is independent of the handle frame and never moves the tool pose.
    The bundled URDF uses LeRobot's middle-of-range zero convention.
    """

    def __init__(self, urdf: pathlib.Path = LEADER_URDF, tool_link: str = "handle_frame_link") -> None:
        import mujoco

        from dream_hackathon import ik

        self.model = ik.urdf_model(urdf.read_text(), tool_link, (0, 0, 0))
        self.data = mujoco.MjData(self.model)
        self._indices = [self.model.jnt_qposadr[self.model.joint(name).id] for name in LEADER_JOINTS]
        self._site = self.model.site(ik.TCP_SITE).id

    def pose(self, action: Mapping[str, float]) -> np.ndarray:
        import mujoco

        self.data.qpos[self._indices] = np.radians([action[f"{name}.pos"] for name in LEADER_JOINTS])
        mujoco.mj_kinematics(self.model, self.data)
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self._site].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self._site]
        return pose


@dataclasses.dataclass
class PoseMapping:
    """Leader tool displacement/rotation mapped into the follower's base frame."""

    leader_start: np.ndarray
    follower_start: np.ndarray
    leader_to_base: np.ndarray
    translation_scale: float = 1.0

    def target(self, leader_pose: np.ndarray) -> np.ndarray:
        rotation = self.leader_to_base
        target = self.follower_start.copy()
        target[:3, 3] += self.translation_scale * rotation @ (leader_pose[:3, 3] - self.leader_start[:3, 3])
        turned = rotation @ leader_pose[:3, :3] @ self.leader_start[:3, :3].T @ rotation.T
        target[:3, :3] = turned @ self.follower_start[:3, :3]
        return target


class ToolTracker:
    """Re-anchor the tool mapping on Track and on a scale change; Untrack freezes it."""

    def __init__(self, leader_to_base: np.ndarray) -> None:
        self.rotation = leader_to_base
        self.mapping: PoseMapping | None = None
        self.revision = -1

    def update(self, leader_pose: np.ndarray, follower_pose: np.ndarray, state: dict) -> np.ndarray | None:
        if not state["tracking"]:
            self.mapping = None
            self.revision = state["revision"]
            return None
        if self.mapping is None or self.revision != state["revision"]:
            self.mapping = PoseMapping(leader_pose.copy(), follower_pose.copy(), self.rotation, state["scale"])
            self.revision = state["revision"]
        return self.mapping.target(leader_pose)


def base_rotation(rpy_deg: list[float]) -> np.ndarray:
    """Roll, pitch, yaw of the leader base in the follower base, degrees."""
    roll, pitch, yaw = map(math.radians, rpy_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def start_arm(controls, acquire, open_arm):
    """Report a failed Track startup on the page; retry only on another Track press."""
    lock = None
    try:
        lock = acquire()
        return open_arm(), lock
    except Exception as error:
        if lock is not None:
            lock.close()
        controls.change({"tracking": False})
        controls.report(arm_open=False, message=f"Arm startup failed: {error}")
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/station.yaml")
    parser.add_argument("--side", default="right", choices=("left", "right"), help="Follower arm (default: right)")
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--leader-kind", choices=("so101", "star"), default="so101")
    parser.add_argument("--leader-id", default="hackathon_so101")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate the leader only; do not open followers")
    parser.add_argument("--read-leader", action="store_true", help="Print leader positions; do not open followers")
    parser.add_argument("--dry-run", action="store_true", help="Read the selected follower and print targets")
    parser.add_argument("--hz", type=float, default=30)
    parser.add_argument("--translation-scale", type=float, default=1.0, help="Follower metres per leader metre")
    parser.add_argument("--replay-start-speed", type=float, default=0.5, help="Maximum approach joint speed, rad/s")
    parser.add_argument("--other-arm-server", help="Opposite arm server for the third recording camera")
    parser.add_argument(
        "--leader-to-base-rpy-deg",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        metavar=("ROLL", "PITCH", "YAW"),
        help="Leader base orientation in follower base",
    )
    parser.add_argument("--leader-urdf", type=pathlib.Path, default=LEADER_URDF)
    parser.add_argument("--leader-tool-link", default="handle_frame_link")
    parser.add_argument("--dashboard-port", type=int, default=8081)
    parser.add_argument("--no-dashboard", action="store_true", help="Track immediately without browser controls")
    parser.add_argument("--remote-server", help="Use an independently owned arm server instead of opening CAN")
    parser.add_argument(
        "--server-gripper-torque", action="store_true", help="Use the arm server's gripper torque setting"
    )
    parser.add_argument(
        "--dataset-dir", type=pathlib.Path, help="Examples folder (default: station dataset/so101/SIDE-context)"
    )
    parser.add_argument("--task", help="Task text saved with each example")
    parser.add_argument(
        "--record-cameras",
        nargs="+",
        choices=("overhead", "wrist_l", "wrist_r"),
        default=["overhead", "wrist_l", "wrist_r"],
        help="Camera roles explicitly recorded",
    )
    parser.add_argument(
        "--gripper-hold-torque-nm",
        type=float,
        default=0.5,
        help="Motor-output holding torque (0.1–2.0 N·m; default: 0.5)",
    )
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz must be positive")
    if not 0.1 <= args.gripper_hold_torque_nm <= 2.0:
        parser.error("--gripper-hold-torque-nm must be between 0.1 and 2.0")

    if args.leader_kind == "star":
        from dream_hackathon.star_leader import StarLeader

        leader = StarLeader(args.leader_port)
        if args.calibrate:
            parser.error("Star uses its existing encoder origins; calibration is a separate physical zero operation")
    else:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))
    robot = None
    lock = None
    dashboard = None
    recorder = None
    hardware_lock = threading.RLock()
    try:
        # Calibration is an explicit, separate operation; running teleop uses the saved calibration.
        leader.connect(calibrate=False)
        if args.calibrate:
            leader.calibrate()
            return
        if args.read_leader:
            while True:
                print(leader.get_action(), flush=True)
                time.sleep(1 / args.hz)

        from dream_hackathon import config, station_lock, teleop
        from dream_hackathon.arm_context import read as read_context_pose
        from dream_hackathon.arms.makermods import MakerModsArms, teleop_ik
        from dream_hackathon.remote_arm import RemoteArms, call, remote_cameras
        from dream_hackathon.so101_dashboard import Controls, LeaderDashboard
        from dream_hackathon.so101_gripper import apply_hold_torque
        from dream_hackathon.so101_gripper import feedback as gripper_feedback
        from dream_hackathon.so101_power import kill_controlled_arm
        from dream_hackathon.so101_recording import ExampleRecorder, recording_frame_enabled
        from dream_hackathon.star_replay import Replay, latest_take

        station = config.load(args.config)
        if station.arm != "metal":
            parser.error("this command expects a Metal station config")
        whole_station = station
        station = dataclasses.replace(station, arms={args.side: station.arms[args.side]})
        if args.leader_kind == "star":
            from dream_hackathon.star_leader import StarFK

            leader_fk = StarFK()
        else:
            leader_fk = LeaderFK(args.leader_urdf, args.leader_tool_link)
        solver = teleop_ik(station)[args.side]
        pause_untracked = args.leader_kind == "star"
        controls = Controls(
            args.translation_scale,
            args.side,
            args.dry_run,
            args.gripper_hold_torque_nm,
            pause_recording_untracked=pause_untracked,
        )
        controls.change({"replay_start_speed": args.replay_start_speed})
        tracker = ToolTracker(base_rotation(args.leader_to_base_rpy_deg))
        root = (args.dataset_dir or whole_station.dataset_dir / "so101" / f"{args.side}-context").expanduser()
        task = args.task or whole_station.task or "do the demonstrated task"
        if args.leader_kind == "star":
            from dream_hackathon.star_leader import recording_destination

            root, task = recording_destination(root, args.side, task)
        other_side = "left" if args.side == "right" else "right"
        recorder = ExampleRecorder(
            whole_station,
            args.side,
            root,
            task,
            args.record_cameras,
            context_pose=None if args.remote_server else lambda: read_context_pose(other_side),
            camera_factory=(
                remote_cameras(args.remote_server, side=args.side, other_base=args.other_arm_server)
                if args.remote_server
                else None
            ),
            include_inactive_wrist=args.leader_kind == "star" and bool(args.other_arm_server),
        )
        controls.report(dataset_dir=str(root), saved_episodes=recorder.saved_count())

        def kill_arm():
            nonlocal robot, lock
            # Serialize disable with motor commands: no old command can follow the disable.
            with hardware_lock:
                active, robot = robot, None
                tracker.mapping = None
                try:
                    if args.remote_server:
                        return RemoteArms.disable(args.remote_server)
                    return kill_controlled_arm(station, args.side, active)
                finally:
                    if lock is not None:
                        lock.close()
                        lock = None

        if args.no_dashboard:
            controls.change({"tracking": True})
        else:

            def change_control(values):
                with hardware_lock:
                    return controls.change(values)

            dashboard = LeaderDashboard(controls, args.dashboard_port, on_kill=kill_arm, on_control=change_control)
            print(f"dashboard: http://127.0.0.1:{dashboard.port} (starts untracked)", flush=True)
        print(
            f"{args.leader_kind} tool pose -> {args.side} Metal tool pose; "
            f"{'dry run' if args.dry_run else 'live'}. Ctrl+C exits.",
            flush=True,
        )
        printed = 0.0
        next_frame = 0.0
        return_goal = None
        grip_revision = -1
        grip_feedback_at = 0.0
        replay = None
        while True:
            tick = time.monotonic()
            action = leader.get_action()
            sample = None
            with hardware_lock:
                state = controls.snapshot()
                if not state["replay_requested"]:
                    replay = None
                    if state["replay_phase"] in ("pending", "approaching", "playing"):
                        controls.report(replay_phase="idle")
                if args.remote_server and robot is None and not state["tracking"]:
                    remote_state = call(args.remote_server, "/state")
                    if remote_state["arm_open"]:
                        robot = RemoteArms(args.remote_server, args.side, energize=not args.dry_run, open_arm=False)
                        # Attach to an already-powered server without changing its held targets.
                        command = np.asarray(robot.connected_state["goal"], dtype=float)
                        grip = float(robot.connected_state["goal_grip"])
                        controls.report(arm_open=True, power_off=False, message="Arm on; untracked")
                        grip_revision = -1
                if (state["tracking"] or state["replay_requested"]) and robot is None:
                    robot, lock = start_arm(
                        controls,
                        lambda: (
                            ExitStack()
                            if args.remote_server
                            else station_lock.acquire_sides(f"so101 {args.side}", [args.side])
                        ),
                        lambda: (
                            RemoteArms(args.remote_server, args.side, energize=not args.dry_run)
                            if args.remote_server
                            else MakerModsArms(station, energize=not args.dry_run)
                        ),
                    )
                    if robot is not None:
                        joints, grips = robot.read()
                        command, grip = joints[args.side].copy(), grips[args.side]
                        action = leader.get_action()
                        controls.report(arm_open=True, power_off=False, message="")
                        grip_revision = -1
                if robot is not None:
                    if not args.dry_run and not args.server_gripper_torque and grip_revision != state["grip_revision"]:
                        try:
                            apply_hold_torque(robot, args.side, state["grip_torque_nm"])
                            controls.report(grip_error="")
                        except Exception as error:
                            controls.report(grip_error=f"Grip torque: {error}")
                        grip_revision = state["grip_revision"]
                    if not args.dry_run and tick >= grip_feedback_at:
                        controls.report(**gripper_feedback(robot, args.side))
                        grip_feedback_at = tick + 0.2
                    if state["tracking"] and tracker.mapping is None:
                        joints, grips = robot.read()
                        command, grip = joints[args.side].copy(), grips[args.side]
                    target_pose = None
                    replay_command = False
                    if state["home_requested"]:
                        if return_goal is None:
                            return_goal = robot.start_joints[args.side].copy()
                        tracker.mapping = None
                        command, arrived = teleop.home_step(command, return_goal, 1 / args.hz)
                        if arrived:
                            return_goal = None
                            controls.report(
                                home_requested=False, homing=False, message="Returned to the captured startup pose"
                            )
                    elif state["replay_requested"] and recorder.episode is None:
                        tracker.mapping = None
                        return_goal = None
                        try:
                            if replay is None:
                                measured, measured_grips = robot.read()
                                take = latest_take(root, args.side, station.arm)
                                replay = Replay(
                                    take,
                                    measured[args.side],
                                    measured_grips[args.side],
                                    state["replay_start_speed"],
                                    tick,
                                )
                                controls.report(
                                    replay_episode=take.episode, replay_duration=float(take.timestamps[-1])
                                )
                            command, grip, finished = replay.step(tick)
                            replay_command = True
                            controls.report(replay_phase=replay.phase, replay_elapsed=replay.elapsed)
                            if finished:
                                controls.report(
                                    replay_requested=False, message=f"Replayed example {replay.take.episode}"
                                )
                        except Exception as error:
                            controls.report(replay_requested=False, replay_phase="error", replay_error=str(error))
                            replay = None
                    else:
                        return_goal = None
                        target_pose = tracker.update(leader_fk.pose(action), solver.fk(command), state)
                        if target_pose is not None:
                            command = solver.step(target_pose, command, 1 / args.hz)
                            grip = action["gripper.pos"] / 100.0
                    if args.dry_run and target_pose is not None and tick - printed >= 1:
                        print(
                            f"{args.side}: tool_xyz_m={target_pose[:3, 3].round(4)} "
                            f"joints_deg={np.degrees(command).round(2)} gripper={grip:.3f}",
                            flush=True,
                        )
                        printed = tick
                    if not args.dry_run and (
                        not args.remote_server or state["tracking"] or state["home_requested"] or replay_command
                    ):
                        robot.command({args.side: command}, {args.side: grip})
                    elif args.remote_server and not args.dry_run:
                        held = call(args.remote_server, "/state")
                        command = np.asarray(held["goal"], dtype=float)
                        grip = float(held["goal_grip"])
                    if recording_frame_enabled(state, pause_untracked=pause_untracked):
                        measured, measured_grips = robot.read()
                        sample = (measured[args.side], measured_grips[args.side], command.copy(), grip)
            # Camera opening and video encoding never hold the motor-command mutex,
            # so Kill arm can disable motors while an example starts or saves.
            try:
                record_state = controls.snapshot()
                if record_state["reject_requested"]:
                    message = recorder.reject(current_only=record_state["reject_kind"] == "current")
                    controls.report(
                        reject_requested=False,
                        recording=False,
                        frames=0,
                        recording_error="",
                        saved_episodes=recorder.saved_count(),
                        message=message,
                    )
                requested = controls.snapshot()["recording_requested"]
                if requested and recorder.episode is None:
                    recorder.start()
                    controls.report(recording=True, frames=0, message="", recording_error="")
                    next_frame = 0.0
                    sample = None
                if not requested and recorder.episode is not None:
                    index = recorder.stop()
                    controls.report(
                        recording=False,
                        saved_episodes=recorder.writer.total_episodes,
                        message=f"Saved example {index} · {root}",
                    )
                if (
                    recorder.episode is not None
                    and sample is not None
                    and tick >= next_frame
                    and recording_frame_enabled(controls.snapshot(), pause_untracked=pause_untracked)
                ):
                    recorder.add(*sample)
                    controls.report(frames=recorder.frames)
                    next_frame = tick + 1 / whole_station.fps - 1e-6
            except Exception as error:
                controls.change({"recording_requested": False})
                controls.report(recording=False, reject_requested=False, recording_error=str(error))
                print(f"Recording failed: {error}", flush=True)
                if recorder.episode is not None:
                    try:
                        recorder.stop()
                    except Exception as save_error:
                        controls.report(recording_error=f"{error}; save: {save_error}")
                        print(f"Recording save failed: {save_error}", flush=True)
            time.sleep(max(0, 1 / args.hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if robot is not None:
                robot.close()
        finally:
            try:
                if leader.is_connected:
                    leader.disconnect()
            finally:
                if lock is not None:
                    lock.close()
                if dashboard is not None:
                    dashboard.close()
                if recorder is not None:
                    recorder.close()


if __name__ == "__main__":
    main()
