"""Unified page, source handoff, remote adapters and independent power lifecycle; fake hardware only."""

import json
import pathlib
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from conftest import filled_template
from test_armserver import request

from dream_hackathon import config, remote_arm, unified_control
from dream_hackathon.armserver import ArmBinding
from dream_hackathon.managed_arms import ManagedArm
from dream_hackathon.so101_dashboard import Controls


class Robot:
    def __init__(self, side):
        self.side = side
        self.start_joints = {side: np.zeros(6)}
        self.q, self.grip = np.zeros(6), 0.5
        self.events = []
        self.targets = []
        self.torque = 0.5

    def read(self):
        return {self.side: self.q.copy()}, {self.side: self.grip}

    def command(self, joints, grips):
        self.events.append("command")
        self.targets.append((joints[self.side].copy(), grips[self.side]))
        self.q, self.grip = joints[self.side].copy(), grips[self.side]

    def set_hold_torque(self, torque):
        self.torque = torque

    def gripper_feedback(self):
        return dict(grip_measured_torque_nm=0.2, grip_applied_torque_nm=self.torque, grip_holding=False)

    def power_off(self):
        self.events.append("off")
        return []

    def close(self):
        self.events.append("close")


def make_node(tmp_path, side, *, opener=None):
    station = config.load(filled_template("yam", tmp_path))
    robot = Robot(side)
    hub = SimpleNamespace(subscribe=lambda *args: None, unsubscribe=lambda *args: None)
    node = ManagedArm(
        station,
        side,
        ArmBinding(side, None, "wrist", "scene", tmp_path / "arms.lock"),
        hub,
        0,
        acquire=ExitStack,
        opener=opener or (lambda: robot),
        kill=lambda: [],
        zero_path=tmp_path / f"{side}-zero.json",
    )
    return node, robot


def test_remote_tracking_adapter_uses_server_and_close_never_powers_down(tmp_path):
    node, robot = make_node(tmp_path, "right")
    base = f"http://127.0.0.1:{node.port}"
    try:
        adapter = remote_arm.RemoteArms(base, "right")
        assert node.snapshot()["arm_open"]
        adapter.command({"right": np.full(6, 0.04)}, {"right": 0.8})
        deadline = time.monotonic() + 2
        while robot.q[0] < 0.039:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        adapter.set_hold_torque(1.0)
        deadline = time.monotonic() + 2
        while adapter.gripper_feedback()["grip_applied_torque_nm"] != 1.0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert adapter.gripper_feedback()["grip_applied_torque_nm"] == 1.0
        adapter.close()
        assert node.snapshot()["arm_open"] and "off" not in robot.events
        assert adapter.power_off() == []
        assert robot.events[-2:] == ["off", "close"]
        count = robot.events.count("command")
        time.sleep(0.05)
        assert robot.events.count("command") == count
        assert not node.snapshot()["arm_open"]
    finally:
        node.close()


def test_blocked_left_startup_does_not_block_right_power_or_http(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def slow_open():
        entered.set()
        assert release.wait(5)
        return Robot("left")

    left, _ = make_node(tmp_path, "left", opener=slow_open)
    right, robot = make_node(tmp_path, "right")
    pending = threading.Thread(target=left.power, args=(True,))
    try:
        pending.start()
        assert entered.wait(2)
        assert right.power(True)["arm_open"]
        assert request(f"http://127.0.0.1:{right.port}", "/move", {"q": [0.01] * 6})[0] == 200
        assert right.power(False)["problems"] == []
        assert robot.events[-2:] == ["off", "close"]
    finally:
        release.set()
        pending.join(2)
        left.close()
        right.close()


def test_assignment_switch_changes_only_leader_process_and_preserves_settings(tmp_path, monkeypatch):
    commands, signals = [], []

    class Process:
        dead = False

        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            return 0 if self.dead else None

        def send_signal(self, signal):
            signals.append(signal)
            self.dead = True

        def wait(self):
            return 0

    monkeypatch.setattr(unified_control.subprocess, "Popen", Process)
    monkeypatch.setattr(
        unified_control, "call", lambda *args, **kwargs: Controls(1.6, "right", grip_torque_nm=1).snapshot()
    )
    leader = unified_control.LeaderAssignment(
        pathlib.Path("station.yaml"),
        "leader",
        {"left": "http://left", "right": "http://right"},
        tmp_path,
        log_dir=tmp_path,
    )
    leader.select("right")
    assert "--server-gripper-torque" in commands[-1]
    assert commands[-1][commands[-1].index("--remote-server") + 1] == "http://right"
    leader.select("left")
    assert commands[-1][commands[-1].index("--remote-server") + 1] == "http://left"
    assert leader.settings["right"] == {"scale": 1.6, "grip_torque_nm": 1}
    leader.select(None)
    assert len(commands) == 2 and len(signals) == 2 and leader.side is None
    assert not any("armserver" in str(command) for command in commands)


def test_page_contains_every_existing_leader_control_and_routes_to_selected_child(tmp_path):
    actions = []

    class Leader:
        side = "right"

        def state(self):
            return Controls(1, "right").snapshot()

        def action(self, path, body):
            actions.append((path, body))

        def select(self, side):
            self.side = side

    nodes = {
        side: SimpleNamespace(
            snapshot=lambda: dict(arm_open=False, requested_power=False, message="Off", grip_torque_nm=1.0),
            power=lambda enabled: {"problems": []},
            cancel_home=lambda: None,
            return_zero=lambda: actions.append(("home", {})),
            set_zero=lambda: actions.append(("zero", {})),
        )
        for side in ("left", "right")
    }
    page = unified_control.UnifiedPage(unified_control.UnifiedController(nodes, Leader()), 0)
    base = f"http://127.0.0.1:{page.port}"
    try:
        code, html, _ = request(base, "/")
        assert code == 200
        for control in (
            "track",
            "untrack",
            "tracking-state",
            "record",
            "reject",
            "scale",
            "left-grip-torque",
            "left-grip-feedback",
            "right-grip-torque",
            "right-grip-feedback",
            "left-power",
            "right-power",
            "left-home",
            "right-home",
            "left-zero",
            "right-zero",
            "left-input-so101",
            "right-input-so101",
            "left-input-dream",
            "right-input-dream",
            "left-input-auto",
            "right-input-auto",
            "so101-panel",
            "dream-panel",
        ):
            assert f'id="{control}"'.encode() in html
        assert html.index(b'id="so101-panel"') < html.index(b'id="dream-panel"')
        for control in (
            "api-selection",
            "api-task",
            "api-checkpoint",
            "api-connect",
            "api-call",
            "api-run",
            "api-stop",
        ):
            assert f'id="{control}"'.encode() in html
        for path in ("/api/control", "/api/home", "/api/reject", "/api/kill"):
            assert request(base, path, {})[0] == 200
        assert [a[0] for a in actions] == ["/api/control", "/api/control", "home", "/api/reject", "/api/kill"]
        assert request(base, "/api/arms/left/home", {})[0] == 200
        assert request(base, "/api/arms/right/zero", {})[0] == 200
        assert request(base, "/api/arms/left/power", {"enabled": True})[0] == 200
        assert request(base, "/api/assignment", {"side": None})[0] == 200
        state = json.loads(request(base, "/api/state")[1])
        assert state["assigned_side"] is None and set(state["arms"]) == {"left", "right"}
    finally:
        page.close()


def test_recorder_borrows_camera_jpegs_without_reopening_devices(tmp_path):
    node, robot = make_node(tmp_path, "right")
    try:
        frame = np.full((72, 128, 3), 100, np.uint8)
        jpeg = cv2.imencode(".jpg", frame)[1].tobytes()
        node.control.camera_frame("wrist", jpeg, 123)
        node.control.camera_frame("overhead", jpeg, 123)
        factory = remote_arm.remote_cameras(f"http://127.0.0.1:{node.port}")
        cameras = factory([SimpleNamespace(role="overhead"), SimpleNamespace(role="wrist_r")], 320, 180)
        for camera in cameras.values():
            assert camera.latest().shape == (180, 320, 3)
            assert np.mean(camera.latest()) == pytest.approx(100, abs=1)
            camera.close()
        assert robot.events == [] and not node.snapshot()["arm_open"]
    finally:
        node.close()


def test_remote_dry_run_never_enables_or_commands(monkeypatch):
    requests = []

    def read_only(base, path, body=None):
        requests.append((path, body))
        return {"arm_open": True, "start_joints": [0] * 6}

    monkeypatch.setattr(remote_arm, "call", read_only)
    robot = remote_arm.RemoteArms("http://server", "right", energize=False)
    with pytest.raises(RuntimeError, match="dry run"):
        robot.command({"right": np.zeros(6)}, {"right": 0.5})
    robot.close()
    assert requests == [("/state", None)]


def test_leader_targets_reach_sdk_without_added_speed_or_gripper_slew_limit(tmp_path):
    node, robot = make_node(tmp_path, "left")
    try:
        adapter = remote_arm.RemoteArms(f"http://127.0.0.1:{node.port}", "left")
        target = np.array([0.20, -0.10, 0.15, 0.04, -0.12, 0.06])
        adapter.command({"left": target}, {"left": 0.9})
        deadline = time.monotonic() + 2
        while not any(np.any(q) for q, _ in robot.targets):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        first_target = next((q, g) for q, g in robot.targets if np.any(q))
        np.testing.assert_array_equal(first_target[0], target)
        assert first_target[1] == 0.9
        assert node.snapshot()["target_mode"] == "direct"
    finally:
        node.close()


def test_main_page_routes_api_actions_and_power_off_cancels_only_matching_callers():
    from test_dream_caller import Model, fake_arm, listener, wait_for

    from dream_hackathon.dream_caller import ArmEndpoint, StationCaller, handler_for

    model = Model()
    model.release.set()
    logs = {"left": [], "right": []}
    with listener(fake_arm(logs["left"])) as left, listener(fake_arm(logs["right"])) as right:
        endpoints = {"left": ArmEndpoint(left), "right": ArmEndpoint(right)}
        workers = {
            side: StationCaller(side, endpoints, client_factory=lambda *_: model) for side in ("left", "right", "both")
        }
        try:
            with listener(handler_for(workers)) as service_url:
                service = SimpleNamespace(url=service_url)
                callers = {side: unified_control.DreamCallerProxy(service, side) for side in workers}
                leader = SimpleNamespace(side=None, state=lambda: Controls(1, "none").snapshot())
                nodes = {
                    side: SimpleNamespace(snapshot=lambda: {"arm_open": True}, power=lambda _: {"problems": []})
                    for side in ("left", "right")
                }
                controller = unified_control.UnifiedController(nodes, leader, callers)
                page = unified_control.UnifiedPage(controller, 0)
                base = f"http://127.0.0.1:{page.port}"
                try:
                    assert request(base, "/api/dream/both/connect", {})[0] == 200
                    wait_for(lambda: workers["both"].sample is not None)
                    assert request(base, "/api/dream/both/call", {"task": "inspect"})[0] == 200
                    wait_for(lambda: workers["both"].snapshot()["calls"] == 1)
                    state = json.loads(request(base, "/api/state")[1])
                    assert set(state["dream"]) == {"left", "right", "both"}
                    assert len(state["dream"]["both"]["rows"][0]) == 14
                    assert not any(path == "/target" for log in logs.values() for path, _ in log)
                    assert request(base, "/api/arms/left/power", {"enabled": False})[0] == 200
                    assert workers["left"].cancel.is_set() and workers["both"].cancel.is_set()
                    assert not workers["right"].cancel.is_set()
                    assert request(base, "/api/dream/both/stop", {})[0] == 200
                    assert request(base, "/api/dream/invalid/run", {})[0] == 400
                finally:
                    page.close()
        finally:
            for worker in workers.values():
                worker.close()


def test_saved_zero_survives_kill_reopen_and_service_restart(tmp_path):
    from test_dream_caller import wait_for

    node, robot = make_node(tmp_path, "left")
    try:
        node.power(True)
        assert node.snapshot()["zero_joints"] == [0] * 6
        node.control.target([0.15] * 6, 0.8)
        wait_for(lambda: node.snapshot()["q"] == [0.15] * 6)
        node.set_zero()
        node.power(False)
        robot.q = np.full(6, -0.1)
        robot.start_joints["left"] = robot.q.copy()
        node.power(True)
        assert node.snapshot()["start_joints"] == [-0.1] * 6
        assert node.snapshot()["zero_joints"] == [0.15] * 6
        node.return_zero()
        wait_for(lambda: not node.snapshot()["homing"] and np.allclose(robot.q, 0.15))
        node.power(False)
        assert node.snapshot()["zero_joints"] == [0.15] * 6
    finally:
        node.close()
    reopened, _ = make_node(tmp_path, "left")
    try:
        assert reopened.snapshot()["zero_joints"] == [0.15] * 6
        # Return is available even with the arm off, and powers only this arm.
        reopened.return_zero()
        wait_for(lambda: not reopened.snapshot()["homing"])
        assert reopened.snapshot()["arm_open"]
    finally:
        reopened.power(False)
        reopened.close()


def test_input_selection_hands_off_and_so101_is_exclusive(tmp_path):
    from test_dream_caller import Model, fake_arm, listener, wait_for

    from dream_hackathon.dream_caller import ArmEndpoint, StationCaller

    class Leader:
        side = None

        def state(self):
            return Controls(1, self.side or "none").snapshot()

        def select(self, side):
            self.side = side

        def action(self, path, body):
            pass

    left_node, _ = make_node(tmp_path, "left")
    right_node, _ = make_node(tmp_path, "right")
    nodes = {"left": left_node, "right": right_node}
    logs, model = {"left": [], "right": []}, Model()
    with listener(fake_arm(logs["left"])) as left, listener(fake_arm(logs["right"])) as right:
        endpoints = {"left": ArmEndpoint(left), "right": ArmEndpoint(right)}
        callers = {
            side: StationCaller(side, endpoints, client_factory=lambda *_: model) for side in ("left", "right", "both")
        }
        controller = unified_control.UnifiedController(nodes, Leader(), callers)
        try:
            controller.set_arm_torque("left", 1.35)
            controller.set_arm_torque("right", 0.75)
            controller.select_input("right", "so101")
            controller.select_input("left", "so101")
            assert controller.inputs == {"left": "so101", "right": None}
            assert controller.leader.side == "left"
            controller.select_input("left", None)
            assert controller.leader.side is None
            for side in nodes:
                controller.select_input(side, "dream")
            assert controller.inputs == {"left": "dream", "right": "dream"}
            assert left_node.snapshot()["grip_torque_nm"] == 1.35
            assert right_node.snapshot()["grip_torque_nm"] == 0.75
            callers["both"].connect()
            wait_for(lambda: callers["both"].sample is not None)
            controller.dream_action("both", "run", {"task": "move"})
            assert model.entered.wait(2)
            controller.select_input("right", "so101")
            assert controller.inputs == {"left": "dream", "right": "so101"}
            assert left_node.snapshot()["grip_torque_nm"] == 1.35
            assert right_node.snapshot()["grip_torque_nm"] == 0.75
            model.release.set()
            wait_for(lambda: not callers["both"].snapshot()["busy"])
            assert not any(path == "/target" for log in logs.values() for path, _ in log)
            with pytest.raises(ValueError, match="Select Dream API"):
                controller.dream_action("both", "run", {"task": "move"})
            # A stale API Stop must not send a hold to an arm that now follows the leader.
            controller.dream_action("both", "stop", {})
            assert not any(path == "/stop" for log in logs.values() for path, _ in log)
        finally:
            model.release.set()
            for caller in callers.values():
                caller.close()
            for node in nodes.values():
                node.close()


def test_arm_torque_endpoint_works_without_leader_and_is_applied_during_api_targets(tmp_path):
    from test_dream_caller import wait_for

    left, lrobot = make_node(tmp_path, "left")
    right, rrobot = make_node(tmp_path, "right")
    leader = SimpleNamespace(side=None, state=lambda: Controls(1, "none").snapshot())
    page = unified_control.UnifiedPage(unified_control.UnifiedController({"left": left, "right": right}, leader), 0)
    base = f"http://127.0.0.1:{page.port}"
    try:
        assert request(base, "/api/arms/left/grip-torque", {"torque_nm": 1.4})[0] == 200
        assert request(base, "/api/arms/right/grip-torque", {"torque_nm": 0.7})[0] == 200
        left.power(True)
        right.power(True)
        assert request(f"http://127.0.0.1:{left.port}", "/target", {"q": [0.1] * 6, "grip": 0.8})[0] == 200
        wait_for(lambda: lrobot.torque == 1.4 and rrobot.torque == 0.7)
        assert lrobot.torque == 1.4 and rrobot.torque == 0.7
        assert request(base, "/api/arms/left/grip-torque", {"torque_nm": 1.7})[0] == 200
        wait_for(lambda: lrobot.torque == 1.7)
        assert rrobot.torque == 0.7
        assert request(base, "/api/arms/left/grip-torque", {"torque_nm": 3})[0] == 400
    finally:
        page.close()
        for node in (left, right):
            node.power(False)
            node.close()
