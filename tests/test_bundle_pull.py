"""Offline vision perturbations, visual servo, rigid retargeting, and replay sequencing."""

import json
import pathlib

import cv2
import numpy as np
import pytest

from dream_hackathon.bundle_pull import (
    HttpArm,
    ToolKinematics,
    load_plan,
    play,
    retarget,
    run,
    sample,
    section,
    shortcut,
)
from dream_hackathon.bundle_vision import servo

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN = ROOT / "recordings/usb-bundle-pull/plan.json"


def test_real_reference_and_small_perspective_displacement():
    _, _, vision = load_plan(PLAN)
    _, report, _ = vision.measure(vision.reference)
    assert report["aligned"]
    homography = np.array([[1.01, 0.008, 4], [-0.005, 1.005, -3], [0.00001, -0.00002, 1]])
    warped = cv2.warpPerspective(vision.reference, homography, vision.reference.shape[1::-1])
    features, report, _ = vision.measure(warped)
    expected = cv2.perspectiveTransform(vision.anchors, homography).reshape(-1, 2)
    np.testing.assert_allclose(features[:8].reshape(-1, 2), expected, atol=1.1)
    assert not report["aligned"]
    assert report["strip_error_px"] > 4


def test_background_people_do_not_determine_strip_alignment():
    _, _, vision = load_plan(PLAN)
    changed = vision.reference.copy()
    changed[:50] = np.random.default_rng(123).integers(0, 255, changed[:50].shape, dtype=np.uint8)
    _, status, _ = vision.measure(changed)
    assert status["aligned"]


def test_missing_orange_bundle_is_not_reported_aligned():
    _, _, vision = load_plan(PLAN)
    gray = cv2.cvtColor(cv2.cvtColor(vision.reference, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    with pytest.raises(ValueError, match="Orange bundle"):
        vision.measure(gray)


class FakeArm:
    def __init__(self, q=None):
        self.q = np.zeros(6) if q is None else np.asarray(q, float).copy()
        self.grip = 0.44
        self.commands = []

    def state(self):
        return self.q.copy(), self.grip

    def command(self, q, grip):
        self.q, self.grip = np.asarray(q).copy(), float(grip)
        self.commands.append((self.q.copy(), self.grip))

    def image(self):
        return self.q.copy()


def test_servo_calibrates_unknown_image_response_and_converges():
    arm = FakeArm([0.04, -0.03, 0.02, 0.015, -0.02, 0.03])
    matrix = np.random.default_rng(17).normal(size=(10, 6)) * 120
    matrix[:6] += np.eye(6) * 200

    class Vision:
        target = np.zeros(10)

        def measure(self, q):
            features = matrix @ q
            return features, {"aligned": bool(np.max(np.abs(features)) < 0.05)}, None

    plan, _, _ = load_plan(PLAN)
    observations = []
    aligned = servo(arm, Vision(), arm.command, plan["servo"], lambda *args: observations.append(args))
    assert np.max(np.abs(matrix @ aligned)) < 0.05
    assert len(arm.commands) > 18  # plus/minus probes, return, and actual corrections
    assert observations[-1][1]["aligned"]
    assert arm.grip == 0.44


def test_already_aligned_servo_does_not_probe():
    arm = FakeArm()

    class Vision:
        def measure(self, _):
            return np.zeros(10), {"aligned": True}, None

    plan, _, _ = load_plan(PLAN)
    servo(arm, Vision(), arm.command, plan["servo"], lambda *args: None)
    assert arm.commands == []


def test_servo_closes_loop_using_warped_recording_images():
    plan, _, vision = load_plan(PLAN)

    class ImageArm(FakeArm):
        def image(self):
            x, y, scale, shear, tilt_x, tilt_y = self.q
            homography = np.array(
                [[1 + scale, shear, x * 140], [-shear, 1 + scale, y * 140], [tilt_x * 0.004, tilt_y * 0.004, 1]]
            )
            return cv2.warpPerspective(vision.reference, homography, vision.reference.shape[1::-1])

    arm = ImageArm([0.035, -0.025, 0.006, 0.004, 0.002, -0.002])
    observations = []

    def report(_phase, status, _image):
        observations.append(status)
        assert len(observations) < 40, "Image-based simulation did not converge"

    servo(arm, vision, arm.command, plan["servo"], report)
    assert not observations[0]["aligned"]
    assert observations[-1]["aligned"]


def cartesian_fk(q):
    pose = np.eye(4)
    pose[:3, 3] = q[:3]
    pose[:3, :3] = cv2.Rodrigues(np.asarray(q[3:]))[0]
    return pose


def test_rigid_retarget_keeps_tool_path_and_grip_in_aligned_frame():
    kin = ToolKinematics(cartesian_fk)
    aim = np.array([0.2, -0.1, 0.3, 0.1, 0.2, -0.1])
    aligned = aim + [0.01, -0.02, 0.03, 0.04, 0, 0.02]
    rows = [
        {
            "t": i / 30,
            "q": (aim + np.array([i * 0.005, 0, i * 0.003, 0, 0, 0])).tolist(),
            "grip": max(0, 0.44 - i * 0.1),
        }
        for i in range(5)
    ]
    result = retarget(rows, aim, aligned, kin)
    correction = cartesian_fk(aligned) @ np.linalg.inv(cartesian_fk(aim))
    for before, after in zip(rows, result, strict=True):
        np.testing.assert_allclose(cartesian_fk(after["q"]), correction @ cartesian_fk(before["q"]), atol=1e-5)
        assert before["t"] == after["t"] and before["grip"] == after["grip"]
    np.testing.assert_allclose(result[0]["q"], aligned, atol=1e-5)


def test_fk_uses_real_model_and_joint_signs(urdf_path, monkeypatch):
    from types import SimpleNamespace

    from dream_hackathon import config

    mm = SimpleNamespace(
        urdf=urdf_path,
        flange_link="link6",
        urdf_joints=[f"j{i}" for i in range(1, 7)],
        joint_signs=[1, -1, 1, 1, 1, -1],
    )
    station = SimpleNamespace(makermods=mm, arms={"right": SimpleNamespace(tcp_xyz_m=(0.135, 0, 0))})
    monkeypatch.setattr(config, "load", lambda _: station)
    kin = ToolKinematics.from_config("unused", "right")
    q = np.array([0.2, -0.8, 0.4, 0.3, -0.2, 0.1])
    target = kin.fk(q).copy()
    target[:3, 3] += [0.002, 0.001, -0.001]
    solved = kin.solve(target, q)
    np.testing.assert_allclose(kin.fk(solved), target, atol=1e-4)


def test_shortcut_returns_to_exact_original_drop_and_preserves_release():
    plan, rows, _ = load_plan(PLAN)
    last = [r for r in rows if r["segment"] == "recorded pull"][-1]
    corrected = {**last, "q": (np.asarray(last["q"]) + 0.06).tolist()}
    drop = shortcut(rows, corrected, plan)
    assert drop[0]["q"] == corrected["q"]
    assert drop[1]["q"] == rows[-1]["q"]
    assert drop[1]["grip"] == last["grip"]
    assert drop[-1]["q"] == rows[-1]["q"] and drop[-1]["grip"] == rows[-1]["grip"]
    assert all(a["t"] < b["t"] for a, b in zip(drop, drop[1:], strict=False))


def test_full_sequence_aims_before_grasp_and_retargets_before_shortcut(monkeypatch):
    plan, rows, vision = load_plan(PLAN)
    monkeypatch.setattr("dream_hackathon.bundle_pull.time.sleep", lambda _: None)
    arm = FakeArm()
    phases, played = [], []
    correction = np.array([0.001, -0.002, 0.001, 0.001, 0.001, 0.001])

    def play_fn(arm, targets, **_):
        played.append(targets)
        arm.command(targets[-1]["q"], targets[-1]["grip"])

    def move_fn(arm, q, grip, *args, **kwargs):
        arm.command(q, grip)

    def align(arm, *_):
        q, grip = arm.state()
        arm.command(q + correction, grip)
        return arm.q

    run(
        arm,
        rows,
        plan,
        vision,
        ToolKinematics(cartesian_fk),
        lambda phase, *_: phases.append(phase),
        play_fn=play_fn,
        move_fn=move_fn,
        servo_fn=align,
    )
    assert phases == ["start", "approach", "grasp-pull", "shortcut-drop", "complete"]
    assert len(played) == 3
    np.testing.assert_allclose(played[1][0]["q"], np.asarray(played[0][-1]["q"]) + correction, atol=1e-5)
    assert played[1][0]["grip"] == played[0][-1]["grip"]
    assert played[1][-1]["grip"] == 0
    assert played[2][0]["q"] == played[1][-1]["q"]
    np.testing.assert_allclose(arm.q, rows[-1]["q"])
    assert arm.grip == rows[-1]["grip"]


def test_wall_clock_playback_interpolates_and_finishes_at_exact_endpoint():
    arm = FakeArm()
    now = [0.0]

    def sleep(seconds):
        now[0] += max(seconds, 1e-6)

    rows = [{"t": 0, "q": [0] * 6, "grip": 0.4}, {"t": 0.12, "q": [0.12] * 6, "grip": 0}]
    play(arm, rows, hz=30, clock=lambda: now[0], sleep=sleep)
    np.testing.assert_allclose(arm.commands[1][0], np.full(6, 1 / 30))
    np.testing.assert_allclose(arm.q, rows[-1]["q"])
    assert arm.grip == 0


def test_checkpoint_interpolates_both_sides_without_timing_jump():
    _, rows, _ = load_plan(PLAN)
    before, after = section(rows, 0, 7), section(rows, 7, 14.975756715997704)
    assert before[-1]["q"] == after[0]["q"]
    assert before[-1]["grip"] == after[0]["grip"]
    np.testing.assert_allclose(sample(rows, 7)[0], before[-1]["q"])


def test_cli_inspection_has_no_network_requests(monkeypatch, tmp_path):
    from dream_hackathon.bundle_pull import main

    monkeypatch.setattr("sys.argv", ["replay_usb_bundle", str(PLAN), "--output", str(tmp_path)])
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: pytest.fail("Inspection made a network request"))
    main()
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["phase"] == "inspect" and event["reference"]["aligned"]


def test_http_transport_uses_only_state_wrist_and_target(monkeypatch):
    import io

    _, _, vision = load_plan(PLAN)
    _, encoded = cv2.imencode(".jpg", vision.reference)
    requests = []
    stamps = iter(["10", "10", "11"])

    def open_url(request, **_):
        url = request if isinstance(request, str) else request.full_url
        requests.append(request)
        if url.endswith("/state"):
            payload, headers = json.dumps({"q": [0.1] * 6, "grip": 0.44}).encode(), {}
        elif url.endswith("/wrist.jpg"):
            payload, headers = encoded.tobytes(), {"X-Stamp": next(stamps)}
        elif url.endswith("/target"):
            payload, headers = b'{"ok":true}', {}
        else:
            pytest.fail(f"Unexpected endpoint {url}")
        response = io.BytesIO(payload)
        response.headers = headers
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    monkeypatch.setattr("dream_hackathon.bundle_pull.time.sleep", lambda _: None)
    arm = HttpArm("http://local-arm/")
    q, grip = arm.state()
    arm.command(q, grip)
    assert arm.image().shape == vision.reference.shape
    assert arm.image().shape == vision.reference.shape
    assert arm.last_stamp == "11"
    assert len(requests) == 5
    assert json.loads(requests[1].data) == {"q": [0.1] * 6, "grip": 0.44}
