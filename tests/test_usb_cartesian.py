"""World-frame motion with fresh visual feedback, decoupled depth and moving-port corrections."""

import cv2
import numpy as np
import pytest

from dream_hackathon import usb_cartesian as control

J = np.array([[1000, 200, 300], [-100, 800, 600], [500, -300, -500], [400, 900, -200],
              [-700, 600, 400], [300, 400, -900], [-200, 800, 100], [700, -100, 200]], float)
SETTINGS = {'distance_m': 0.004, 'speed_m_s': 0.0015, 'probe_m': 0.0006,
            'lateral_gain': 0.3, 'lateral_step_m': 0.001, 'adapt_gain': 0,
            'motion_rad_s': 0.15, 'settle_s': 0, 'hz': 20, 'depth_tolerance_m': 0.0001}


class Kinematics:
    def fk(self, q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = cv2.Rodrigues(q[3:])[0]
        return pose

    def solve(self, pose, seed):
        return np.r_[pose[:3, 3], cv2.Rodrigues(pose[:3, :3])[0].ravel()]


class Arm:
    def __init__(self):
        self.q = np.array([0., 0., 0., 0., 0.6, 0.2])
        self.commands = []
        self.images = 0

    def state(self):
        return self.q.copy(), 0.12

    def image(self):
        self.images += 1
        return self.q.copy(), self.images

    def command(self, q, grip):
        self.q = q.copy()
        self.commands.append((q.copy(), grip, self.images))


class Vision:
    target = np.zeros(8)

    def measure(self, image):
        q, index = image
        drift = np.array([0.0015, -0.0007, 0]) if index > 10 else np.zeros(3)
        return J @ (q[:3] + drift), {'known': True}, None


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_depth_image_flow_does_not_become_a_false_lateral_correction():
    correction = control.lateral_correction(J, -J[:, 2] * 0.02, 1, 1)
    np.testing.assert_allclose(correction, [0, 0], atol=1e-12)


@pytest.mark.parametrize('adapt_gain', [0, 0.1])
def test_descent_uses_every_frame_corrects_moving_port_and_preserves_world_orientation(adapt_gain):
    arm, clock = Arm(), Clock()
    original_orientation = Kinematics().fk(arm.q)[:3, :3]
    result = control.descend(arm, Vision(), Kinematics(), {**SETTINGS, 'adapt_gain': adapt_gain},
                             0.12, lambda *args: None,
                             jacobian=J.copy(), clock=clock, sleep=clock.sleep)
    assert result['descent_complete'] and len(arm.commands) > 30
    assert arm.q[2] == pytest.approx(-SETTINGS['distance_m'])
    np.testing.assert_allclose(arm.q[:2], [-0.0015, 0.0007], atol=2e-8)
    z = [q[2] for q, grip, image in arm.commands]
    assert np.all(np.diff(z) <= 0)
    assert len({image for q, grip, image in arm.commands}) == len(arm.commands)
    for q, grip, _image in arm.commands:
        assert grip == 0.12
        np.testing.assert_allclose(Kinematics().fk(q)[:3, :3], original_orientation, atol=1e-12)


def test_world_probe_calibration_uses_measured_cartesian_motion(monkeypatch):
    arm = Arm()

    class StableVision:
        def measure(self, image):
            return J @ image[0][:3], {'known': True}, None

    monkeypatch.setattr(control, 'move_to', lambda arm, q, grip, speed, **kwargs: arm.command(q, grip))
    jacobian = control.calibrate(arm, StableVision(), Kinematics(), SETTINGS, 0.12, lambda *args: None)
    np.testing.assert_allclose(jacobian, J, atol=1e-9)
    np.testing.assert_allclose(arm.q[:3], [0, 0, 0])


def test_missing_localization_does_not_produce_a_target_for_that_frame():
    arm, clock = Arm(), Clock()

    class LostVision(Vision):
        def measure(self, image):
            if image[1] == 4:
                raise ValueError('USB tip occluded')
            return super().measure(image)

    with pytest.raises(ValueError, match='occluded'):
        control.descend(arm, LostVision(), Kinematics(), SETTINGS, 0.12, lambda *args: None,
                        jacobian=J.copy(), clock=clock, sleep=clock.sleep)
    assert len(arm.commands) == 2
    assert arm.commands[-1][2] == 3
