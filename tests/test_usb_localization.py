"""Port reacquisition across viewpoints and USB localization outside the old reference crop."""

import json
import pathlib

import cv2
import numpy as np
import pytest

from dream_hackathon.usb_insertion import InsertionVision
from dream_hackathon.usb_localization import HeldUSB, PortRegistration

ASSETS = pathlib.Path(__file__).resolve().parents[1] / 'recordings/usb-insertion'


@pytest.fixture(scope='module')
def model():
    plan = json.loads((ASSETS / 'plan.json').read_text())
    reference = cv2.imread(str(ASSETS / plan['reference']))
    return reference, plan, PortRegistration(reference, plan['vision'])


@pytest.mark.parametrize('angle', [-45, 30, 60])
def test_reacquires_rotated_scaled_port_without_optical_flow_initialization(model, angle):
    reference, plan, registration = model
    transform = np.eye(3)
    transform[:2] = cv2.getRotationMatrix2D((320, 180), angle, 0.9)
    frame = cv2.warpPerspective(reference, transform, (640, 360))
    _, port, status = registration.locate(frame)
    expected = cv2.perspectiveTransform(np.float32(plan['vision']['port_point']).reshape(1, 1, 2), transform)[0, 0]
    np.testing.assert_allclose(port, expected, atol=2)
    assert status['matched_features'] >= plan['vision']['minimum_port_inliers']


def test_tracks_port_and_plug_when_held_body_is_outside_old_small_roi(model):
    reference, plan, _ = model
    vision = InsertionVision(reference, plan['vision'])
    shifted = cv2.warpAffine(reference, np.float32([[1, 0, 32], [0, 1, 25]]), (640, 360))
    _, status, _ = vision.measure(shifted)
    assert status['near_opening'] and status['aligned']
    assert status['plug_center'][1] > plan['vision']['plug_roi'][1] + plan['vision']['plug_roi'][3]


def test_loose_cables_and_gripper_cable_tail_are_not_a_held_connector(model):
    reference, plan, _ = model
    image = reference.copy()
    x, y, w, h = plan['vision']['plug_roi']
    image[y:y+h, x:x+w] = cv2.cvtColor(cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY),
                                     cv2.COLOR_GRAY2BGR)
    with pytest.raises(ValueError, match='Held USB'):
        HeldUSB(plan['vision']).locate(image)


def test_real_multiview_bank_keeps_far_and_near_observations_distinct(model):
    reference, plan, _ = model
    views = [cv2.imread(str(ASSETS / name)) for name in plan['view_images']]
    vision = InsertionVision(reference, plan['vision'], views)
    _, far, _ = vision.measure(cv2.imread(str(ASSETS / 'far-frame-420.png')))
    _, near, _ = vision.measure(cv2.imread(str(ASSETS / 'before-frame-450.png')))
    _, changed_view, _ = vision.measure(cv2.imread(str(ASSETS / 'view-frame-600.png')))
    assert not far['near_opening']
    assert near['near_opening'] and changed_view['near_opening']


def test_blank_view_does_not_produce_a_port(model):
    reference, _plan, registration = model
    with pytest.raises(ValueError, match='port'):
        registration.locate(np.full_like(reference, 180))
