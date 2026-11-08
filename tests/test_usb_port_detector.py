from dataclasses import dataclass, replace

import numpy as np
import pytest

from dream_hackathon.usb_port_detector import UsbPortDetector


@dataclass(frozen=True)
class Stamp:
    camera: str
    frame_id: str
    captured_at: float
    run_id: str = "run"


class Detector:
    def __init__(self, scene=None, wrist=None):
        self.scene = ["occupied"] * 4 if scene is None else scene
        self.wrist = self.scene if wrist is None else wrist
        self.valid = True

    def detect(self, **kwargs):
        return {
            "cameras": {
                camera: {"ports": getattr(self, camera), "geometry_valid": self.valid}
                for camera in ("scene", "wrist")
                if kwargs.get("rgb_" + camera) is not None
            }
        }


RGB = np.zeros((360, 640, 3), dtype=np.uint8)


def observe(adapter, timestamp=1.0, **kwargs):
    stamps = kwargs.pop("stamps", [Stamp("scene", str(timestamp), timestamp)])
    return adapter.observe(
        rgb_scene=kwargs.pop("rgb_scene", RGB), stamps=stamps, now=kwargs.pop("now", timestamp + 0.01), **kwargs
    )


def test_current_ports_do_not_latch_or_grant_stage_permission():
    detector = Detector()
    adapter = UsbPortDetector("run", detector=detector)
    assert observe(adapter)["occupied_count"] == 4
    detector.scene = ["open", "occupied", "open", "occupied"]
    result = observe(adapter, 2.0)
    assert result["occupancy_known"] is True
    assert result["open_count"] == 2
    assert result["ports"][0] == {"port_id": 0, "state": "open"}
    assert "stage_passed" not in result
    assert not result["capabilities"]["gripper_release"]
    detector.scene = ["occupied"] * 4
    assert observe(adapter, 3.0)["open_count"] == 0


def test_partial_visibility_preserves_port_observations_but_not_exact_counts():
    adapter = UsbPortDetector("run", detector=Detector(["open", "unknown", "occupied", "unknown"]))
    result = observe(adapter)
    assert result["ports"][0]["state"] == "open"
    assert not result["occupancy_known"]
    assert result["occupied_count"] is None and result["open_count"] is None
    assert result["unknown_count"] == 2


def test_conflicting_camera_evidence_is_unknown_for_that_port():
    adapter = UsbPortDetector("run", detector=Detector(["open"] * 4, ["occupied"] * 4))
    result = observe(adapter, rgb_wrist=RGB, stamps=[Stamp("scene", "s", 1), Stamp("wrist", "w", 1)])
    assert result["unknown_count"] == 4
    assert result["open_count"] is None


@pytest.mark.parametrize(
    "change",
    [{"run_id": "old"}, {"frame_id": ""}, {"captured_at": -1}, {"captured_at": float("nan")}, {"captured_at": 2.0}],
)
def test_bad_metadata_never_counts_ports(change):
    adapter = UsbPortDetector("run", detector=Detector(["open"] * 4))
    stamp = replace(Stamp("scene", "capture", 1), **change)
    result = observe(adapter, stamps=[stamp])
    assert not result["occupancy_known"]
    assert result["open_count"] is None


def test_reused_capture_id_even_with_a_new_time_is_rejected():
    adapter = UsbPortDetector("run", detector=Detector(["open"] * 4))
    observe(adapter, stamps=[Stamp("scene", "a", 1)])
    observe(adapter, 2.0, stamps=[Stamp("scene", "b", 2)])
    result = observe(adapter, 3.0, stamps=[Stamp("scene", "a", 3)])
    assert result["reason"] == "repeated_or_out_of_order_capture"
    assert result["open_count"] is None


def test_geometry_invalid_and_wrong_image_size_are_unknown():
    detector = Detector(["open"] * 4)
    detector.valid = False
    adapter = UsbPortDetector("run", detector=detector)
    assert observe(adapter)["unknown_count"] == 4
    result = observe(adapter, 2.0, rgb_scene=np.zeros((240, 426, 3), dtype=np.uint8))
    assert result["reason"] == "expected_rgb_uint8_640x360"


def test_camera_skew_and_stale_images_rejected():
    adapter = UsbPortDetector("run", detector=Detector(["open"] * 4))
    result = observe(adapter, rgb_wrist=RGB, stamps=[Stamp("scene", "s", 1), Stamp("wrist", "w", 0.7)])
    assert result["reason"] == "camera_capture_skew"
    assert observe(adapter, now=5.0)["reason"] == "stale_or_future_capture"


def test_detector_failure_is_unknown_and_never_reuses_previous_counts():
    detector = Detector()
    adapter = UsbPortDetector("run", detector=detector)
    assert observe(adapter)["occupancy_known"]

    def broken(**kwargs):
        raise ValueError("registration failed")

    detector.detect = broken
    result = observe(adapter, 2.0)
    assert result["reason"] == "detector_failed"
    assert result["occupied_count"] is None
