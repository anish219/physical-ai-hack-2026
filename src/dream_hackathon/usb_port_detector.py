"""V2 port observations using the existing CV core, with no stage/pass latch."""

import importlib
import math
import pathlib
import sys
from dataclasses import asdict

DEFAULT_SCRIPTS = "/home/pah/usb-geometry-verifier-20261107/usb-geometry-verifier-v1/scripts"


def load_cv_detector(scripts_path=DEFAULT_SCRIPTS):
    """Keep calibration/assets in their existing installation; never modify V1."""
    scripts = pathlib.Path(scripts_path).expanduser().resolve()
    if not (scripts / "geometry_only_usb_verifier.py").is_file():
        raise ValueError(f"USB CV installation missing: {scripts}")
    sys.path.insert(0, str(scripts))
    detector_module = importlib.import_module("geometry_only_usb_verifier")
    camera_module = importlib.import_module("usb_camera_shadow")
    for module in (detector_module, camera_module):
        if pathlib.Path(module.__file__).resolve().parent != scripts:
            raise ValueError(f"{module.__name__} already loaded from another installation")
    return detector_module.GeometryOnlyVerifier(), camera_module.fetch_pair


class UsbPortDetector:
    """Current calibrated port occupancy. Unknown counts are null, never guessed.

    Input stamps use the same monotonic clock as now. The installed HTTP helper
    converts genuine same-host X-Stamp capture times and constructs capture IDs.
    No baseline, removal event, policy permission or manipulation success is inferred.
    """

    def __init__(
        self, run_id, *, detector=None, scripts_path=DEFAULT_SCRIPTS, maximum_age=2.0, maximum_camera_skew=0.15
    ):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("A nonempty run_id is required")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (maximum_age, maximum_camera_skew)
        ):
            raise ValueError("Capture age/skew limits must be finite and positive")
        self.detector = detector if detector is not None else load_cv_detector(scripts_path)[0]
        self.run_id, self.maximum_age, self.maximum_camera_skew = run_id, maximum_age, maximum_camera_skew
        self.last_capture, self.seen_ids = {}, set()

    def _unknown(self, reason, stamps=(), detection=None):
        return dict(
            schema_version=2,
            run_id=self.run_id,
            occupancy_known=False,
            ports=[{"port_id": port, "state": "unknown"} for port in range(4)],
            occupied_count=None,
            open_count=None,
            unknown_count=4,
            reason=reason,
            captures=[asdict(s) for s in stamps],
            detection=detection,
            capabilities={
                "port_occupancy": True,
                "full_insertion": False,
                "gripper_release": False,
                "cable_on_table": False,
            },
        )

    def observe(self, *, rgb_scene=None, rgb_wrist=None, stamps, now):
        stamps = list(stamps)
        images = {c: value for c, value in (("scene", rgb_scene), ("wrist", rgb_wrist)) if value is not None}
        result = self._unknown("unresolved", stamps)
        if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
            return dict(result, reason="invalid_observation_time")
        if not images or len(stamps) != len(images) or {s.camera for s in stamps} != set(images):
            return dict(result, reason="image_capture_metadata_mismatch")
        for stamp in stamps:
            if stamp.run_id != self.run_id:
                return dict(result, reason="wrong_run_id")
            if not isinstance(stamp.frame_id, str) or not stamp.frame_id:
                return dict(result, reason="missing_capture_id")
            if (
                type(stamp.captured_at) not in (int, float)
                or not math.isfinite(stamp.captured_at)
                or not 0 <= now - stamp.captured_at <= self.maximum_age
            ):
                return dict(result, reason="stale_or_future_capture")
            if (
                stamp.captured_at <= self.last_capture.get(stamp.camera, -1)
                or (stamp.camera, stamp.frame_id) in self.seen_ids
            ):
                return dict(result, reason="repeated_or_out_of_order_capture")
        if max(s.captured_at for s in stamps) - min(s.captured_at for s in stamps) > self.maximum_camera_skew:
            return dict(result, reason="camera_capture_skew")
        for stamp in stamps:
            self.last_capture[stamp.camera] = stamp.captured_at
            self.seen_ids.add((stamp.camera, stamp.frame_id))
        # Reject malformed image arrays before invoking even a substituted detector.
        import numpy as np

        if any(
            not isinstance(rgb, np.ndarray) or rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8
            for rgb in images.values()
        ):
            return dict(result, reason="expected_rgb_uint8_640x360")
        try:
            detection = self.detector.detect(rgb_scene=rgb_scene, rgb_wrist=rgb_wrist)
        except Exception as error:
            return dict(result, reason="detector_failed", error_type=type(error).__name__)
        # Reconcile each calibrated physical port; a disagreement is unknown.
        ports = []
        for port in range(4):
            states = set()
            for camera in images:
                view = detection.get("cameras", {}).get(camera, {})
                observations = view.get("ports", [])
                if (
                    view.get("geometry_valid") is True
                    and isinstance(observations, (list, tuple))
                    and len(observations) == 4
                    and observations[port] in ("open", "occupied")
                ):
                    states.add(observations[port])
            state = next(iter(states)) if len(states) == 1 else "unknown"
            ports.append({"port_id": port, "state": state})
        states = [port["state"] for port in ports]
        known = "unknown" not in states
        return dict(
            result,
            ports=ports,
            occupancy_known=known,
            occupied_count=states.count("occupied") if known else None,
            open_count=states.count("open") if known else None,
            unknown_count=states.count("unknown"),
            reason="current_socket_geometry" if known else "occluded_ambiguous_or_conflicting_ports",
            detection=detection,
        )
