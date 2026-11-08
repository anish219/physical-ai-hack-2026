"""Local wrist-image alignment for the USB bundle demo. No camera or robot ownership."""

import cv2
import numpy as np


class BundleVision:
    """Track the strip plane and the orange wires near their seated connectors.

    All coordinates use the reference image size. Four projected strip anchors
    constrain perspective as well as translation; the wire centroid checks the
    bundle independently. This is a local matcher for an already aimed wrist.
    """

    def __init__(self, reference, settings):
        self.reference = reference
        self.settings = settings
        self.gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.polygon = np.asarray(settings["strip_polygon"], np.float32)
        mask = np.zeros(self.gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, self.polygon.astype(np.int32), 255)
        self.points = cv2.goodFeaturesToTrack(self.gray, 100, 0.015, 5, mask=mask)
        if self.points is None or len(self.points) < 8:
            raise ValueError("Reference needs at least eight visible strip features")
        self.anchors = self.polygon.reshape(-1, 1, 2)
        self.wire_center = self._wire_center(reference, np.eye(3))
        self.target = np.r_[self.polygon.ravel(), self.wire_center]

    def _wire_center(self, frame, homography):
        x, y, w, h = self.settings["wire_roi"]
        corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        warped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography)
        roi = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(roi, np.rint(warped).astype(np.int32), 255)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, np.array(self.settings["hsv_low"]), np.array(self.settings["hsv_high"]))
        orange = cv2.bitwise_and(orange, roi)
        moments = cv2.moments(orange, binaryImage=True)
        if moments["m00"] < self.settings["minimum_wire_pixels"]:
            raise ValueError("Orange bundle is not visible in the connector region")
        return np.array([moments["m10"], moments["m01"]]) / moments["m00"]

    def measure(self, image):
        frame = cv2.resize(image, self.reference.shape[1::-1])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        options = dict(winSize=(25, 25), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
        live, found, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **options)
        back, returned, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, live, None, **options)
        valid = found.ravel().astype(bool) & returned.ravel().astype(bool)
        valid &= np.linalg.norm(back - self.points, axis=2).ravel() < 1.0
        if valid.sum() < 8:
            raise ValueError("Too few matching strip features at the aim checkpoint")
        homography, inliers = cv2.findHomography(self.points[valid], live[valid], cv2.RANSAC, 1.5)
        if homography is None or inliers.sum() < 8:
            raise ValueError("Cannot locate the strip plane at the aim checkpoint")
        anchors = cv2.perspectiveTransform(self.anchors, homography).reshape(-1, 2)
        wire = self._wire_center(frame, homography)
        features = np.r_[anchors.ravel(), wire]
        delta = (features - self.target).reshape(-1, 2)
        errors = np.linalg.norm(delta, axis=1)
        report = {"strip_error_px": float(errors[:4].max()), "wire_error_px": float(errors[4]),
                  "matched_features": int(inliers.sum()), "features": features.tolist()}
        report["aligned"] = (report["strip_error_px"] <= self.settings["strip_tolerance_px"]
                             and report["wire_error_px"] <= self.settings["wire_tolerance_px"])
        for target, actual in zip(self.target.reshape(-1, 2), features.reshape(-1, 2), strict=True):
            cv2.circle(frame, tuple(np.rint(target).astype(int)), 4, (0, 255, 0), 1)
            cv2.circle(frame, tuple(np.rint(actual).astype(int)), 3, (0, 0, 255), 1)
        return features, report, frame


def servo(arm, vision, move, settings, report):
    """Learn the image/joint Jacobian at the checkpoint, then minimize image error.

    Probes are explicit small joint motions, not a preinstalled camera calibration.
    Every image is acquired after settling, and every Jacobian column uses measured
    joint displacement. Broyden updates follow the observed response to corrections.
    """
    q, grip = arm.state()
    features, status, image = vision.measure(arm.image())
    report("align", status, image)
    if status["aligned"]:
        return q
    columns = []
    probe = settings["probe_rad"]
    for joint in range(len(q)):
        plus, minus = q.copy(), q.copy()
        plus[joint] += probe
        minus[joint] -= probe
        samples = []
        positions = []
        for target in (plus, minus):
            move(target, grip)
            samples.append(vision.measure(arm.image())[0])
            positions.append(arm.state()[0][joint])
        displacement = positions[0] - positions[1]
        if abs(displacement) < probe * 0.1:
            raise ValueError(f"Joint {joint} did not move enough to measure the image Jacobian")
        columns.append((samples[0] - samples[1]) / displacement)
        move(q, grip)
    jacobian = np.column_stack(columns)
    q, _ = arm.state()
    features, status, image = vision.measure(arm.image())
    while not status["aligned"]:
        error = vision.target - features
        damping = settings["damping"]
        delta = np.linalg.solve(jacobian.T @ jacobian + damping**2 * np.eye(len(q)), jacobian.T @ error)
        delta *= settings["gain"]
        delta *= min(1.0, settings["step_rad"] / max(float(np.max(np.abs(delta))), 1e-12))
        move(q + delta, grip)
        actual, _ = arm.state()
        updated, status, image = vision.measure(arm.image())
        dq = actual - q
        if dq @ dq > 1e-12:
            jacobian += np.outer(updated - features - jacobian @ dq, dq) / (dq @ dq)
        q, features = actual, updated
        report("align", status, image)
    return q
