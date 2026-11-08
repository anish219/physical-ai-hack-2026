"""Viewpoint-tolerant port registration and full-image, gripper-associated USB localization."""

import cv2
import numpy as np


class PortRegistration:
    """Reacquire the labeled port face using affine-view SIFT descriptors.

    Each descriptor retains its point in the original port coordinate system.
    Matching is independent per image; no previous pose or optical-flow window
    is needed. Actual additional views can extend the same object model.
    """

    def __init__(self, reference, settings):
        self.settings = settings
        self.polygon = np.float32(settings['port_polygon'])
        self.sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.01, edgeThreshold=15)
        mask = np.zeros(reference.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, self.polygon.astype(np.int32), 255)
        x, y, w, h = settings['plug_roi']
        mask[y:y+h, x:x+w] = 0
        self.reference_mask = mask
        points, descriptors = [], []
        center = self.polygon.mean(axis=0)
        for tilt in settings.get('affine_tilts', [1.0, 0.8, 0.6, 0.4]):
            for angle in ([0] if tilt == 1 else range(0, 180, 30)):
                angle = np.deg2rad(angle)
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                affine = rotation @ np.diag([1, tilt]) @ rotation.T
                transform = np.eye(3)
                transform[:2, :2] = affine
                transform[:2, 2] = center - affine @ center
                image = cv2.warpPerspective(reference, transform, reference.shape[1::-1])
                projected_mask = cv2.warpPerspective(mask, transform, reference.shape[1::-1])
                keys, desc = self.sift.detectAndCompute(image, projected_mask)
                if desc is not None:
                    canonical = cv2.perspectiveTransform(
                        np.float32([key.pt for key in keys]).reshape(-1, 1, 2), np.linalg.inv(transform))
                    points.extend(canonical.reshape(-1, 2))
                    descriptors.extend(desc)
        if len(points) < 6:
            raise ValueError('Reference port face needs six distinct SIFT features')
        self.points, self.descriptors = np.float32(points), np.float32(descriptors)

    def locate(self, frame):
        keys, descriptors = self.sift.detectAndCompute(frame, None)
        if descriptors is None or len(descriptors) < 2:
            raise ValueError('No port features visible')
        neighbors = cv2.BFMatcher().knnMatch(self.descriptors, descriptors, k=2)
        matches = [pair[0] for pair in neighbors if len(pair) == 2
                   and pair[0].distance < self.settings.get('descriptor_ratio', 0.75) * pair[1].distance]
        # Affine copies of one physical feature are one observation, not votes.
        unique = {}
        for match in sorted(matches, key=lambda value: value.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        minimum = self.settings.get('minimum_port_inliers', 6)
        if len(matches) < minimum:
            raise ValueError(f'Port not localized: {len(matches)} independent feature matches, need {minimum}')
        source = np.float32([self.points[m.queryIdx] for m in matches])
        target = np.float32([keys[m.trainIdx].pt for m in matches])
        homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
        if homography is None or inliers is None or int(inliers.sum()) < minimum:
            raise ValueError('Port matches do not agree on a face geometry')
        inliers = inliers.ravel().astype(bool)
        coverage = cv2.contourArea(cv2.convexHull(source[inliers])) / cv2.contourArea(self.polygon)
        anchors = cv2.perspectiveTransform(self.polygon.reshape(-1, 1, 2), homography).reshape(-1, 2)
        if (not np.isfinite(anchors).all() or not cv2.isContourConvex(anchors.astype(np.float32))
                or cv2.contourArea(anchors) < 30 or coverage < self.settings.get('minimum_port_coverage', 0.08)):
            raise ValueError('Port matches do not span a visible port face')
        # A reflected ordering indicates a mismatched object face, not camera rotation.
        if cv2.contourArea(anchors, oriented=True) * cv2.contourArea(self.polygon, oriented=True) <= 0:
            raise ValueError('Port matches reverse the labeled face')
        port = cv2.perspectiveTransform(np.float32(self.settings['port_point']).reshape(1, 1, 2), homography)[0, 0]
        return anchors, port, {'matched_features': int(inliers.sum()), 'port_coverage': float(coverage),
                               'localizer': 'affine_sift'}

    def add_view(self, frame):
        """Extend with a registered real view; labels remain in canonical port coordinates."""
        anchors, _port, _status = self.locate(frame)
        canonical_to_view = cv2.getPerspectiveTransform(self.polygon, anchors.astype(np.float32))
        mask = cv2.warpPerspective(self.reference_mask, canonical_to_view, frame.shape[1::-1])
        keys, descriptors = self.sift.detectAndCompute(frame, mask)
        if descriptors is None:
            return
        points = cv2.perspectiveTransform(np.float32([key.pt for key in keys]).reshape(-1, 1, 2),
                                         np.linalg.inv(canonical_to_view)).reshape(-1, 2)
        self.points = np.vstack([self.points, points])
        self.descriptors = np.vstack([self.descriptors, descriptors])


class HeldUSB:
    """Find the orange USB body across the image, associated with the wrist gripper.

    Connector position, axis and size come from the live component. Thin cable
    segments and remote loose connectors are not treated as a held USB body.
    The search is not clipped to a recording's small fixed pixel rectangle.
    """

    def __init__(self, settings):
        self.settings = settings

    def locate(self, frame):
        height, width = frame.shape[:2]
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                           np.array(self.settings['hsv_low']), np.array(self.settings['hsv_high']))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        anchor = np.asarray(self.settings.get('grip_anchor', [0.5, 0.65])) * [width, height]
        gripper_base = np.asarray(self.settings.get('grip_base', [0.5, 1.0])) * [width, height]
        candidates = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < self.settings['minimum_plug_pixels']:
                continue
            yy, xx = np.nonzero(labels == label)
            pixels = np.column_stack([xx, yy]).astype(float)
            center = pixels.mean(axis=0)
            distance = float(np.linalg.norm(center - anchor))
            if distance > self.settings.get('grip_search_radius', 0.35) * height:
                continue
            _, _, axes = np.linalg.svd(pixels - center, full_matrices=False)
            direction = axes[0]
            toward_gripper = gripper_base - center
            toward_gripper /= max(np.linalg.norm(toward_gripper), 1e-9)
            if direction @ toward_gripper < 0:
                direction = -direction
            across = np.array([direction[1], -direction[0]])
            length, body_width = np.ptp(pixels @ direction), np.ptp(pixels @ across)
            aspect = length / max(body_width, 1e-9)
            if not (self.settings.get('plug_min_aspect', 1.2) <= aspect
                    <= self.settings.get('plug_max_aspect', 6.0)):
                continue
            if body_width < self.settings.get('plug_min_width_px', 3):
                continue
            association = float(direction @ toward_gripper)
            if association < self.settings.get('grip_axis_cosine', 0.7):
                continue
            positions = pixels @ direction
            body_tip = pixels[positions <= np.quantile(positions, 0.15)].mean(axis=0)
            tip = body_tip - direction * body_width * self.settings.get('metal_tip_width_ratio', 0.0)
            basis = np.column_stack([across, direction])
            candidates.append((distance, tip, basis, {'plug_center': center.tolist(), 'plug_tip': tip.tolist(),
                                                     'plug_axis': direction.tolist(), 'plug_width_px': float(body_width),
                                                     'gripper_association': association}))
        if not candidates:
            raise ValueError('Held USB body not localized in the gripper region')
        _, tip, basis, status = min(candidates, key=lambda item: item[0])
        return tip, basis, status
