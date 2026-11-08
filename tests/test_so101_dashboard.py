"""Track/untrack, scale changes, and browser-to-control HTTP wiring."""

import json
import unittest
import urllib.request

import numpy as np

from dream_hackathon.so101 import ToolTracker
from dream_hackathon.so101_dashboard import Controls, LeaderDashboard


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.controls = Controls(1.0, "right")
        self.tracker = ToolTracker(np.eye(3))
        self.leader = np.eye(4)
        self.follower = np.eye(4)
        self.follower[:3, 3] = [0.4, 0.1, 0.2]

    def update(self):
        return self.tracker.update(self.leader, self.follower, self.controls.snapshot())

    def test_starts_untracked_and_track_anchors_current_poses(self):
        self.assertIsNone(self.update())
        self.controls.change({"tracking": True})
        np.testing.assert_allclose(self.update(), self.follower)
        self.leader[0, 3] += 0.1
        self.assertAlmostEqual(self.update()[0, 3], 0.5)

    def test_untrack_and_retrack_after_repositioning(self):
        self.controls.change({"tracking": True})
        self.update()
        self.controls.change({"tracking": False})
        self.leader[0, 3] = 1
        self.assertIsNone(self.update())
        self.controls.change({"tracking": True})
        np.testing.assert_allclose(self.update(), self.follower)
        self.leader[1, 3] += 0.1
        self.assertAlmostEqual(self.update()[1, 3], 0.2)

    def test_scale_change_keeps_current_target_then_scales_new_motion(self):
        self.controls.change({"tracking": True})
        self.update()
        self.leader[0, 3] = 0.1
        self.follower = self.update()
        before = self.follower.copy()
        self.controls.change({"scale": 2})
        np.testing.assert_allclose(self.update(), before)
        self.leader[0, 3] += 0.1
        self.assertAlmostEqual(self.update()[0, 3], before[0, 3] + 0.2)

    def test_untrack_and_track_between_ticks_reanchors(self):
        self.controls.change({"tracking": True})
        self.update()
        self.controls.change({"tracking": False})
        self.controls.change({"tracking": True})
        self.leader[0, 3] = 1
        np.testing.assert_allclose(self.update(), self.follower)


class DashboardTests(unittest.TestCase):
    def test_page_controls_reach_backend_and_state_survives_reload(self):
        controls = Controls(1.0, "right")
        dashboard = LeaderDashboard(controls, 0)
        url = f"http://127.0.0.1:{dashboard.port}"
        try:
            with urllib.request.urlopen(url) as response:
                html = response.read().decode()
            self.assertEqual(html.count('<button '), 6)
            self.assertEqual(html.count('type="range"'), 2)
            self.assertIn('id="tracking-state"', html)
            self.assertIn('button.onclick=()=>send({tracking:true})', html)
            self.assertIn('untrack.onclick=()=>send({tracking:false})', html)
            for change in ({"tracking": True}, {"scale": 1.75}, {"tracking": False}):
                request = urllib.request.Request(url + "/api/control", data=json.dumps(change).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request) as response:
                    state = json.load(response)
                for key, value in change.items():
                    self.assertEqual(state[key], value)
            with urllib.request.urlopen(url + "/api/state") as response:
                state = json.load(response)
            self.assertFalse(state["tracking"])
            self.assertEqual(state["scale"], 1.75)
        finally:
            dashboard.close()

    def test_kill_calls_power_off_and_clears_tracking_and_recording(self):
        from unittest.mock import Mock

        controls = Controls(1.0, "right")
        controls.report(arm_open=True)
        controls.change({"tracking": True, "recording_requested": True})
        disable = Mock(return_value=[])
        dashboard = LeaderDashboard(controls, 0, on_kill=disable)
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{dashboard.port}/api/kill", data=b"")
            with urllib.request.urlopen(request) as response:
                state = json.load(response)
            disable.assert_called_once()
            self.assertFalse(state["tracking"])
            self.assertFalse(state["recording_requested"])
            self.assertFalse(state["arm_open"])
            self.assertTrue(state["power_off"])
            self.assertEqual(state["message"], "Right arm: motor torque off")
        finally:
            dashboard.close()

    def test_record_controls_do_not_reanchor_tracking(self):
        controls = Controls(1.0, "right")
        controls.report(arm_open=True)
        controls.change({"tracking": True})
        revision = controls.snapshot()["revision"]
        self.assertTrue(controls.change({"recording_requested": True})["recording_requested"])
        self.assertFalse(controls.change({"recording_requested": False})["recording_requested"])
        self.assertEqual(controls.snapshot()["revision"], revision)

    def test_recording_error_survives_home_and_tracking_status_updates(self):
        controls = Controls(1.0, "right")
        controls.report(arm_open=True, recording_error="camera wrist_l: cannot open device")
        controls.request_home()
        controls.report(homing=False, home_requested=False, message="Returned to the captured startup pose")
        self.assertIn("camera wrist_l", controls.snapshot()["recording_error"])
        controls.change({"tracking": True})
        controls.report(message="")
        self.assertIn("camera wrist_l", controls.snapshot()["recording_error"])

    def test_reject_routes_current_and_saved_examples_without_untracking(self):
        for current in (True, False):
            with self.subTest(current=current):
                controls = Controls(1.0, "right")
                controls.report(arm_open=True, saved_episodes=2, recording=current)
                controls.change({"tracking": True})
                dashboard = LeaderDashboard(controls, 0)
                try:
                    request = urllib.request.Request(f"http://127.0.0.1:{dashboard.port}/api/reject", data=b"")
                    with urllib.request.urlopen(request) as response:
                        state = json.load(response)
                    self.assertTrue(state["tracking"])
                    self.assertTrue(state["reject_requested"])
                    self.assertFalse(state["recording_requested"])
                    self.assertEqual(state["reject_kind"], "current" if current else "saved")
                finally:
                    dashboard.close()

    def test_grip_setting_changes_effort_revision_not_pose_anchor(self):
        controls = Controls(1.0, "right")
        controls.change({"tracking": True})
        before = controls.snapshot()
        after = controls.change({"grip_torque_nm": 1.25})
        self.assertEqual(after["grip_torque_nm"], 1.25)
        self.assertEqual(after["grip_revision"], before["grip_revision"] + 1)
        self.assertEqual(after["revision"], before["revision"])

    def test_return_endpoint_uses_home_request_and_stops_tracking(self):
        controls = Controls(1.0, "right")
        controls.report(arm_open=True)
        controls.change({"tracking": True})
        dashboard = LeaderDashboard(controls, 0)
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{dashboard.port}/api/home", data=b"")
            with urllib.request.urlopen(request) as response:
                state = json.load(response)
            self.assertFalse(state["tracking"])
            self.assertTrue(state["home_requested"])
            self.assertTrue(state["homing"])
            self.assertEqual(state["side"], "right")
            state = controls.change({"tracking": True})
            self.assertFalse(state["home_requested"])
            self.assertFalse(state["homing"])
        finally:
            dashboard.close()

    def test_home_matches_existing_recording_rule(self):
        controls = Controls(1.0, "right")
        controls.report(arm_open=True, recording=True)
        with self.assertRaisesRegex(ValueError, "Save the recording"):
            controls.request_home()
        self.assertFalse(controls.snapshot()["home_requested"])


class ReturnMotionTests(unittest.TestCase):
    def test_shared_home_motion_returns_to_nonzero_start_at_existing_speed(self):
        from dream_hackathon import teleop

        start = np.array([0.4, -0.7, 0.2, 0.9, -0.1, 0.3])
        command = start + np.array([0.2, -0.3, 0.1, -0.2, 0.4, -0.4])
        arrived = False
        for _ in range(120):
            previous = command.copy()
            command, arrived = teleop.home_step(command, start, 1 / 30)
            self.assertLessEqual(float(np.max(np.abs(command - previous))), 0.5 / 30 + 1e-12)
            if arrived:
                break
        self.assertTrue(arrived)
        np.testing.assert_allclose(command, start, atol=1e-3)


class StartupTests(unittest.TestCase):
    def test_failed_track_keeps_http_page_available_and_allows_explicit_retry(self):
        from unittest.mock import Mock

        from dream_hackathon.so101 import start_arm

        controls = Controls(1.6, "right")
        controls.change({"tracking": True})
        lock = Mock()
        open_arm = Mock(side_effect=RuntimeError("Release the gripper before starting holding control"))
        robot, handle = start_arm(controls, lambda: lock, open_arm)
        self.assertIsNone(robot)
        self.assertIsNone(handle)
        lock.close.assert_called_once()
        open_arm.assert_called_once()
        dashboard = LeaderDashboard(controls, 0)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{dashboard.port}/api/state") as response:
                state = json.load(response)
            self.assertFalse(state["tracking"])
            self.assertIn("Release the gripper", state["message"])
            self.assertEqual(state["scale"], 1.6)
            controls.change({"tracking": True})
            expected = Mock()
            actual, handle = start_arm(controls, lambda: lock, lambda: expected)
            self.assertIs(actual, expected)
            self.assertIs(handle, lock)
        finally:
            dashboard.close()


if __name__ == "__main__":
    unittest.main()
