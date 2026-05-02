"""Unit tests for conservative follow-controller behavior."""

from __future__ import annotations

import math
import unittest

from autonomous_drone.config import (
    CameraConfig,
    ControlConfig,
    OrbitConfig,
    SafetyConfig,
    TrackingConfig,
)
from autonomous_drone.control import FollowController, OrbitController
from autonomous_drone.models import BoundingBox, TargetObservation


class FollowControllerTest(unittest.TestCase):
    """Validate simple centering and safe target-loss behavior."""

    def setUp(self) -> None:
        self.camera = CameraConfig(width=1280, height=720, horizontal_fov_deg=78.0, vertical_fov_deg=49.0)
        self.tracking = TrackingConfig(
            detector_confidence=0.4,
            min_box_area_ratio=0.01,
            desired_box_area_ratio=0.08,
            acquisition_confirm_frames=3,
            loss_timeout_s=0.5,
        )
        self.control = ControlConfig(
            loop_rate_hz=10.0,
            yaw_deadband_deg=4.0,
            max_forward_speed_m_s=0.8,
            max_reverse_speed_m_s=0.2,
            max_yaw_rate_deg_s=12.0,
            speed_slew_limit_m_s2=0.4,
            yaw_slew_limit_deg_s2=30.0,
        )
        self.controller = FollowController(self.camera, self.tracking, self.control)

    def _make_observation(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        track_id: int,
        timestamp_s: float,
    ) -> TargetObservation:
        half_width = width * 0.5
        half_height = height * 0.5
        return TargetObservation(
            track_id=track_id,
            bbox=BoundingBox(
                x1=center_x - half_width,
                y1=center_y - half_height,
                x2=center_x + half_width,
                y2=center_y + half_height,
            ),
            confidence=0.9,
            timestamp_s=timestamp_s,
        )

    def test_centered_target_stays_near_zero(self) -> None:
        """A well-centered, correctly sized target should not induce oscillation."""

        box_area = self.camera.width * self.camera.height * self.tracking.desired_box_area_ratio
        box_side = math.sqrt(box_area)
        now_s = 0.0
        for _ in range(4):
            observation = self._make_observation(
                center_x=self.camera.width * 0.5,
                center_y=self.camera.height * 0.5,
                width=box_side,
                height=box_side,
                track_id=1,
                timestamp_s=now_s,
            )
            command = self.controller.step(observation, now_s, follow_allowed=True)
            now_s += 0.1

        self.assertAlmostEqual(command.velocity_forward_m_s, 0.0, places=3)
        self.assertAlmostEqual(command.velocity_right_m_s, 0.0, places=3)
        self.assertAlmostEqual(command.yaw_rate_rad_s, 0.0, places=3)

    def test_command_changes_are_rate_limited(self) -> None:
        """Large target error should still ramp up gradually."""

        box_area = self.camera.width * self.camera.height * 0.03
        box_side = math.sqrt(box_area)
        command = None
        now_s = 0.0
        for _ in range(4):
            observation = self._make_observation(
                center_x=self.camera.width * 0.72,
                center_y=self.camera.height * 0.5,
                width=box_side,
                height=box_side,
                track_id=1,
                timestamp_s=now_s,
            )
            command = self.controller.step(observation, now_s, follow_allowed=True)
            now_s += 0.1

        assert command is not None
        self.assertLessEqual(command.velocity_forward_m_s, 0.12)
        self.assertLessEqual(abs(command.yaw_rate_rad_s), math.radians(9.0))

    def test_target_loss_ramps_back_to_zero(self) -> None:
        """When the target is lost, the controller should stop instead of chasing ghosts."""

        box_area = self.camera.width * self.camera.height * 0.03
        box_side = math.sqrt(box_area)
        now_s = 0.0
        for _ in range(4):
            observation = self._make_observation(
                center_x=self.camera.width * 0.7,
                center_y=self.camera.height * 0.5,
                width=box_side,
                height=box_side,
                track_id=1,
                timestamp_s=now_s,
            )
            command = self.controller.step(observation, now_s, follow_allowed=True)
            now_s += 0.1

        lost_command = self.controller.step(None, now_s + 0.7, follow_allowed=True)
        self.assertFalse(lost_command.active)
        self.assertLessEqual(abs(lost_command.velocity_forward_m_s), abs(command.velocity_forward_m_s))
        self.assertLessEqual(abs(lost_command.yaw_rate_rad_s), abs(command.yaw_rate_rad_s))
        self.assertIn("lost", lost_command.reason)

    def test_small_visible_target_is_still_tracked(self) -> None:
        """A small but visible target should not be treated as lost."""

        now_s = 0.0
        command = None
        for _ in range(4):
            observation = self._make_observation(
                center_x=self.camera.width * 0.68,
                center_y=self.camera.height * 0.5,
                width=40.0,
                height=60.0,
                track_id=1,
                timestamp_s=now_s,
            )
            command = self.controller.step(observation, now_s, follow_allowed=True)
            now_s += 0.1

        assert command is not None
        self.assertTrue(command.active)
        self.assertNotIn("lost", command.reason)
        self.assertGreater(abs(command.yaw_rate_rad_s), 0.0)


class OrbitControllerTest(unittest.TestCase):
    """Validate the optional single-orbit maneuver behavior."""

    def setUp(self) -> None:
        self.camera = CameraConfig(width=1280, height=720, horizontal_fov_deg=78.0, vertical_fov_deg=49.0)
        self.tracking = TrackingConfig(
            desired_box_area_ratio=0.08,
            emergency_stop_area_ratio=0.18,
        )
        self.control = ControlConfig(
            loop_rate_hz=10.0,
            speed_slew_limit_m_s2=0.4,
            yaw_slew_limit_deg_s2=30.0,
        )
        self.safety = SafetyConfig(stand_off_distance_m=4.0)
        self.orbit = OrbitConfig(
            clockwise=True,
            lateral_speed_m_s=0.4,
            forward_correction_gain=4.0,
            max_forward_correction_m_s=0.3,
            yaw_gain=1.2,
            max_yaw_rate_deg_s=18.0,
            completion_angle_deg=20.0,
            max_duration_s=30.0,
        )
        self.controller = OrbitController(
            self.camera,
            self.tracking,
            self.control,
            self.safety,
            self.orbit,
        )

    def _make_observation(self, center_x: float, width: float, height: float) -> TargetObservation:
        return TargetObservation(
            track_id=1,
            bbox=BoundingBox(
                x1=center_x - width * 0.5,
                y1=self.camera.height * 0.5 - height * 0.5,
                x2=center_x + width * 0.5,
                y2=self.camera.height * 0.5 + height * 0.5,
            ),
            confidence=0.9,
            timestamp_s=0.0,
        )

    def test_orbit_outputs_lateral_motion_when_active(self) -> None:
        """Starting an orbit should generate a bounded lateral command."""

        observation = self._make_observation(center_x=640.0, width=300.0, height=245.0)
        self.controller.start(now_s=0.0)

        command = self.controller.step(observation, now_s=0.1, follow_allowed=True)

        assert command is not None
        self.assertTrue(command.active)
        self.assertGreater(command.velocity_right_m_s, 0.0)
        self.assertEqual(command.reason, "orbiting target")

    def test_orbit_completes_and_yields_back(self) -> None:
        """The maneuver should stop after one configured revolution."""

        observation = self._make_observation(center_x=640.0, width=300.0, height=245.0)
        self.controller.start(now_s=0.0)

        command = None
        now_s = 0.1
        for _ in range(50):
            command = self.controller.step(observation, now_s=now_s, follow_allowed=True)
            now_s += 0.1
            if command is None:
                break

        self.assertIsNone(command)
        self.assertFalse(self.controller.status.active)
        self.assertIn("complete", self.controller.status.reason)

    def test_orbit_aborts_when_follow_is_disabled(self) -> None:
        """RC or mode gating should cancel the maneuver immediately."""

        observation = self._make_observation(center_x=640.0, width=300.0, height=245.0)
        self.controller.start(now_s=0.0)

        command = self.controller.step(observation, now_s=0.1, follow_allowed=False)

        self.assertIsNone(command)
        self.assertFalse(self.controller.status.active)
        self.assertIn("disabled", self.controller.status.reason)


if __name__ == "__main__":
    unittest.main()
