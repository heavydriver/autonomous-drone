"""Unit tests for target selection continuity across tracker dropouts."""

from __future__ import annotations

import unittest

from autonomous_drone.config import PoseConfig, TrackingConfig
from autonomous_drone.models import BoundingBox, Detection, Track
from autonomous_drone.perception import (
    PoseKeypoint,
    PrimaryTargetSelector,
    detect_right_hand_up,
)


class PrimaryTargetSelectorTest(unittest.TestCase):
    """Validate stable target selection when tracking briefly blips."""

    def setUp(self) -> None:
        self.config = TrackingConfig(
            detector_confidence=0.4,
            selector_reacquire_min_iou=0.1,
            selector_reacquire_max_center_shift_ratio=0.9,
        )
        self.selector = PrimaryTargetSelector(config=self.config)

    def test_reuses_raw_detection_when_tracker_misses(self) -> None:
        """A nearby detection should preserve the active target through a tracker miss."""

        initial_track = Track(
            track_id=7,
            bbox=BoundingBox(100.0, 100.0, 220.0, 320.0),
            confidence=0.92,
            class_id=0,
        )
        observation = self.selector.select([initial_track], [], now_s=1.0)
        assert observation is not None
        self.assertEqual(observation.track_id, 7)

        fallback_detection = Detection(
            bbox=BoundingBox(108.0, 108.0, 232.0, 334.0),
            confidence=0.88,
            class_id=0,
        )
        observation = self.selector.select([], [fallback_detection], now_s=1.1)
        assert observation is not None
        self.assertEqual(observation.track_id, 7)
        self.assertAlmostEqual(observation.bbox.center_x, fallback_detection.bbox.center_x)
        self.assertAlmostEqual(observation.bbox.center_y, fallback_detection.bbox.center_y)

    def test_reuses_nearby_retracked_person_after_id_switch(self) -> None:
        """A new tracker id near the prior box should not force a target reset."""

        initial_track = Track(
            track_id=3,
            bbox=BoundingBox(200.0, 120.0, 320.0, 360.0),
            confidence=0.95,
            class_id=0,
        )
        observation = self.selector.select([initial_track], [], now_s=2.0)
        assert observation is not None
        self.assertEqual(observation.track_id, 3)

        replacement_track = Track(
            track_id=11,
            bbox=BoundingBox(210.0, 128.0, 332.0, 364.0),
            confidence=0.91,
            class_id=0,
        )
        observation = self.selector.select([replacement_track], [], now_s=2.1)
        assert observation is not None
        self.assertEqual(observation.track_id, 3)

    def test_ignores_far_detection_and_reports_loss(self) -> None:
        """A distant detection should not be mistaken for the locked target."""

        initial_track = Track(
            track_id=5,
            bbox=BoundingBox(120.0, 120.0, 240.0, 300.0),
            confidence=0.9,
            class_id=0,
        )
        observation = self.selector.select([initial_track], [], now_s=3.0)
        assert observation is not None

        unrelated_detection = Detection(
            bbox=BoundingBox(500.0, 100.0, 620.0, 300.0),
            confidence=0.93,
            class_id=0,
        )
        observation = self.selector.select([], [unrelated_detection], now_s=3.1)
        self.assertIsNone(observation)

    def test_bootstraps_target_from_detection_when_no_track_exists(self) -> None:
        """The selector should be able to start from a detection before tracking stabilizes."""

        detection = Detection(
            bbox=BoundingBox(150.0, 90.0, 290.0, 340.0),
            confidence=0.86,
            class_id=0,
        )
        observation = self.selector.select([], [detection], now_s=4.0)
        assert observation is not None
        self.assertLess(observation.track_id, 0)
        self.assertAlmostEqual(observation.bbox.area, detection.bbox.area)

    def test_reacquires_after_one_empty_frame(self) -> None:
        """A single blank frame should not erase the last lock footprint."""

        initial_track = Track(
            track_id=9,
            bbox=BoundingBox(140.0, 100.0, 260.0, 320.0),
            confidence=0.9,
            class_id=0,
        )
        observation = self.selector.select([initial_track], [], now_s=5.0)
        assert observation is not None

        observation = self.selector.select([], [], now_s=5.1)
        self.assertIsNone(observation)

        fallback_detection = Detection(
            bbox=BoundingBox(148.0, 104.0, 272.0, 328.0),
            confidence=0.85,
            class_id=0,
        )
        observation = self.selector.select([], [fallback_detection], now_s=5.2)
        assert observation is not None
        self.assertEqual(observation.track_id, 9)


class PoseGestureDetectionTest(unittest.TestCase):
    """Validate conservative right-hand-raise classification."""

    def setUp(self) -> None:
        self.config = PoseConfig(
            min_keypoint_confidence=0.3,
            wrist_above_shoulder_margin_ratio=0.05,
            wrist_above_elbow_margin_ratio=0.02,
            max_wrist_shoulder_offset_ratio=0.35,
        )
        self.bbox = BoundingBox(100.0, 100.0, 220.0, 340.0)

    def _blank_pose(self) -> list[PoseKeypoint]:
        return [PoseKeypoint(0.0, 0.0, 0.0) for _ in range(17)]

    def test_detects_raised_right_hand(self) -> None:
        """A wrist clearly above the right shoulder and elbow should trigger."""

        keypoints = self._blank_pose()
        keypoints[6] = PoseKeypoint(170.0, 180.0, 0.9)
        keypoints[8] = PoseKeypoint(168.0, 160.0, 0.9)
        keypoints[10] = PoseKeypoint(166.0, 130.0, 0.9)

        result = detect_right_hand_up(keypoints, self.bbox, self.config)

        self.assertTrue(result.sampled)
        self.assertTrue(result.right_hand_up)

    def test_rejects_wrist_below_shoulder(self) -> None:
        """A lowered wrist should not trigger the maneuver."""

        keypoints = self._blank_pose()
        keypoints[6] = PoseKeypoint(170.0, 180.0, 0.9)
        keypoints[8] = PoseKeypoint(168.0, 200.0, 0.9)
        keypoints[10] = PoseKeypoint(166.0, 220.0, 0.9)

        result = detect_right_hand_up(keypoints, self.bbox, self.config)

        self.assertTrue(result.sampled)
        self.assertFalse(result.right_hand_up)

    def test_rejects_low_confidence_arm_landmarks(self) -> None:
        """Low-confidence arm keypoints should not be treated as a positive gesture."""

        keypoints = self._blank_pose()
        keypoints[6] = PoseKeypoint(170.0, 180.0, 0.25)
        keypoints[8] = PoseKeypoint(168.0, 160.0, 0.9)
        keypoints[10] = PoseKeypoint(166.0, 130.0, 0.9)

        result = detect_right_hand_up(keypoints, self.bbox, self.config)

        self.assertTrue(result.sampled)
        self.assertIsNone(result.right_hand_up)
        self.assertIn("low confidence", result.reason)


if __name__ == "__main__":
    unittest.main()
