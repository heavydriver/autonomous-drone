"""Unit tests for CSV logging and offline evaluation summaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autonomous_drone.control import TargetAngles
from autonomous_drone.generate_graphs import (
    GroundTruthFrame,
    compute_ground_truth_metrics,
    compute_runtime_summary,
    load_csv_rows,
)
from autonomous_drone.metrics import CsvRunLogger, GateSnapshot, TimingSnapshot
from autonomous_drone.models import BoundingBox, Detection, FollowCommand, TargetObservation, Track


class CsvRunLoggerTest(unittest.TestCase):
    """Validate the runtime CSV logger schema and stateful metrics."""

    def test_logger_writes_frame_detection_and_track_rows(self) -> None:
        """The logger should emit CSV rows and track retention and ID switches."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = CsvRunLogger(
                output_dir=Path(tmp_dir),
                desired_area_ratio=0.08,
            )
            bbox = BoundingBox(10.0, 20.0, 50.0, 80.0)
            detection = Detection(bbox=bbox, confidence=0.9, class_id=0)
            track = Track(track_id=5, bbox=bbox, confidence=0.88, class_id=0)

            logger.log_detections(0, [detection], frame_width=100, frame_height=100)
            logger.log_tracks(0, [track], frame_width=100, frame_height=100)
            logger.log_frame(
                frame_index=0,
                now_s=1.0,
                dt_s=0.1,
                frame_width=100,
                frame_height=100,
                detections_count=1,
                tracks_count=1,
                observation=TargetObservation(
                    track_id=5,
                    bbox=bbox,
                    confidence=0.9,
                    timestamp_s=1.0,
                ),
                target_angles=TargetAngles(
                    horizontal_rad=0.1,
                    vertical_camera_rad=0.2,
                    vertical_body_rad=0.3,
                ),
                command=FollowCommand(
                    velocity_forward_m_s=0.2,
                    velocity_right_m_s=0.0,
                    velocity_down_m_s=0.0,
                    yaw_rate_rad_s=0.1,
                    active=True,
                    reason="tracking",
                ),
                gate=GateSnapshot(
                    follow_allowed=True,
                    guided_mode=True,
                    rc_switch_high=True,
                    rc_channel_pwm=1900,
                    gate_text="enabled",
                ),
                vehicle_state=None,
                timings=TimingSnapshot(
                    frame_grab_latency_s=0.01,
                    detection_latency_s=0.02,
                    tracking_latency_s=0.01,
                    selection_latency_s=0.001,
                    control_latency_s=0.002,
                    mavlink_latency_s=0.003,
                    total_loop_latency_s=0.05,
                ),
            )
            logger.log_frame(
                frame_index=1,
                now_s=1.2,
                dt_s=0.2,
                frame_width=100,
                frame_height=100,
                detections_count=1,
                tracks_count=1,
                observation=TargetObservation(
                    track_id=7,
                    bbox=bbox,
                    confidence=0.85,
                    timestamp_s=1.2,
                ),
                target_angles=TargetAngles(
                    horizontal_rad=0.05,
                    vertical_camera_rad=0.1,
                    vertical_body_rad=0.2,
                ),
                command=FollowCommand.zero("tracking"),
                gate=GateSnapshot(
                    follow_allowed=True,
                    guided_mode=True,
                    rc_switch_high=True,
                    rc_channel_pwm=1900,
                    gate_text="enabled",
                ),
                vehicle_state=None,
                timings=TimingSnapshot(
                    frame_grab_latency_s=0.01,
                    detection_latency_s=0.02,
                    tracking_latency_s=0.01,
                    selection_latency_s=0.001,
                    control_latency_s=0.002,
                    mavlink_latency_s=0.003,
                    total_loop_latency_s=0.05,
                ),
            )
            logger.close()

            frame_rows = load_csv_rows(logger.frames_path)
            self.assertEqual(len(frame_rows), 2)
            self.assertGreater(float(frame_rows[1]["target_retention_time_s"]), 0.0)
            self.assertEqual(int(frame_rows[1]["track_id_switches"]), 1)
            self.assertEqual(logger.frames_path.parent, logger.session_dir)
            self.assertEqual(logger.session_dir.name[:11], "follow-run-")

            detections_rows = load_csv_rows(logger.session_dir / "detections.csv")
            tracks_rows = load_csv_rows(logger.session_dir / "tracks.csv")
            self.assertEqual(len(detections_rows), 1)
            self.assertEqual(len(tracks_rows), 1)


class EvaluationSummaryTest(unittest.TestCase):
    """Validate summary metrics computed from offline logs."""

    def test_runtime_summary_aggregates_key_metrics(self) -> None:
        """Runtime summary should compute basic ratios and percentiles."""

        rows = [
            {
                "monotonic_time_s": "10.0",
                "instant_fps": "10.0",
                "total_loop_latency_s": "0.08",
                "detection_latency_s": "0.03",
                "tracking_latency_s": "0.02",
                "control_latency_s": "0.01",
                "mavlink_latency_s": "0.005",
                "process_cpu_percent": "20.0",
                "process_rss_mb": "100.0",
                "target_present": "true",
                "follow_allowed": "true",
                "command_active": "true",
                "center_error_norm": "0.05",
                "area_ratio_error": "0.01",
                "target_retention_time_s": "0.5",
                "track_id_switches": "0",
            },
            {
                "monotonic_time_s": "10.1",
                "instant_fps": "8.0",
                "total_loop_latency_s": "0.10",
                "detection_latency_s": "0.04",
                "tracking_latency_s": "0.02",
                "control_latency_s": "0.01",
                "mavlink_latency_s": "0.006",
                "process_cpu_percent": "30.0",
                "process_rss_mb": "110.0",
                "target_present": "false",
                "follow_allowed": "true",
                "command_active": "false",
                "center_error_norm": "",
                "area_ratio_error": "",
                "target_retention_time_s": "0.0",
                "track_id_switches": "1",
            },
        ]

        summary = compute_runtime_summary(rows, center_error_threshold=0.1)

        self.assertAlmostEqual(summary["frame_count"], 2.0)
        self.assertAlmostEqual(summary["mean_fps"], 9.0)
        self.assertAlmostEqual(summary["target_present_ratio"], 0.5)
        self.assertAlmostEqual(summary["follow_allowed_ratio"], 1.0)
        self.assertAlmostEqual(summary["framing_accuracy_ratio"], 0.5)
        self.assertEqual(summary["track_id_switches"], 1.0)

    def test_ground_truth_metrics_compute_single_target_scores(self) -> None:
        """Ground-truth metrics should score matched detections and selected tracks."""

        frame_rows = [
            {
                "frame_index": "0",
                "target_present": "true",
                "selected_track_id": "5",
                "bbox_x1_px": "10",
                "bbox_y1_px": "10",
                "bbox_x2_px": "50",
                "bbox_y2_px": "50",
            },
            {
                "frame_index": "1",
                "target_present": "true",
                "selected_track_id": "5",
                "bbox_x1_px": "60",
                "bbox_y1_px": "60",
                "bbox_x2_px": "90",
                "bbox_y2_px": "90",
            },
        ]
        detection_rows = [
            {
                "frame_index": "0",
                "bbox_x1_px": "10",
                "bbox_y1_px": "10",
                "bbox_x2_px": "50",
                "bbox_y2_px": "50",
            },
            {
                "frame_index": "1",
                "bbox_x1_px": "0",
                "bbox_y1_px": "0",
                "bbox_x2_px": "20",
                "bbox_y2_px": "20",
            },
        ]
        ground_truth_rows = [
            GroundTruthFrame(
                frame_index=0,
                present=True,
                bbox=BoundingBox(10.0, 10.0, 50.0, 50.0),
                track_id="person",
            ),
            GroundTruthFrame(
                frame_index=1,
                present=True,
                bbox=BoundingBox(62.0, 62.0, 92.0, 92.0),
                track_id="person",
            ),
        ]

        summary = compute_ground_truth_metrics(
            frame_rows=frame_rows,
            detection_rows=detection_rows,
            ground_truth_rows=ground_truth_rows,
            iou_threshold=0.5,
        )

        self.assertAlmostEqual(summary["detection_precision"], 0.5)
        self.assertAlmostEqual(summary["detection_recall"], 0.5)
        self.assertAlmostEqual(summary["detection_mean_iou"], 1.0)
        self.assertAlmostEqual(summary["tracking_retention_ratio"], 1.0)
        self.assertAlmostEqual(summary["tracking_idf1_single_target"], 1.0)
        self.assertAlmostEqual(summary["tracking_mota_single_target"], 1.0)


if __name__ == "__main__":
    unittest.main()
