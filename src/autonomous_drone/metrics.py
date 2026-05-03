"""CSV logging helpers for runtime evaluation of the follow stack."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO

from autonomous_drone.control import TargetAngles
from autonomous_drone.models import Detection, FollowCommand, TargetObservation, Track, VehicleState


@dataclass(frozen=True, slots=True)
class GateSnapshot:
    """Snapshot of the autonomy gate state for one frame."""

    follow_allowed: bool
    guided_mode: bool
    rc_switch_high: bool
    rc_channel_pwm: int | None
    gate_text: str


@dataclass(frozen=True, slots=True)
class TimingSnapshot:
    """Per-frame runtime timings measured inside the main loop."""

    frame_grab_latency_s: float
    detection_latency_s: float
    tracking_latency_s: float
    selection_latency_s: float
    control_latency_s: float
    mavlink_latency_s: float
    total_loop_latency_s: float


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Process resource usage sampled for one frame."""

    process_cpu_percent: float | None
    process_rss_mb: float | None


class ProcessMonitor:
    """Sample lightweight process resource usage without affecting control flow."""

    def __init__(self) -> None:
        """Initialize the optional psutil-backed process monitor."""

        try:
            import psutil
        except ImportError:  # pragma: no cover - depends on runtime env
            self._process = None
            return

        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(interval=None)

    def sample(self) -> ResourceSnapshot:
        """Return the latest process CPU and memory measurements."""

        if self._process is None:
            return ResourceSnapshot(process_cpu_percent=None, process_rss_mb=None)

        cpu_percent = self._process.cpu_percent(interval=None)
        rss_mb = self._process.memory_info().rss / (1024.0 * 1024.0)
        return ResourceSnapshot(
            process_cpu_percent=float(cpu_percent),
            process_rss_mb=float(rss_mb),
        )


class CsvRunLogger:
    """Persist frame-by-frame runtime data for offline evaluation."""

    _FRAME_COLUMNS = (
        "session_id",
        "frame_index",
        "wall_time_utc",
        "monotonic_time_s",
        "dt_s",
        "instant_fps",
        "frame_width_px",
        "frame_height_px",
        "detections_count",
        "tracks_count",
        "target_present",
        "target_in_frame",
        "selected_track_id",
        "selected_confidence",
        "bbox_x1_px",
        "bbox_y1_px",
        "bbox_x2_px",
        "bbox_y2_px",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_area_ratio",
        "desired_area_ratio",
        "area_ratio_error",
        "target_center_x_px",
        "target_center_y_px",
        "target_center_x_norm",
        "target_center_y_norm",
        "center_error_x_norm",
        "center_error_y_norm",
        "center_error_norm",
        "horizontal_error_rad",
        "vertical_camera_error_rad",
        "vertical_body_error_rad",
        "target_retention_time_s",
        "track_id_switches",
        "follow_allowed",
        "guided_mode",
        "rc_switch_high",
        "rc_channel_pwm",
        "gate_text",
        "pose_sampled",
        "pose_right_hand_up",
        "pose_reason",
        "orbit_active",
        "orbit_progress_deg",
        "orbit_reason",
        "command_active",
        "command_type",
        "command_reason",
        "command_velocity_forward_m_s",
        "command_velocity_right_m_s",
        "command_velocity_down_m_s",
        "command_yaw_rate_rad_s",
        "command_attitude_roll_rad",
        "command_attitude_pitch_rad",
        "command_attitude_yaw_rad",
        "command_climb_rate_fraction",
        "command_manual_pitch",
        "command_manual_roll",
        "command_manual_throttle",
        "command_manual_yaw",
        "vehicle_mode",
        "vehicle_armed",
        "vehicle_roll_rad",
        "vehicle_pitch_rad",
        "vehicle_yaw_rad",
        "frame_grab_latency_s",
        "detection_latency_s",
        "tracking_latency_s",
        "selection_latency_s",
        "control_latency_s",
        "mavlink_latency_s",
        "total_loop_latency_s",
        "process_cpu_percent",
        "process_rss_mb",
    )
    _DETECTION_COLUMNS = (
        "session_id",
        "frame_index",
        "detection_index",
        "confidence",
        "class_id",
        "bbox_x1_px",
        "bbox_y1_px",
        "bbox_x2_px",
        "bbox_y2_px",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_area_ratio",
    )
    _TRACK_COLUMNS = (
        "session_id",
        "frame_index",
        "track_index",
        "track_id",
        "confidence",
        "class_id",
        "bbox_x1_px",
        "bbox_y1_px",
        "bbox_x2_px",
        "bbox_y2_px",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_area_ratio",
    )

    def __init__(
        self,
        output_dir: Path,
        desired_area_ratio: float,
    ) -> None:
        """Create a new logging session.

        Args:
            output_dir: Directory that will contain the CSV log files.
            desired_area_ratio: Configured target area ratio used for distance error.
        """

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        self._session_id = f"follow-run-{timestamp}"
        self._desired_area_ratio = desired_area_ratio
        output_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir = output_dir / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._frames_path = self._session_dir / "frames.csv"
        self._detections_path = self._session_dir / "detections.csv"
        self._tracks_path = self._session_dir / "tracks.csv"
        self._frames_handle = self._open_csv(self._frames_path)
        self._detections_handle = self._open_csv(self._detections_path)
        self._tracks_handle = self._open_csv(self._tracks_path)
        self._frame_writer = csv.DictWriter(
            self._frames_handle,
            fieldnames=self._FRAME_COLUMNS,
        )
        self._detection_writer = csv.DictWriter(
            self._detections_handle,
            fieldnames=self._DETECTION_COLUMNS,
        )
        self._track_writer = csv.DictWriter(
            self._tracks_handle,
            fieldnames=self._TRACK_COLUMNS,
        )
        self._frame_writer.writeheader()
        self._detection_writer.writeheader()
        self._track_writer.writeheader()
        self._resource_monitor = ProcessMonitor()
        self._frame_index = 0
        self._retention_started_s: float | None = None
        self._last_target_track_id: int | None = None
        self._last_target_present = False
        self._track_id_switches = 0

    @property
    def frames_path(self) -> Path:
        """Return the primary per-frame CSV path."""

        return self._frames_path

    @property
    def session_dir(self) -> Path:
        """Return the output directory for this logging session."""

        return self._session_dir

    def next_frame_index(self) -> int:
        """Return the next frame index to use for this session."""

        return self._frame_index

    def log_detections(
        self,
        frame_index: int,
        detections: Iterable[Detection],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Write raw detector outputs for one frame."""

        for detection_index, detection in enumerate(detections):
            self._detection_writer.writerow(
                {
                    "session_id": self._session_id,
                    "frame_index": frame_index,
                    "detection_index": detection_index,
                    "confidence": detection.confidence,
                    "class_id": detection.class_id,
                    "bbox_x1_px": detection.bbox.x1,
                    "bbox_y1_px": detection.bbox.y1,
                    "bbox_x2_px": detection.bbox.x2,
                    "bbox_y2_px": detection.bbox.y2,
                    "bbox_width_px": detection.bbox.width,
                    "bbox_height_px": detection.bbox.height,
                    "bbox_area_ratio": detection.bbox.area_ratio(
                        frame_width,
                        frame_height,
                    ),
                }
            )
        self._detections_handle.flush()

    def log_tracks(
        self,
        frame_index: int,
        tracks: Iterable[Track],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Write tracker outputs for one frame."""

        for track_index, track in enumerate(tracks):
            self._track_writer.writerow(
                {
                    "session_id": self._session_id,
                    "frame_index": frame_index,
                    "track_index": track_index,
                    "track_id": track.track_id,
                    "confidence": track.confidence,
                    "class_id": track.class_id,
                    "bbox_x1_px": track.bbox.x1,
                    "bbox_y1_px": track.bbox.y1,
                    "bbox_x2_px": track.bbox.x2,
                    "bbox_y2_px": track.bbox.y2,
                    "bbox_width_px": track.bbox.width,
                    "bbox_height_px": track.bbox.height,
                    "bbox_area_ratio": track.bbox.area_ratio(
                        frame_width,
                        frame_height,
                    ),
                }
            )
        self._tracks_handle.flush()

    def log_frame(
        self,
        *,
        frame_index: int,
        now_s: float,
        dt_s: float,
        frame_width: int,
        frame_height: int,
        detections_count: int,
        tracks_count: int,
        observation: TargetObservation | None,
        target_angles: TargetAngles | None,
        command: FollowCommand,
        gate: GateSnapshot,
        vehicle_state: VehicleState | None,
        pose_sampled: bool,
        pose_right_hand_up: bool | None,
        pose_reason: str,
        orbit_active: bool,
        orbit_progress_deg: float,
        orbit_reason: str,
        timings: TimingSnapshot,
    ) -> None:
        """Write aggregated metrics for one frame."""

        resource = self._resource_monitor.sample()
        if observation is not None:
            bbox = observation.bbox
            area_ratio = bbox.area_ratio(frame_width, frame_height)
            center_x_norm = bbox.center_x / max(frame_width, 1)
            center_y_norm = bbox.center_y / max(frame_height, 1)
            center_error_x_norm = center_x_norm - 0.5
            center_error_y_norm = center_y_norm - 0.5
            center_error_norm = (
                center_error_x_norm * center_error_x_norm
                + center_error_y_norm * center_error_y_norm
            ) ** 0.5
            target_in_frame = (
                bbox.x1 >= 0.0
                and bbox.y1 >= 0.0
                and bbox.x2 <= frame_width
                and bbox.y2 <= frame_height
            )
            area_ratio_error = self._desired_area_ratio - area_ratio
            retention_time_s = self._update_retention_state(
                track_id=observation.track_id,
                now_s=now_s,
            )
        else:
            bbox = None
            area_ratio = None
            center_x_norm = None
            center_y_norm = None
            center_error_x_norm = None
            center_error_y_norm = None
            center_error_norm = None
            target_in_frame = False
            area_ratio_error = None
            retention_time_s = self._update_retention_state(track_id=None, now_s=now_s)

        self._frame_writer.writerow(
            {
                "session_id": self._session_id,
                "frame_index": frame_index,
                "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_time_s": now_s,
                "dt_s": dt_s,
                "instant_fps": (1.0 / dt_s) if dt_s > 0.0 else 0.0,
                "frame_width_px": frame_width,
                "frame_height_px": frame_height,
                "detections_count": detections_count,
                "tracks_count": tracks_count,
                "target_present": observation is not None,
                "target_in_frame": target_in_frame,
                "selected_track_id": (
                    observation.track_id if observation is not None else None
                ),
                "selected_confidence": (
                    observation.confidence if observation is not None else None
                ),
                "bbox_x1_px": bbox.x1 if bbox is not None else None,
                "bbox_y1_px": bbox.y1 if bbox is not None else None,
                "bbox_x2_px": bbox.x2 if bbox is not None else None,
                "bbox_y2_px": bbox.y2 if bbox is not None else None,
                "bbox_width_px": bbox.width if bbox is not None else None,
                "bbox_height_px": bbox.height if bbox is not None else None,
                "bbox_area_ratio": area_ratio,
                "desired_area_ratio": self._desired_area_ratio,
                "area_ratio_error": area_ratio_error,
                "target_center_x_px": bbox.center_x if bbox is not None else None,
                "target_center_y_px": bbox.center_y if bbox is not None else None,
                "target_center_x_norm": center_x_norm,
                "target_center_y_norm": center_y_norm,
                "center_error_x_norm": center_error_x_norm,
                "center_error_y_norm": center_error_y_norm,
                "center_error_norm": center_error_norm,
                "horizontal_error_rad": (
                    target_angles.horizontal_rad if target_angles is not None else None
                ),
                "vertical_camera_error_rad": (
                    target_angles.vertical_camera_rad
                    if target_angles is not None
                    else None
                ),
                "vertical_body_error_rad": (
                    target_angles.vertical_body_rad if target_angles is not None else None
                ),
                "target_retention_time_s": retention_time_s,
                "track_id_switches": self._track_id_switches,
                "follow_allowed": gate.follow_allowed,
                "guided_mode": gate.guided_mode,
                "rc_switch_high": gate.rc_switch_high,
                "rc_channel_pwm": gate.rc_channel_pwm,
                "gate_text": gate.gate_text,
                "pose_sampled": pose_sampled,
                "pose_right_hand_up": pose_right_hand_up,
                "pose_reason": pose_reason,
                "orbit_active": orbit_active,
                "orbit_progress_deg": orbit_progress_deg,
                "orbit_reason": orbit_reason,
                "command_active": command.active,
                "command_type": command.command_type,
                "command_reason": command.reason,
                "command_velocity_forward_m_s": command.velocity_forward_m_s,
                "command_velocity_right_m_s": command.velocity_right_m_s,
                "command_velocity_down_m_s": command.velocity_down_m_s,
                "command_yaw_rate_rad_s": command.yaw_rate_rad_s,
                "command_attitude_roll_rad": command.attitude_roll_rad,
                "command_attitude_pitch_rad": command.attitude_pitch_rad,
                "command_attitude_yaw_rad": command.attitude_yaw_rad,
                "command_climb_rate_fraction": command.climb_rate_fraction,
                "command_manual_pitch": command.manual_pitch,
                "command_manual_roll": command.manual_roll,
                "command_manual_throttle": command.manual_throttle,
                "command_manual_yaw": command.manual_yaw,
                "vehicle_mode": vehicle_state.mode if vehicle_state is not None else None,
                "vehicle_armed": (
                    vehicle_state.armed if vehicle_state is not None else None
                ),
                "vehicle_roll_rad": (
                    vehicle_state.roll_rad if vehicle_state is not None else None
                ),
                "vehicle_pitch_rad": (
                    vehicle_state.pitch_rad if vehicle_state is not None else None
                ),
                "vehicle_yaw_rad": (
                    vehicle_state.yaw_rad if vehicle_state is not None else None
                ),
                "frame_grab_latency_s": timings.frame_grab_latency_s,
                "detection_latency_s": timings.detection_latency_s,
                "tracking_latency_s": timings.tracking_latency_s,
                "selection_latency_s": timings.selection_latency_s,
                "control_latency_s": timings.control_latency_s,
                "mavlink_latency_s": timings.mavlink_latency_s,
                "total_loop_latency_s": timings.total_loop_latency_s,
                "process_cpu_percent": resource.process_cpu_percent,
                "process_rss_mb": resource.process_rss_mb,
            }
        )
        self._frames_handle.flush()
        self._frame_index += 1

    def close(self) -> None:
        """Close all CSV handles."""

        self._frames_handle.close()
        self._detections_handle.close()
        self._tracks_handle.close()

    def _update_retention_state(
        self,
        track_id: int | None,
        now_s: float,
    ) -> float:
        """Track continuous target retention and id-switch count."""

        if track_id is None:
            self._retention_started_s = None
            self._last_target_track_id = None
            self._last_target_present = False
            return 0.0

        if not self._last_target_present:
            self._retention_started_s = now_s
        elif self._last_target_track_id is not None and track_id != self._last_target_track_id:
            self._track_id_switches += 1

        self._last_target_track_id = track_id
        self._last_target_present = True
        if self._retention_started_s is None:
            self._retention_started_s = now_s
        return max(0.0, now_s - self._retention_started_s)

    @staticmethod
    def _open_csv(path: Path) -> TextIO:
        """Open a CSV file for line-buffered text output."""

        return path.open("w", encoding="utf-8", newline="")
