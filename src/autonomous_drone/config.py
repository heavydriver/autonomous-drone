"""Configuration models and file loading for the autonomous drone follow stack."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CameraConfig:
    """Camera calibration and mounting configuration."""

    width: int = 640
    height: int = 640
    horizontal_fov_deg: float = 62.0
    vertical_fov_deg: float = 49.0
    mount_pitch_deg: float = 10.0


@dataclass(slots=True)
class MavlinkConfig:
    """MAVLink connection and RC gating configuration."""

    connection_string: str | None = None
    transport: str = "udp"
    udp_host: str = "127.0.0.1"
    udp_port: int = 14550
    serial_device: str = "/dev/serial0"
    baud_rate: int = 57600
    stream_rate_hz: int = 10
    guided_mode_name: str = "GUIDED"
    follow_enable_channel: int = 7
    follow_enable_high_pwm: int = 1800

    def resolved_connection_string(self) -> str:
        """Return the pymavlink connection string for the configured transport.

        Returns:
            A connection string accepted by ``pymavlink.mavutil.mavlink_connection``.

        Raises:
            ValueError: If the configured transport or endpoint is invalid.
        """

        if self.connection_string:
            return self.connection_string

        transport = self.transport.strip().lower()
        if transport == "udp":
            if not self.udp_host:
                raise ValueError("mavlink.udp_host must be set when transport='udp'")
            if self.udp_port <= 0:
                raise ValueError("mavlink.udp_port must be greater than zero")
            return f"udp:{self.udp_host}:{self.udp_port}"

        if transport == "serial":
            if not self.serial_device:
                raise ValueError(
                    "mavlink.serial_device must be set when transport='serial'"
                )
            return self.serial_device

        raise ValueError(
            f"Unsupported MAVLink transport '{self.transport}'. "
            "Expected 'udp' or 'serial'."
        )

    def describe_endpoint(self) -> str:
        """Return a concise human-readable endpoint description."""

        if self.connection_string:
            return f"raw:{self.connection_string}"

        transport = self.transport.strip().lower()
        if transport == "udp":
            return f"udp:{self.udp_host}:{self.udp_port}"
        if transport == "serial":
            return f"serial:{self.serial_device}@{self.baud_rate}"
        return transport


@dataclass(slots=True)
class TrackingConfig:
    """Perception and target-lock configuration."""

    model_path: str = "models/yolo11n.pt"
    detector_confidence: float = 0.40
    person_class_id: int = 0
    min_box_area_ratio: float = 0.002
    desired_box_area_ratio: float = 0.080
    emergency_stop_area_ratio: float = 0.180
    loss_timeout_s: float = 2
    acquisition_confirm_frames: int = 3
    tracker_frame_rate: float = 10.0
    tracker_track_buffer: int = 30
    tracker_match_thresh: float = 0.60
    selector_reacquire_min_iou: float = 0.10
    selector_reacquire_max_center_shift_ratio: float = 0.90


@dataclass(slots=True)
class ControlConfig:
    """Conservative follow-control gains and limits."""

    loop_rate_hz: float = 10.0
    horizontal_angle_filter_alpha: float = 0.25
    box_area_filter_alpha: float = 0.18
    yaw_gain: float = 1.0
    forward_gain: float = 6.0
    lateral_gain: float = 0.0
    yaw_deadband_deg: float = 4.0
    horizontal_alignment_window_deg: float = 10.0
    box_area_deadband_ratio: float = 0.012
    max_forward_speed_m_s: float = 1.5
    max_reverse_speed_m_s: float = 1.5
    max_lateral_speed_m_s: float = 0.15
    max_vertical_speed_m_s: float = 0.0
    max_yaw_rate_deg_s: float = 12.0
    speed_slew_limit_m_s2: float = 0.35
    yaw_slew_limit_deg_s2: float = 30.0
    vertical_slowdown_angle_deg: float = 18.0
    enable_lateral_motion: bool = False


@dataclass(slots=True)
class SafetyConfig:
    """High-level safety and planning configuration."""

    stand_off_distance_m: float = 4.0
    hold_altitude_m: float = 2.4
    max_roll_pitch_deg: float = 10.0


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime behavior for SITL and local testing."""

    dry_run: bool = False
    skip_rc_gate: bool = False
    visualize: bool = False
    video_source: str = "0"
    video_backend: str = "auto"
    model_device: str = "cpu"


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    mavlink: MavlinkConfig = field(default_factory=MavlinkConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_config_file(path: str | Path) -> AppConfig:
    """Load application configuration overrides from a JSON file.

    Args:
        path: Path to a JSON config file.

    Returns:
        A populated application configuration with overrides applied.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a top-level object: {config_path}")

    config = AppConfig()
    apply_overrides(config, payload)
    return config


def apply_overrides(instance: Any, overrides: dict[str, Any]) -> None:
    """Recursively apply dict overrides to a dataclass instance."""

    if not is_dataclass(instance):
        raise TypeError(f"Expected a dataclass instance, got: {type(instance)!r}")

    valid_fields = {field_info.name: field_info for field_info in fields(instance)}
    for key, value in overrides.items():
        if key not in valid_fields:
            raise KeyError(
                f"Unknown config field '{key}' for {type(instance).__name__}"
            )
        current_value = getattr(instance, key)
        if is_dataclass(current_value):
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected nested object for field '{key}', got {type(value).__name__}"
                )
            apply_overrides(current_value, value)
        else:
            setattr(instance, key, value)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Return a JSON-serializable dictionary for the current configuration."""

    return asdict(config)
