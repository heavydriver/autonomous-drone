"""Shared data models for detections, tracking, and control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Return the box width in pixels."""

        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Return the box height in pixels."""

        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Return the box area in square pixels."""

        return self.width * self.height

    @property
    def center_x(self) -> float:
        """Return the horizontal center of the box in pixels."""

        return (self.x1 + self.x2) * 0.5

    @property
    def center_y(self) -> float:
        """Return the vertical center of the box in pixels."""

        return (self.y1 + self.y2) * 0.5

    def area_ratio(self, frame_width: int, frame_height: int) -> float:
        """Return box area divided by frame area."""

        if frame_width <= 0 or frame_height <= 0:
            return 0.0
        return self.area / float(frame_width * frame_height)

    def intersection_over_union(self, other: "BoundingBox") -> float:
        """Return the intersection-over-union with another box."""

        intersection_x1 = max(self.x1, other.x1)
        intersection_y1 = max(self.y1, other.y1)
        intersection_x2 = min(self.x2, other.x2)
        intersection_y2 = min(self.y2, other.y2)
        intersection_width = max(0.0, intersection_x2 - intersection_x1)
        intersection_height = max(0.0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        union_area = self.area + other.area - intersection_area
        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area


@dataclass(frozen=True, slots=True)
class Detection:
    """Single detector output for one object."""

    bbox: BoundingBox
    confidence: float
    class_id: int


@dataclass(frozen=True, slots=True)
class Track:
    """Single tracked object with a persistent track id."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    class_id: int


@dataclass(frozen=True, slots=True)
class TargetObservation:
    """Selected primary target used by the controller."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    timestamp_s: float


@dataclass(frozen=True, slots=True)
class FollowCommand:
    """Generic follow command for either velocity or attitude control."""

    velocity_forward_m_s: float
    velocity_right_m_s: float
    velocity_down_m_s: float
    yaw_rate_rad_s: float
    active: bool
    reason: str
    command_type: Literal["velocity_body", "attitude", "manual_control"] = "velocity_body"
    attitude_roll_rad: float | None = None
    attitude_pitch_rad: float | None = None
    attitude_yaw_rad: float | None = None
    climb_rate_fraction: float | None = None
    manual_pitch: float | None = None
    manual_roll: float | None = None
    manual_throttle: float | None = None
    manual_yaw: float | None = None

    @classmethod
    def zero(cls, reason: str, active: bool = False) -> "FollowCommand":
        """Return a zero-motion command."""

        return cls(
            velocity_forward_m_s=0.0,
            velocity_right_m_s=0.0,
            velocity_down_m_s=0.0,
            yaw_rate_rad_s=0.0,
            active=active,
            reason=reason,
        )

    @classmethod
    def neutral_attitude(
        cls,
        reason: str,
        yaw_rad: float,
        *,
        active: bool = False,
        climb_rate_fraction: float = 0.5,
    ) -> "FollowCommand":
        """Return a level attitude command with neutral climb rate.

        Args:
            reason: Human-readable reason for the command.
            yaw_rad: Absolute yaw target in radians.
            active: Whether the command represents active tracking.
            climb_rate_fraction: ArduPilot climb-rate fraction where 0.5 is neutral
                when ``GUID_OPTIONS = 0``.
        """

        return cls(
            velocity_forward_m_s=0.0,
            velocity_right_m_s=0.0,
            velocity_down_m_s=0.0,
            yaw_rate_rad_s=0.0,
            active=active,
            reason=reason,
            command_type="attitude",
            attitude_roll_rad=0.0,
            attitude_pitch_rad=0.0,
            attitude_yaw_rad=yaw_rad,
            climb_rate_fraction=climb_rate_fraction,
        )

    @classmethod
    def neutral_manual_control(
        cls,
        reason: str,
        *,
        active: bool = False,
        throttle: float = 0.5,
    ) -> "FollowCommand":
        """Return neutral stick inputs for ``ALT_HOLD`` follow.

        Args:
            reason: Human-readable reason for the command.
            active: Whether the command represents active tracking.
            throttle: Normalized throttle in ``[0, 1]`` where ``0.5`` is
                neutral climb in ``ALT_HOLD``.
        """

        return cls(
            velocity_forward_m_s=0.0,
            velocity_right_m_s=0.0,
            velocity_down_m_s=0.0,
            yaw_rate_rad_s=0.0,
            active=active,
            reason=reason,
            command_type="manual_control",
            manual_pitch=0.0,
            manual_roll=0.0,
            manual_throttle=throttle,
            manual_yaw=0.0,
        )


@dataclass(slots=True)
class VehicleState:
    """Subset of vehicle state needed by the follow application."""

    mode: str = "UNKNOWN"
    armed: bool = False
    rc_channels: Dict[int, int] = field(default_factory=dict)
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    yaw_rad: float = 0.0
    last_heartbeat_monotonic_s: float = 0.0

    def rc_channel_pwm(self, channel: int) -> int | None:
        """Return the most recent PWM reading for an RC channel."""

        return self.rc_channels.get(channel)
