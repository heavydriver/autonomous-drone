"""Shared data models for detections, tracking, and control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


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
    """Velocity and yaw-rate command for the drone."""

    velocity_forward_m_s: float
    velocity_right_m_s: float
    velocity_down_m_s: float
    yaw_rate_rad_s: float
    active: bool
    reason: str

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
