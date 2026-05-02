"""Simple person-follow and orbit controllers for person-centric flight."""

from __future__ import annotations

import math
from dataclasses import dataclass

from autonomous_drone.config import (
    CameraConfig,
    ControlConfig,
    OrbitConfig,
    SafetyConfig,
    TrackingConfig,
)
from autonomous_drone.models import FollowCommand, TargetObservation


@dataclass(frozen=True, slots=True)
class TargetAngles:
    """Angular location of the target relative to the drone body frame."""

    horizontal_rad: float
    vertical_camera_rad: float
    vertical_body_rad: float


@dataclass(frozen=True, slots=True)
class OrbitStatus:
    """Current state of the optional one-shot orbit maneuver."""

    active: bool
    progress_rad: float
    reason: str


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value to a fixed range."""

    return max(minimum, min(maximum, value))


def _low_pass(previous: float | None, current: float, alpha: float) -> float:
    """Apply a first-order low-pass filter."""

    if previous is None:
        return current
    return previous + alpha * (current - previous)


def _rate_limit(current: float, target: float, max_delta: float) -> float:
    """Limit how much a signal can change in one update."""

    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target


def compute_target_angles(
    camera: CameraConfig,
    observation: TargetObservation,
) -> TargetAngles:
    """Convert a target box center into camera/body angular offsets."""

    frame_half_width = camera.width * 0.5
    frame_half_height = camera.height * 0.5
    normalized_x = _clamp(
        (observation.bbox.center_x - frame_half_width) / frame_half_width,
        -1.0,
        1.0,
    )
    normalized_y = _clamp(
        (observation.bbox.center_y - frame_half_height) / frame_half_height,
        -1.0,
        1.0,
    )
    half_hfov_rad = math.radians(camera.horizontal_fov_deg) * 0.5
    half_vfov_rad = math.radians(camera.vertical_fov_deg) * 0.5
    horizontal_rad = math.atan(normalized_x * math.tan(half_hfov_rad))
    vertical_camera_rad = math.atan(normalized_y * math.tan(half_vfov_rad))
    vertical_body_rad = vertical_camera_rad + math.radians(camera.mount_pitch_deg)
    return TargetAngles(
        horizontal_rad=horizontal_rad,
        vertical_camera_rad=vertical_camera_rad,
        vertical_body_rad=vertical_body_rad,
    )


class FollowController:
    """Stateful follow controller that centers the target and holds distance."""

    def __init__(
        self,
        camera: CameraConfig,
        tracking: TrackingConfig,
        control: ControlConfig,
    ) -> None:
        """Initialize the controller.

        Args:
            camera: Camera calibration and mounting configuration.
            tracking: Tracking thresholds and lock behavior.
            control: Control gains and motion limits.
        """

        self._camera = camera
        self._tracking = tracking
        self._control = control
        self._filtered_horizontal_rad: float | None = None
        self._filtered_area_ratio: float | None = None
        self._last_command = FollowCommand.zero("controller not started")
        self._last_update_s: float | None = None
        self._last_seen_target_s: float | None = None
        self._locked_track_id: int | None = None
        self._confirmed_frames = 0

    @property
    def locked_track_id(self) -> int | None:
        """Return the currently locked target id, if any."""

        return self._locked_track_id

    def step(
        self,
        observation: TargetObservation | None,
        now_s: float,
        follow_allowed: bool,
    ) -> FollowCommand:
        """Advance the controller by one frame.

        Args:
            observation: Currently selected target observation, or ``None``.
            now_s: Monotonic timestamp in seconds.
            follow_allowed: Whether external gating allows autonomous follow.

        Returns:
            A body-frame velocity and yaw-rate command.
        """

        dt = self._compute_dt(now_s)
        if not follow_allowed:
            self._confirmed_frames = 0
            return self._ramp_to_zero(dt, "follow disabled")

        if observation is None:
            return self._handle_missing_target(now_s, dt)

        area_ratio = observation.bbox.area_ratio(self._camera.width, self._camera.height)

        if observation.track_id != self._locked_track_id:
            self._locked_track_id = observation.track_id
            self._confirmed_frames = 0
            self._filtered_horizontal_rad = None
            self._filtered_area_ratio = None

        self._confirmed_frames += 1
        self._last_seen_target_s = now_s

        angles = self.compute_target_angles(observation)
        self._filtered_horizontal_rad = _low_pass(
            self._filtered_horizontal_rad,
            angles.horizontal_rad,
            self._control.horizontal_angle_filter_alpha,
        )
        self._filtered_area_ratio = _low_pass(
            self._filtered_area_ratio,
            area_ratio,
            self._control.box_area_filter_alpha,
        )

        if self._confirmed_frames < self._tracking.acquisition_confirm_frames:
            return self._ramp_to_zero(dt, "acquiring target")

        command = self._compute_tracking_command(
            horizontal_rad=self._filtered_horizontal_rad,
            area_ratio=self._filtered_area_ratio,
        )
        return self._apply_rate_limits(command, dt)

    def compute_target_angles(self, observation: TargetObservation) -> TargetAngles:
        """Convert a target box center into camera/body angular offsets."""

        return compute_target_angles(self._camera, observation)

    def _compute_dt(self, now_s: float) -> float:
        """Compute a sane control-step delta time."""

        if self._last_update_s is None:
            self._last_update_s = now_s
            return 1.0 / self._control.loop_rate_hz
        dt = max(now_s - self._last_update_s, 1e-3)
        self._last_update_s = now_s
        return dt

    def _handle_missing_target(
        self,
        now_s: float,
        dt: float,
        reason: str = "target missing",
    ) -> FollowCommand:
        """Handle target loss with a conservative ramp-down."""

        if self._last_seen_target_s is None:
            self._confirmed_frames = 0
            return self._ramp_to_zero(dt, reason)
        if now_s - self._last_seen_target_s > self._tracking.loss_timeout_s:
            self._confirmed_frames = 0
            return self._ramp_to_zero(dt, "target lost")
        return self._ramp_to_zero(dt, "target temporarily missing")

    def _compute_tracking_command(
        self,
        horizontal_rad: float,
        area_ratio: float,
    ) -> FollowCommand:
        """Compute a follow command from filtered image alignment and size."""

        yaw_deadband_rad = math.radians(self._control.yaw_deadband_deg)
        if abs(horizontal_rad) <= yaw_deadband_rad:
            yaw_rate_rad_s = 0.0
        else:
            yaw_rate_rad_s = self._control.yaw_gain * horizontal_rad

        area_error = self._tracking.desired_box_area_ratio - area_ratio
        if abs(area_error) <= self._control.box_area_deadband_ratio:
            forward_speed_m_s = 0.0
        else:
            forward_speed_m_s = self._control.forward_gain * area_error

        if area_ratio >= self._tracking.emergency_stop_area_ratio:
            forward_speed_m_s = min(forward_speed_m_s, 0.0)

        lateral_speed_m_s = 0.0
        if self._control.enable_lateral_motion:
            alignment_window_rad = math.radians(self._control.horizontal_alignment_window_deg)
            if abs(horizontal_rad) <= alignment_window_rad:
                lateral_speed_m_s = self._control.lateral_gain * horizontal_rad

        reason = "tracking target"
        if area_ratio < self._tracking.min_box_area_ratio:
            reason = "tracking small target"

        command = FollowCommand(
            velocity_forward_m_s=_clamp(
                forward_speed_m_s,
                -self._control.max_reverse_speed_m_s,
                self._control.max_forward_speed_m_s,
            ),
            velocity_right_m_s=_clamp(
                lateral_speed_m_s,
                -self._control.max_lateral_speed_m_s,
                self._control.max_lateral_speed_m_s,
            ),
            velocity_down_m_s=0.0,
            yaw_rate_rad_s=_clamp(
                yaw_rate_rad_s,
                -math.radians(self._control.max_yaw_rate_deg_s),
                math.radians(self._control.max_yaw_rate_deg_s),
            ),
            active=True,
            reason=reason,
        )
        return command

    def _ramp_to_zero(self, dt: float, reason: str) -> FollowCommand:
        """Return a zero-motion command reached through the rate limiter."""

        return self._apply_rate_limits(FollowCommand.zero(reason), dt)

    def _apply_rate_limits(self, target: FollowCommand, dt: float) -> FollowCommand:
        """Apply slew-rate limits to the commanded outputs."""

        max_speed_delta = self._control.speed_slew_limit_m_s2 * dt
        max_yaw_delta = math.radians(self._control.yaw_slew_limit_deg_s2) * dt
        limited = FollowCommand(
            velocity_forward_m_s=_rate_limit(
                self._last_command.velocity_forward_m_s,
                target.velocity_forward_m_s,
                max_speed_delta,
            ),
            velocity_right_m_s=_rate_limit(
                self._last_command.velocity_right_m_s,
                target.velocity_right_m_s,
                max_speed_delta,
            ),
            velocity_down_m_s=_rate_limit(
                self._last_command.velocity_down_m_s,
                target.velocity_down_m_s,
                max_speed_delta,
            ),
            yaw_rate_rad_s=_rate_limit(
                self._last_command.yaw_rate_rad_s,
                target.yaw_rate_rad_s,
                max_yaw_delta,
            ),
            active=target.active,
            reason=target.reason,
        )
        self._last_command = limited
        return limited


class OrbitController:
    """Generate a single conservative orbit around the selected person.

    The orbit is intentionally body-frame and target-relative: it commands a bounded
    lateral velocity, uses forward correction to hold stand-off, and adds yaw
    feed-forward plus image-based correction so the drone circles the person once
    before yielding back to normal follow behavior.
    """

    def __init__(
        self,
        camera: CameraConfig,
        tracking: TrackingConfig,
        control: ControlConfig,
        safety: SafetyConfig,
        orbit: OrbitConfig,
    ) -> None:
        """Initialize the orbit controller."""

        self._camera = camera
        self._tracking = tracking
        self._control = control
        self._safety = safety
        self._orbit = orbit
        self._active = False
        self._progress_rad = 0.0
        self._started_s: float | None = None
        self._last_update_s: float | None = None
        self._last_reason = "orbit idle"
        self._last_command = FollowCommand.zero("orbit idle")

    @property
    def status(self) -> OrbitStatus:
        """Return the current orbit state."""

        return OrbitStatus(
            active=self._active,
            progress_rad=self._progress_rad,
            reason=self._last_reason,
        )

    def start(self, now_s: float) -> None:
        """Begin a new one-shot orbit from the current target state."""

        self._active = True
        self._progress_rad = 0.0
        self._started_s = now_s
        self._last_update_s = now_s
        self._last_reason = "orbit active"
        self._last_command = FollowCommand.zero("orbit active")

    def abort(self, reason: str) -> None:
        """Abort the orbit and clear internal state."""

        self._active = False
        self._progress_rad = 0.0
        self._started_s = None
        self._last_update_s = None
        self._last_reason = reason
        self._last_command = FollowCommand.zero(reason)

    def step(
        self,
        observation: TargetObservation | None,
        now_s: float,
        follow_allowed: bool,
    ) -> FollowCommand | None:
        """Advance the orbit and return the override command when active.

        Args:
            observation: Current tracked target, or ``None`` if unavailable.
            now_s: Monotonic timestamp in seconds.
            follow_allowed: Whether external gates still permit autonomy.

        Returns:
            A maneuver override command while the orbit is active. Returns ``None``
            when the orbit has completed or is inactive so the caller can fall back
            to normal follow behavior.
        """

        if not self._active:
            return None
        if not follow_allowed:
            self.abort("orbit aborted: follow disabled")
            return None
        if observation is None:
            self.abort("orbit aborted: target missing")
            return FollowCommand.zero("orbit target missing")

        dt = self._compute_dt(now_s)
        if self._started_s is not None and (
            now_s - self._started_s >= self._orbit.max_duration_s
        ):
            self.abort("orbit timed out")
            return None

        command = self._compute_orbit_command(observation, dt)
        if self._progress_rad >= math.radians(self._orbit.completion_angle_deg):
            self.abort("orbit complete")
            return None
        return command

    def _compute_dt(self, now_s: float) -> float:
        """Compute a bounded orbit update period."""

        if self._last_update_s is None:
            self._last_update_s = now_s
            return 1.0 / self._control.loop_rate_hz
        dt = max(now_s - self._last_update_s, 1e-3)
        self._last_update_s = now_s
        return dt

    def _compute_orbit_command(
        self,
        observation: TargetObservation,
        dt: float,
    ) -> FollowCommand:
        """Build one conservative orbit step around the current target."""

        direction_sign = 1.0 if self._orbit.clockwise else -1.0
        stand_off_distance_m = max(self._safety.stand_off_distance_m, 0.5)
        angles = compute_target_angles(self._camera, observation)
        area_ratio = observation.bbox.area_ratio(self._camera.width, self._camera.height)
        area_error = self._tracking.desired_box_area_ratio - area_ratio
        forward_correction_m_s = _clamp(
            area_error * self._orbit.forward_correction_gain,
            -self._orbit.max_forward_correction_m_s,
            self._orbit.max_forward_correction_m_s,
        )
        if area_ratio >= self._tracking.emergency_stop_area_ratio:
            forward_correction_m_s = min(forward_correction_m_s, 0.0)

        yaw_feed_forward_rad_s = direction_sign * (
            self._orbit.lateral_speed_m_s / stand_off_distance_m
        )
        yaw_correction_rad_s = self._orbit.yaw_gain * angles.horizontal_rad
        command = FollowCommand(
            velocity_forward_m_s=forward_correction_m_s,
            velocity_right_m_s=direction_sign * self._orbit.lateral_speed_m_s,
            velocity_down_m_s=0.0,
            yaw_rate_rad_s=_clamp(
                yaw_feed_forward_rad_s + yaw_correction_rad_s,
                -math.radians(self._orbit.max_yaw_rate_deg_s),
                math.radians(self._orbit.max_yaw_rate_deg_s),
            ),
            active=True,
            reason="orbiting target",
        )
        self._progress_rad += abs(self._orbit.lateral_speed_m_s) * dt / stand_off_distance_m
        self._last_reason = "orbit active"
        return self._apply_rate_limits(command, dt)

    def _apply_rate_limits(self, target: FollowCommand, dt: float) -> FollowCommand:
        """Apply the standard conservative slew-rate limits to orbit commands."""

        max_speed_delta = self._control.speed_slew_limit_m_s2 * dt
        max_yaw_delta = math.radians(self._control.yaw_slew_limit_deg_s2) * dt
        limited = FollowCommand(
            velocity_forward_m_s=_rate_limit(
                self._last_command.velocity_forward_m_s,
                target.velocity_forward_m_s,
                max_speed_delta,
            ),
            velocity_right_m_s=_rate_limit(
                self._last_command.velocity_right_m_s,
                target.velocity_right_m_s,
                max_speed_delta,
            ),
            velocity_down_m_s=_rate_limit(
                self._last_command.velocity_down_m_s,
                target.velocity_down_m_s,
                max_speed_delta,
            ),
            yaw_rate_rad_s=_rate_limit(
                self._last_command.yaw_rate_rad_s,
                target.yaw_rate_rad_s,
                max_yaw_delta,
            ),
            active=target.active,
            reason=target.reason,
        )
        self._last_command = limited
        return limited
