"""pymavlink integration for ArduPilot Guided and Guided_NoGPS control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic

from autonomous_drone.config import MavlinkConfig
from autonomous_drone.models import FollowCommand, VehicleState


@dataclass(slots=True)
class FollowGate:
    """Computed autonomy gate derived from RC state and current mode."""

    guided_mode: bool
    rc_switch_high: bool

    @property
    def follow_allowed(self) -> bool:
        """Return whether follow autonomy is allowed."""

        return self.guided_mode and self.rc_switch_high


class MavlinkFollowerClient:
    """Minimal MAVLink client for conservative follow control."""

    _VELOCITY_YAWRATE_TYPE_MASK = 1479
    _ATTITUDE_IGNORE_BODY_RATES_MASK = 0x07

    def __init__(self, config: MavlinkConfig) -> None:
        """Create a MAVLink client.

        Args:
            config: MAVLink connection and RC gate configuration.
        """

        try:
            from pymavlink import mavutil
        except ImportError as exc:  # pragma: no cover - import path depends on user env
            raise RuntimeError(
                "pymavlink is required to use the MAVLink client. "
                "Install project dependencies before running live control."
            ) from exc

        self._config = config
        self._mavutil = mavutil
        self._master = mavutil.mavlink_connection(
            config.resolved_connection_string(),
            baud=config.baud_rate,
            autoreconnect=True,
        )
        self._state = VehicleState()
        self._sent_nonzero_command = False
        self._last_command_type = "velocity_body"

    def connect(self, timeout_s: float = 30.0) -> None:
        """Wait for heartbeat and request telemetry streams."""

        self._master.wait_heartbeat(timeout=timeout_s)
        self._request_streams()

    def poll_state(self) -> VehicleState:
        """Poll MAVLink messages and update the latest vehicle state."""

        while True:
            message = self._master.recv_match(blocking=False)
            if message is None:
                break
            message_type = message.get_type()
            if message_type == "BAD_DATA":
                continue
            if message_type == "HEARTBEAT":
                self._state.mode = self._mavutil.mode_string_v10(message)
                armed_flag = self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                self._state.armed = bool(message.base_mode & armed_flag)
                self._state.last_heartbeat_monotonic_s = monotonic()
            elif message_type == "RC_CHANNELS":
                rc_channels = {}
                for channel in range(1, 19):
                    rc_value = getattr(message, f"chan{channel}_raw", 0)
                    if rc_value:
                        rc_channels[channel] = int(rc_value)
                self._state.rc_channels = rc_channels
            elif message_type == "ATTITUDE":
                self._state.roll_rad = float(message.roll)
                self._state.pitch_rad = float(message.pitch)
                self._state.yaw_rad = float(message.yaw)
        return self._state

    def compute_follow_gate(self, expected_mode_name: str | None = None) -> FollowGate:
        """Compute whether follow autonomy is allowed from mode and RC state."""

        state = self._state
        rc_value = state.rc_channel_pwm(self._config.follow_enable_channel)
        mode_name = expected_mode_name or self._config.guided_mode_name
        return FollowGate(
            guided_mode=state.mode.upper() == mode_name.upper(),
            rc_switch_high=(
                rc_value is not None
                and rc_value >= self._config.follow_enable_high_pwm
            ),
        )

    def send_follow_command(self, command: FollowCommand) -> None:
        """Send the requested follow command using the matching MAVLink primitive."""

        if command.command_type == "attitude":
            self._send_attitude_command(command)
            return
        self._send_velocity_command(command)

    def send_zero_once(self, reason: str = "stopping autonomy") -> None:
        """Send a single neutral command if the client was previously moving the drone."""

        if not self._sent_nonzero_command:
            return
        if self._last_command_type == "attitude":
            self.send_follow_command(
                FollowCommand.neutral_attitude(
                    reason=reason,
                    yaw_rad=self._state.yaw_rad,
                )
            )
        else:
            self.send_follow_command(FollowCommand.zero(reason=reason))
        self._sent_nonzero_command = False

    def _send_velocity_command(self, command: FollowCommand) -> None:
        """Send a body-frame velocity command with yaw rate in Guided mode."""

        self._master.mav.set_position_target_local_ned_send(
            0,
            self._master.target_system,
            self._master.target_component,
            self._mavutil.mavlink.MAV_FRAME_BODY_NED,
            self._VELOCITY_YAWRATE_TYPE_MASK,
            0.0,
            0.0,
            0.0,
            command.velocity_forward_m_s,
            command.velocity_right_m_s,
            command.velocity_down_m_s,
            0.0,
            0.0,
            0.0,
            0.0,
            command.yaw_rate_rad_s,
        )
        self._last_command_type = "velocity_body"
        self._sent_nonzero_command = any(
            abs(value) > 1e-6
            for value in (
                command.velocity_forward_m_s,
                command.velocity_right_m_s,
                command.velocity_down_m_s,
                command.yaw_rate_rad_s,
            )
        )

    def _send_attitude_command(self, command: FollowCommand) -> None:
        """Send an absolute attitude target for ``GUIDED_NOGPS`` follow."""

        if (
            command.attitude_roll_rad is None
            or command.attitude_pitch_rad is None
            or command.attitude_yaw_rad is None
            or command.climb_rate_fraction is None
        ):
            raise ValueError("Attitude follow command is missing attitude fields")

        self._master.mav.set_attitude_target_send(
            0,
            self._master.target_system,
            self._master.target_component,
            self._ATTITUDE_IGNORE_BODY_RATES_MASK,
            _euler_to_quaternion(
                command.attitude_roll_rad,
                command.attitude_pitch_rad,
                command.attitude_yaw_rad,
            ),
            0.0,
            0.0,
            0.0,
            _clamp_fraction(command.climb_rate_fraction),
        )
        self._last_command_type = "attitude"
        self._sent_nonzero_command = (
            abs(command.attitude_roll_rad) > 1e-6
            or abs(command.attitude_pitch_rad) > 1e-6
            or abs(_wrap_angle(command.attitude_yaw_rad - self._state.yaw_rad)) > 1e-3
            or abs(command.climb_rate_fraction - 0.5) > 1e-6
        )

    def _request_streams(self) -> None:
        """Request telemetry streams from the autopilot."""

        self._master.mav.request_data_stream_send(
            self._master.target_system,
            self._master.target_component,
            self._mavutil.mavlink.MAV_DATA_STREAM_ALL,
            self._config.stream_rate_hz,
            1,
        )


def _euler_to_quaternion(
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> tuple[float, float, float, float]:
    """Convert roll, pitch, yaw Euler angles into a MAVLink quaternion."""

    half_roll = roll_rad * 0.5
    half_pitch = pitch_rad * 0.5
    half_yaw = yaw_rad * 0.5
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _wrap_angle(angle_rad: float) -> float:
    """Wrap an angle into ``[-pi, pi)``."""

    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _clamp_fraction(value: float) -> float:
    """Clamp a climb-rate or thrust fraction into the MAVLink-supported range."""

    return max(0.0, min(1.0, value))
