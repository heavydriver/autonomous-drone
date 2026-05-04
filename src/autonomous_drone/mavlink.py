"""pymavlink integration for ArduPilot Guided and no-GPS follow control."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
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
            source_system=config.source_system,
            source_component=config.source_component,
        )
        self._state = VehicleState()
        self._sent_nonzero_command = False
        self._last_command_type = "velocity_body"
        self._last_manual_transport = "manual_control"
        self._parameter_cache: dict[str, float] = {}

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
            self._handle_message(message)
        return self._state

    def manual_control_transport_name(self) -> str:
        """Return the active transport name for no-GPS stick control."""

        if self._use_rc_overrides_for_manual_control():
            return "RC_CHANNELS_OVERRIDE"
        return "MANUAL_CONTROL"

    def manual_control_preflight_warnings(self) -> list[str]:
        """Return warnings for common ArduPilot RC-input rejection causes."""

        parameter_names = [
            "MAV_GCS_SYSID",
            "MAV_GCS_SYSID_HI",
            "MAV_OPTIONS",
            "RC_OPTIONS",
        ]
        if self._config.alt_hold_use_rc_overrides:
            parameter_names.extend(
                (
                    "RCMAP_ROLL",
                    "RCMAP_PITCH",
                    "RCMAP_THROTTLE",
                    "RCMAP_YAW",
                )
            )
        parameters = {
            name: value
            for name in parameter_names
            if (value := self._fetch_parameter_value(name)) is not None
        }
        return _manual_control_preflight_warnings(
            self._config,
            using_rc_overrides=self._use_rc_overrides_for_manual_control(),
            parameters=parameters,
        )

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
        if command.command_type == "manual_control":
            if not command.active and self._use_rc_overrides_for_manual_control():
                self._release_rc_overrides()
                return
            self._send_manual_control_command(command)
            return
        self._send_velocity_command(command)

    def send_zero_once(self, reason: str = "stopping autonomy") -> None:
        """Stop the previous command once, or release RC overrides if applicable."""

        if not self._sent_nonzero_command:
            return
        if self._last_command_type == "attitude":
            self.send_follow_command(
                FollowCommand.neutral_attitude(
                    reason=reason,
                    yaw_rad=self._state.yaw_rad,
                )
            )
        elif self._last_command_type == "manual_control":
            if self._last_manual_transport == "rc_override":
                self._release_rc_overrides()
            else:
                self.send_follow_command(
                    FollowCommand.neutral_manual_control(
                        reason=reason,
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

    def _send_manual_control_command(self, command: FollowCommand) -> None:
        """Send normalized pilot stick inputs for ``ALT_HOLD`` follow."""

        if (
            command.manual_pitch is None
            or command.manual_roll is None
            or command.manual_throttle is None
            or command.manual_yaw is None
        ):
            raise ValueError("Manual-control follow command is missing stick fields")

        if self._use_rc_overrides_for_manual_control():
            self._send_rc_override_command(command)
            return

        self._master.mav.manual_control_send(
            self._master.target_system,
            _scale_manual_axis(command.manual_pitch),
            _scale_manual_axis(command.manual_roll),
            _scale_manual_throttle(command.manual_throttle),
            _scale_manual_axis(command.manual_yaw),
            0,
        )
        self._last_command_type = "manual_control"
        self._last_manual_transport = "manual_control"
        self._sent_nonzero_command = (
            abs(command.manual_pitch) > 1e-6
            or abs(command.manual_roll) > 1e-6
            or abs(command.manual_yaw) > 1e-6
            or abs(command.manual_throttle - 0.5) > 1e-6
        )

    def _send_rc_override_command(self, command: FollowCommand) -> None:
        """Send ``ALT_HOLD`` stick inputs as RC channel overrides."""

        overrides = {
            self._config.rc_override_roll_channel: _scale_manual_axis_to_pwm(
                command.manual_roll,
                min_pwm=self._config.rc_override_min_pwm,
                trim_pwm=self._config.rc_override_trim_pwm,
                max_pwm=self._config.rc_override_max_pwm,
            ),
            self._config.rc_override_pitch_channel: _scale_manual_axis_to_pwm(
                command.manual_pitch,
                min_pwm=self._config.rc_override_min_pwm,
                trim_pwm=self._config.rc_override_trim_pwm,
                max_pwm=self._config.rc_override_max_pwm,
            ),
            self._config.rc_override_throttle_channel: _scale_manual_throttle_to_pwm(
                command.manual_throttle,
                min_pwm=self._config.rc_override_min_pwm,
                max_pwm=self._config.rc_override_max_pwm,
            ),
            self._config.rc_override_yaw_channel: _scale_manual_axis_to_pwm(
                command.manual_yaw,
                min_pwm=self._config.rc_override_min_pwm,
                trim_pwm=self._config.rc_override_trim_pwm,
                max_pwm=self._config.rc_override_max_pwm,
            ),
        }
        self._send_rc_override_payload(overrides)
        self._last_command_type = "manual_control"
        self._last_manual_transport = "rc_override"
        self._sent_nonzero_command = (
            abs(command.manual_pitch) > 1e-6
            or abs(command.manual_roll) > 1e-6
            or abs(command.manual_yaw) > 1e-6
            or abs(command.manual_throttle - 0.5) > 1e-6
        )

    def _release_rc_overrides(self) -> None:
        """Release any RC channels used for follow back to the pilot/receiver."""

        release_values = {
            self._config.rc_override_roll_channel: _rc_override_release_value(
                self._config.rc_override_roll_channel
            ),
            self._config.rc_override_pitch_channel: _rc_override_release_value(
                self._config.rc_override_pitch_channel
            ),
            self._config.rc_override_throttle_channel: _rc_override_release_value(
                self._config.rc_override_throttle_channel
            ),
            self._config.rc_override_yaw_channel: _rc_override_release_value(
                self._config.rc_override_yaw_channel
            ),
        }
        self._send_rc_override_payload(release_values)
        self._last_command_type = "manual_control"
        self._last_manual_transport = "rc_override"
        self._sent_nonzero_command = False

    def _send_rc_override_payload(self, overrides: dict[int, int]) -> None:
        """Transmit a sparse RC override payload using the supported field count."""

        sender = self._master.mav.rc_channels_override_send
        channel_field_count = _rc_override_channel_field_count(sender)
        channel_values = [
            _rc_override_ignore_value(channel_index)
            for channel_index in range(1, channel_field_count + 1)
        ]
        for channel, value in overrides.items():
            if 1 <= channel <= channel_field_count:
                channel_values[channel - 1] = value
        sender(
            self._master.target_system,
            self._master.target_component,
            *channel_values,
        )

    def _use_rc_overrides_for_manual_control(self) -> bool:
        """Return whether ``ALT_HOLD`` follow should drive sticks via RC override."""

        return self._config.alt_hold_use_rc_overrides and hasattr(
            self._master.mav,
            "rc_channels_override_send",
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

    def _handle_message(self, message: object) -> None:
        """Update cached state from a received MAVLink message."""

        message_type = message.get_type()
        if message_type == "BAD_DATA":
            return
        if message_type == "HEARTBEAT":
            self._state.mode = self._mavutil.mode_string_v10(message)
            armed_flag = self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            self._state.armed = bool(message.base_mode & armed_flag)
            self._state.last_heartbeat_monotonic_s = monotonic()
            return
        if message_type == "RC_CHANNELS":
            rc_channels = {}
            for channel in range(1, 19):
                rc_value = getattr(message, f"chan{channel}_raw", 0)
                if rc_value:
                    rc_channels[channel] = int(rc_value)
            self._state.rc_channels = rc_channels
            return
        if message_type == "ATTITUDE":
            self._state.roll_rad = float(message.roll)
            self._state.pitch_rad = float(message.pitch)
            self._state.yaw_rad = float(message.yaw)
            return
        if message_type == "PARAM_VALUE":
            parameter_name = getattr(message, "param_id", "")
            if isinstance(parameter_name, bytes):
                parameter_name = parameter_name.decode("utf-8", errors="ignore")
            parameter_name = str(parameter_name).rstrip("\x00").upper()
            if parameter_name:
                self._parameter_cache[parameter_name] = float(message.param_value)

    def _fetch_parameter_value(
        self,
        name: str,
        timeout_s: float = 1.0,
    ) -> float | None:
        """Fetch one parameter value from the autopilot if available."""

        normalized_name = name.upper()
        cached = self._parameter_cache.get(normalized_name)
        if cached is not None:
            return cached
        if not hasattr(self._master, "param_fetch_one"):
            return None

        self._master.param_fetch_one(normalized_name)
        deadline = monotonic() + max(timeout_s, 0.1)
        while monotonic() < deadline:
            remaining_s = max(deadline - monotonic(), 0.0)
            message = self._master.recv_match(
                blocking=True,
                timeout=remaining_s,
            )
            if message is None:
                continue
            self._handle_message(message)
            cached = self._parameter_cache.get(normalized_name)
            if cached is not None:
                return cached
        return self._parameter_cache.get(normalized_name)


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


def _scale_manual_axis(value: float) -> int:
    """Scale a normalized axis from ``[-1, 1]`` to MAVLink MANUAL_CONTROL units."""

    clamped = max(-1.0, min(1.0, value))
    return int(round(clamped * 1000.0))


def _scale_manual_throttle(value: float) -> int:
    """Scale normalized throttle from ``[0, 1]`` to MAVLink MANUAL_CONTROL units."""

    return int(round(_clamp_fraction(value) * 1000.0))


def _scale_manual_axis_to_pwm(
    value: float,
    *,
    min_pwm: int,
    trim_pwm: int,
    max_pwm: int,
) -> int:
    """Scale a normalized stick axis from ``[-1, 1]`` into RC override PWM."""

    clamped = max(-1.0, min(1.0, value))
    if clamped >= 0.0:
        return int(round(trim_pwm + clamped * (max_pwm - trim_pwm)))
    return int(round(trim_pwm + clamped * (trim_pwm - min_pwm)))


def _scale_manual_throttle_to_pwm(
    value: float,
    *,
    min_pwm: int,
    max_pwm: int,
) -> int:
    """Scale normalized throttle from ``[0, 1]`` into RC override PWM."""

    clamped = _clamp_fraction(value)
    return int(round(min_pwm + clamped * (max_pwm - min_pwm)))


def _rc_override_channel_field_count(sender: object) -> int:
    """Return how many RC channel fields the bound sender accepts."""

    parameters = tuple(inspect.signature(sender).parameters)
    return max(0, len(parameters) - 2)


def _rc_override_ignore_value(channel: int) -> int:
    """Return the MAVLink 'ignore this RC channel' sentinel value."""

    return 65535


def _rc_override_release_value(channel: int) -> int:
    """Return the MAVLink 'release this RC channel' sentinel value."""

    if channel <= 8:
        return 0
    return 65534


def _manual_control_preflight_warnings(
    config: MavlinkConfig,
    *,
    using_rc_overrides: bool,
    parameters: Mapping[str, float],
) -> list[str]:
    """Return startup warnings for common ArduPilot RC-input pitfalls."""

    warnings: list[str] = []
    mav_options = int(round(parameters.get("MAV_OPTIONS", 0.0)))
    accepted_gcs_sysid = int(round(parameters.get("MAV_GCS_SYSID", 255.0)))
    accepted_gcs_sysid_hi = int(round(parameters.get("MAV_GCS_SYSID_HI", -1.0)))
    if mav_options & 0x01:
        if accepted_gcs_sysid_hi >= accepted_gcs_sysid:
            sysid_ok = accepted_gcs_sysid <= config.source_system <= accepted_gcs_sysid_hi
        else:
            sysid_ok = config.source_system == accepted_gcs_sysid
        if not sysid_ok:
            accepted_text = (
                str(accepted_gcs_sysid)
                if accepted_gcs_sysid_hi < accepted_gcs_sysid
                else f"{accepted_gcs_sysid}-{accepted_gcs_sysid_hi}"
            )
            warnings.append(
                "ArduPilot is filtering MAVLink pilot-input messages by GCS "
                f"sysid. MAV_OPTIONS bit 0 is set, source sysid={config.source_system}, "
                f"accepted sysid range={accepted_text}. RC overrides and MANUAL_CONTROL "
                "will be ignored until they match."
            )

    if using_rc_overrides:
        rc_options = int(round(parameters.get("RC_OPTIONS", 0.0)))
        if rc_options & 0x02:
            warnings.append(
                "RC_OPTIONS bit 1 is set, so ArduPilot will ignore "
                "RC_CHANNELS_OVERRIDE messages. Disable that bit or run the app with "
                "--disable-alt-hold-rc-overrides to use MANUAL_CONTROL instead."
            )

        rcmap_expectations = (
            ("roll", "RCMAP_ROLL", config.rc_override_roll_channel),
            ("pitch", "RCMAP_PITCH", config.rc_override_pitch_channel),
            ("throttle", "RCMAP_THROTTLE", config.rc_override_throttle_channel),
            ("yaw", "RCMAP_YAW", config.rc_override_yaw_channel),
        )
        for axis_name, parameter_name, configured_channel in rcmap_expectations:
            mapped_channel = parameters.get(parameter_name)
            if mapped_channel is None:
                continue
            mapped_channel_int = int(round(mapped_channel))
            if mapped_channel_int != configured_channel:
                warnings.append(
                    "RC override channel mapping mismatch: "
                    f"{axis_name} is mapped to RC channel {mapped_channel_int} in "
                    f"ArduPilot, but the app is overriding channel {configured_channel}. "
                    "Update the config rc_override_*_channel values or switch to "
                    "MANUAL_CONTROL for no-GPS follow."
                )
    return warnings
