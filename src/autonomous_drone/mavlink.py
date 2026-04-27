"""pymavlink integration for ArduPilot Guided-mode control."""

from __future__ import annotations

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
    """Minimal MAVLink client for conservative Guided-mode follow control."""

    _VELOCITY_YAWRATE_TYPE_MASK = 1479

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

    def compute_follow_gate(self) -> FollowGate:
        """Compute whether follow autonomy is allowed from mode and RC state."""

        state = self._state
        rc_value = state.rc_channel_pwm(self._config.follow_enable_channel)
        return FollowGate(
            guided_mode=state.mode.upper() == self._config.guided_mode_name.upper(),
            rc_switch_high=(rc_value is not None and rc_value >= self._config.follow_enable_high_pwm),
        )

    def send_follow_command(self, command: FollowCommand) -> None:
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
        self._sent_nonzero_command = any(
            abs(value) > 1e-6
            for value in (
                command.velocity_forward_m_s,
                command.velocity_right_m_s,
                command.velocity_down_m_s,
                command.yaw_rate_rad_s,
            )
        )

    def send_zero_once(self, reason: str = "stopping autonomy") -> None:
        """Send a single zero command if the client was previously moving the drone."""

        if not self._sent_nonzero_command:
            return
        self.send_follow_command(FollowCommand.zero(reason=reason))
        self._sent_nonzero_command = False

    def _request_streams(self) -> None:
        """Request telemetry streams from the autopilot."""

        self._master.mav.request_data_stream_send(
            self._master.target_system,
            self._master.target_component,
            self._mavutil.mavlink.MAV_DATA_STREAM_ALL,
            self._config.stream_rate_hz,
            1,
        )
