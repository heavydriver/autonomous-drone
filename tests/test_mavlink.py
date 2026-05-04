"""Unit tests for MAVLink transport selection and no-GPS stick output."""

from __future__ import annotations

import unittest

from autonomous_drone.config import MavlinkConfig
from autonomous_drone.mavlink import (
    MavlinkFollowerClient,
    _scale_manual_axis_to_pwm,
    _scale_manual_throttle_to_pwm,
)
from autonomous_drone.models import FollowCommand


class _FakeMav:
    """Record outbound MAVLink commands for assertions."""

    def __init__(self) -> None:
        self.manual_control_calls: list[tuple[int, int, int, int, int, int]] = []
        self.rc_override_calls: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []

    def manual_control_send(
        self,
        target_system: int,
        x: int,
        y: int,
        z: int,
        r: int,
        buttons: int,
    ) -> None:
        """Capture a ``MANUAL_CONTROL`` packet."""

        self.manual_control_calls.append((target_system, x, y, z, r, buttons))

    def rc_channels_override_send(
        self,
        target_system: int,
        target_component: int,
        chan1_raw: int,
        chan2_raw: int,
        chan3_raw: int,
        chan4_raw: int,
        chan5_raw: int,
        chan6_raw: int,
        chan7_raw: int,
        chan8_raw: int,
    ) -> None:
        """Capture an ``RC_CHANNELS_OVERRIDE`` packet."""

        self.rc_override_calls.append(
            (
                target_system,
                target_component,
                chan1_raw,
                chan2_raw,
                chan3_raw,
                chan4_raw,
                chan5_raw,
                chan6_raw,
                chan7_raw,
                chan8_raw,
            )
        )


class _FakeMaster:
    """Minimal MAVLink master stub exposing IDs and a fake sender."""

    def __init__(self) -> None:
        self.target_system = 1
        self.target_component = 1
        self.mav = _FakeMav()


class MavlinkManualControlTransportTest(unittest.TestCase):
    """Validate no-GPS stick command transport and safe release behavior."""

    def _make_client(self, *, use_rc_overrides: bool = True) -> MavlinkFollowerClient:
        client = object.__new__(MavlinkFollowerClient)
        client._config = MavlinkConfig(alt_hold_use_rc_overrides=use_rc_overrides)
        client._master = _FakeMaster()
        client._state = None
        client._sent_nonzero_command = False
        client._last_command_type = "velocity_body"
        client._last_manual_transport = "manual_control"
        return client

    def test_active_alt_hold_command_uses_rc_override_by_default(self) -> None:
        """Active no-GPS stick commands should be forwarded as RC overrides."""

        client = self._make_client()
        command = FollowCommand(
            velocity_forward_m_s=0.0,
            velocity_right_m_s=0.0,
            velocity_down_m_s=0.0,
            yaw_rate_rad_s=0.0,
            active=True,
            reason="track",
            command_type="manual_control",
            manual_pitch=0.25,
            manual_roll=-0.10,
            manual_throttle=0.50,
            manual_yaw=0.20,
        )

        client.send_follow_command(command)

        self.assertEqual(client._last_manual_transport, "rc_override")
        self.assertEqual(len(client._master.mav.rc_override_calls), 1)
        self.assertEqual(len(client._master.mav.manual_control_calls), 0)
        self.assertEqual(
            client._master.mav.rc_override_calls[0],
            (1, 1, 1450, 1625, 1500, 1600, 65535, 65535, 65535, 65535),
        )

    def test_inactive_alt_hold_command_releases_rc_overrides(self) -> None:
        """Inactive no-GPS commands should release overridden sticks immediately."""

        client = self._make_client()
        command = FollowCommand.neutral_manual_control(
            reason="target lost",
            active=False,
        )

        client.send_follow_command(command)

        self.assertEqual(client._last_manual_transport, "rc_override")
        self.assertEqual(
            client._master.mav.rc_override_calls[0],
            (1, 1, 0, 0, 0, 0, 65535, 65535, 65535, 65535),
        )

    def test_send_zero_once_releases_prior_rc_overrides(self) -> None:
        """Gating a moving no-GPS follower off should release the stick channels."""

        client = self._make_client()
        client._sent_nonzero_command = True
        client._last_command_type = "manual_control"
        client._last_manual_transport = "rc_override"

        client.send_zero_once(reason="follow disabled")

        self.assertFalse(client._sent_nonzero_command)
        self.assertEqual(
            client._master.mav.rc_override_calls[0],
            (1, 1, 0, 0, 0, 0, 65535, 65535, 65535, 65535),
        )

    def test_manual_control_fallback_remains_available(self) -> None:
        """Disabling RC override transport should keep the MANUAL_CONTROL path usable."""

        client = self._make_client(use_rc_overrides=False)
        command = FollowCommand(
            velocity_forward_m_s=0.0,
            velocity_right_m_s=0.0,
            velocity_down_m_s=0.0,
            yaw_rate_rad_s=0.0,
            active=True,
            reason="track",
            command_type="manual_control",
            manual_pitch=0.25,
            manual_roll=-0.10,
            manual_throttle=0.50,
            manual_yaw=0.20,
        )

        client.send_follow_command(command)

        self.assertEqual(client._last_manual_transport, "manual_control")
        self.assertEqual(len(client._master.mav.rc_override_calls), 0)
        self.assertEqual(
            client._master.mav.manual_control_calls[0],
            (1, 250, -100, 500, 200, 0),
        )


class MavlinkScalingTest(unittest.TestCase):
    """Validate normalized stick scaling for RC override output."""

    def test_axis_scaling_uses_trim_as_neutral(self) -> None:
        """Axis scaling should respect asymmetric RC spans around trim."""

        self.assertEqual(
            _scale_manual_axis_to_pwm(
                -1.0,
                min_pwm=1100,
                trim_pwm=1520,
                max_pwm=1910,
            ),
            1100,
        )
        self.assertEqual(
            _scale_manual_axis_to_pwm(
                0.0,
                min_pwm=1100,
                trim_pwm=1520,
                max_pwm=1910,
            ),
            1520,
        )
        self.assertEqual(
            _scale_manual_axis_to_pwm(
                1.0,
                min_pwm=1100,
                trim_pwm=1520,
                max_pwm=1910,
            ),
            1910,
        )

    def test_throttle_scaling_maps_zero_to_full_range(self) -> None:
        """Throttle scaling should cover the configured RC PWM range."""

        self.assertEqual(
            _scale_manual_throttle_to_pwm(
                0.0,
                min_pwm=1000,
                max_pwm=2000,
            ),
            1000,
        )
        self.assertEqual(
            _scale_manual_throttle_to_pwm(
                0.5,
                min_pwm=1000,
                max_pwm=2000,
            ),
            1500,
        )
        self.assertEqual(
            _scale_manual_throttle_to_pwm(
                1.0,
                min_pwm=1000,
                max_pwm=2000,
            ),
            2000,
        )


if __name__ == "__main__":
    unittest.main()
