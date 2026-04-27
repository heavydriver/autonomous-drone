"""Unit tests for CLI-driven application configuration."""

from __future__ import annotations

import unittest

from autonomous_drone.app import build_config, parse_args


class AppCliConfigTest(unittest.TestCase):
    """Validate CLI overrides for MAVLink transport selection."""

    def test_serial_device_implies_serial_transport(self) -> None:
        """Providing only a serial device should switch the endpoint to UART mode."""

        args = parse_args(["--serial-device", "/dev/ttyUSB0"])

        config = build_config(args)

        self.assertEqual(config.mavlink.transport, "serial")
        self.assertEqual(config.mavlink.serial_device, "/dev/ttyUSB0")

    def test_transport_specific_flags_override_defaults(self) -> None:
        """Explicit serial transport flags should configure the UART endpoint."""

        args = parse_args(
            [
                "--transport",
                "serial",
                "--serial-device",
                "/dev/ttyUSB0",
                "--baud",
                "921600",
            ]
        )

        config = build_config(args)

        self.assertEqual(config.mavlink.transport, "serial")
        self.assertEqual(config.mavlink.serial_device, "/dev/ttyUSB0")
        self.assertEqual(config.mavlink.baud_rate, 921600)
        self.assertIsNone(config.mavlink.connection_string)

    def test_raw_connection_can_be_replaced_by_structured_flags(self) -> None:
        """Structured transport flags should clear a previously supplied raw URI."""

        args = parse_args(
            [
                "--connection",
                "udpin:0.0.0.0:14550",
                "--transport",
                "udp",
                "--udp-host",
                "192.168.1.20",
                "--udp-port",
                "14555",
            ]
        )

        config = build_config(args)

        self.assertIsNone(config.mavlink.connection_string)
        self.assertEqual(config.mavlink.transport, "udp")
        self.assertEqual(config.mavlink.udp_host, "192.168.1.20")
        self.assertEqual(config.mavlink.udp_port, 14555)


if __name__ == "__main__":
    unittest.main()
