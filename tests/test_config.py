"""Unit tests for application configuration loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autonomous_drone.config import MavlinkConfig, config_to_dict, load_config_file


class ConfigLoadTest(unittest.TestCase):
    """Validate JSON config loading and overrides."""

    def test_load_config_file_applies_nested_overrides(self) -> None:
        """Nested JSON config overrides should populate the dataclass tree."""

        payload = {
            "camera": {"mount_pitch_deg": 12.5},
            "tracking": {"model_path": "/tmp/model.pt"},
            "runtime": {"skip_rc_gate": True, "video_source": "test-pipeline"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config_file(config_path)

        config_dict = config_to_dict(config)
        self.assertEqual(config_dict["camera"]["mount_pitch_deg"], 12.5)
        self.assertEqual(config_dict["tracking"]["model_path"], "/tmp/model.pt")
        self.assertTrue(config_dict["runtime"]["skip_rc_gate"])
        self.assertEqual(config_dict["runtime"]["video_source"], "test-pipeline")

    def test_mavlink_udp_transport_resolves_to_connection_string(self) -> None:
        """UDP transport fields should build a pymavlink UDP endpoint."""

        config = MavlinkConfig(transport="udp", udp_host="127.0.0.1", udp_port=14551)

        self.assertEqual(config.resolved_connection_string(), "udp:127.0.0.1:14551")
        self.assertEqual(config.describe_endpoint(), "udp:127.0.0.1:14551")

    def test_mavlink_serial_transport_resolves_to_device_path(self) -> None:
        """Serial transport fields should resolve to the configured UART device."""

        config = MavlinkConfig(
            transport="serial",
            serial_device="/dev/ttyUSB0",
            baud_rate=921600,
        )

        self.assertEqual(config.resolved_connection_string(), "/dev/ttyUSB0")
        self.assertEqual(config.describe_endpoint(), "serial:/dev/ttyUSB0@921600")

    def test_raw_connection_string_override_wins(self) -> None:
        """A raw pymavlink connection string should bypass transport synthesis."""

        config = MavlinkConfig(
            connection_string="udpin:0.0.0.0:14550",
            transport="serial",
            serial_device="/dev/ttyUSB0",
        )

        self.assertEqual(config.resolved_connection_string(), "udpin:0.0.0.0:14550")
        self.assertEqual(config.describe_endpoint(), "raw:udpin:0.0.0.0:14550")


if __name__ == "__main__":
    unittest.main()
