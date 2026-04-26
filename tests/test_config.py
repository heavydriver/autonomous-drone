"""Unit tests for application configuration loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autonomous_drone.config import config_to_dict, load_config_file


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


if __name__ == "__main__":
    unittest.main()
