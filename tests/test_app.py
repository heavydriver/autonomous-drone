"""Unit tests for CLI-driven application configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autonomous_drone.app import AnnotatedClipRecorder, build_config, parse_args


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

    def test_recording_flags_enable_annotated_clip_output(self) -> None:
        """Recording flags should opt in to annotated video clip capture."""

        args = parse_args(
            [
                "--record-annotated-video",
                "--recording-output-dir",
                "captures",
                "--recording-clip-duration-s",
                "45",
            ]
        )

        config = build_config(args)

        self.assertTrue(config.runtime.record_annotated_video)
        self.assertEqual(config.runtime.recording_output_dir, "captures")
        self.assertEqual(config.runtime.recording_clip_duration_s, 45.0)

    def test_log_flags_enable_csv_logging(self) -> None:
        """Logging flags should opt in to CSV metrics output."""

        args = parse_args(
            [
                "--log-data",
                "--log-output-dir",
                "logs/custom",
            ]
        )

        config = build_config(args)

        self.assertTrue(config.runtime.log_data)
        self.assertEqual(config.runtime.log_output_dir, "logs/custom")

    def test_pose_flags_enable_hand_raise_circle(self) -> None:
        """Pose flags should enable the optional hand-raise orbit feature."""

        args = parse_args(
            [
                "--enable-hand-raise-circle",
                "--pose-model",
                "/tmp/yolo11n-pose.pt",
                "--pose-interval-s",
                "6.5",
            ]
        )

        config = build_config(args)

        self.assertTrue(config.pose.hand_raise_circle_enabled)
        self.assertEqual(config.pose.model_path, "/tmp/yolo11n-pose.pt")
        self.assertEqual(config.pose.inference_interval_s, 6.5)


class _FakeFrame:
    """Minimal frame object exposing the shape OpenCV code expects."""

    shape = (720, 1280, 3)


class _FakeWriter:
    """Simple in-memory stand-in for OpenCV's ``VideoWriter``."""

    def __init__(self, path: str, fps: float, frame_size: tuple[int, int]) -> None:
        self.path = path
        self.fps = fps
        self.frame_size = frame_size
        self.frames: list[object] = []
        self.released = False

    def isOpened(self) -> bool:
        """Report that the writer initialized successfully."""

        return True

    def write(self, frame: object) -> None:
        """Record a frame write for assertions."""

        self.frames.append(frame)

    def release(self) -> None:
        """Mark the writer as released."""

        self.released = True


class _FakeCv2:
    """Small subset of the OpenCV API needed by ``AnnotatedClipRecorder``."""

    def __init__(self) -> None:
        self.writers: list[_FakeWriter] = []

    def VideoWriter_fourcc(self, *_chars: str) -> int:
        """Return a placeholder codec identifier."""

        return 0

    def VideoWriter(
        self,
        path: str,
        _fourcc: int,
        fps: float,
        frame_size: tuple[int, int],
    ) -> _FakeWriter:
        """Create a fake writer and remember it for later inspection."""

        writer = _FakeWriter(path=path, fps=fps, frame_size=frame_size)
        self.writers.append(writer)
        return writer


class AnnotatedClipRecorderTest(unittest.TestCase):
    """Validate clip rotation for annotated video recording."""

    def test_recorder_rotates_clips_after_thirty_seconds(self) -> None:
        """Frames at or beyond the duration limit should start a new clip."""

        fake_cv2 = _FakeCv2()
        frame = _FakeFrame()
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = AnnotatedClipRecorder(
                cv2=fake_cv2,
                output_dir=Path(tmp_dir),
                clip_duration_s=30.0,
                fps=10.0,
            )

            recorder.write_frame(frame, now_s=0.0)
            recorder.write_frame(frame, now_s=29.9)
            recorder.write_frame(frame, now_s=30.0)
            recorder.close()

        self.assertEqual(len(fake_cv2.writers), 2)
        self.assertEqual(len(fake_cv2.writers[0].frames), 2)
        self.assertEqual(len(fake_cv2.writers[1].frames), 1)
        self.assertTrue(fake_cv2.writers[0].released)
        self.assertTrue(fake_cv2.writers[1].released)
        self.assertEqual(fake_cv2.writers[0].frame_size, (1280, 720))


if __name__ == "__main__":
    unittest.main()
