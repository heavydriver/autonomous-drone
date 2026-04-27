"""CLI entrypoint for SITL-first person-follow experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from autonomous_drone.config import AppConfig, config_to_dict, load_config_file
from autonomous_drone.control import FollowController
from autonomous_drone.mavlink import MavlinkFollowerClient
from autonomous_drone.perception import (
    ByteTrackPersonTracker,
    PrimaryTargetSelector,
    YoloPersonDetector,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Person follower")
    parser.add_argument("--config")
    parser.add_argument("--connection")
    parser.add_argument("--transport", choices=("udp", "serial"))
    parser.add_argument("--udp-host")
    parser.add_argument("--udp-port", type=int)
    parser.add_argument("--serial-device")
    parser.add_argument("--baud", type=int)
    parser.add_argument("--video-source")
    parser.add_argument("--video-backend", choices=("auto", "gstreamer"))
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--mount-pitch-deg", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rc-gate", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--record-annotated-video", action="store_true")
    parser.add_argument("--recording-output-dir")
    parser.add_argument("--recording-clip-duration-s", type=float)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> AppConfig:
    """Build an application config from CLI arguments."""

    config = load_config_file(args.config) if args.config else AppConfig()
    explicit_transport = args.transport is not None
    if args.connection is not None:
        config.mavlink.connection_string = args.connection
    if args.transport is not None:
        config.mavlink.transport = args.transport
        config.mavlink.connection_string = None
    if args.udp_host is not None:
        if not explicit_transport:
            config.mavlink.transport = "udp"
        config.mavlink.udp_host = args.udp_host
        config.mavlink.connection_string = None
    if args.udp_port is not None:
        if not explicit_transport:
            config.mavlink.transport = "udp"
        config.mavlink.udp_port = args.udp_port
        config.mavlink.connection_string = None
    if args.serial_device is not None:
        if not explicit_transport:
            config.mavlink.transport = "serial"
        config.mavlink.serial_device = args.serial_device
        config.mavlink.connection_string = None
    if args.baud is not None:
        config.mavlink.baud_rate = args.baud
    if args.model is not None:
        config.tracking.model_path = args.model
    if args.video_source is not None:
        config.runtime.video_source = args.video_source
    if args.video_backend is not None:
        config.runtime.video_backend = args.video_backend
    if args.device is not None:
        config.runtime.model_device = args.device
    if args.mount_pitch_deg is not None:
        config.camera.mount_pitch_deg = args.mount_pitch_deg
    if args.dry_run:
        config.runtime.dry_run = True
    if args.skip_rc_gate:
        config.runtime.skip_rc_gate = True
    if args.visualize:
        config.runtime.visualize = True
    if args.record_annotated_video:
        config.runtime.record_annotated_video = True
    if args.recording_output_dir is not None:
        config.runtime.recording_output_dir = args.recording_output_dir
    if args.recording_clip_duration_s is not None:
        config.runtime.recording_clip_duration_s = args.recording_clip_duration_s
    return config


def open_video_source(video_source: str, backend: str):
    """Open an OpenCV video source.

    Args:
        video_source: Camera index or backend-specific source string.
        backend: Backend selection, such as ``auto`` or ``gstreamer``.

    Returns:
        A tuple of the imported ``cv2`` module and an opened ``VideoCapture``.

    Raises:
        RuntimeError: If OpenCV is unavailable or the source cannot be opened.
    """

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "opencv-python is required to open the video source."
        ) from exc

    if video_source.isdigit():
        source = int(video_source)
    else:
        source = video_source
    if backend == "gstreamer":
        capture = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Unable to open video source with backend={backend!r}: {video_source}"
        )
    return cv2, capture


def draw_overlay(
    cv2,
    frame,
    observation,
    command,
    follow_allowed,
    gate_text: str,
    detection_count: int,
    track_count: int,
    area_ratio: float | None,
) -> None:
    """Draw lightweight debugging overlays for local testing."""

    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(
        frame, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20
    )
    if observation is not None:
        bbox = observation.bbox
        cv2.rectangle(
            frame,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"id={observation.track_id} conf={observation.confidence:.2f}",
            (int(bbox.x1), max(20, int(bbox.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.putText(
        frame,
        f"follow_allowed={follow_allowed} {gate_text}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        (
            "cmd "
            f"vx={command.velocity_forward_m_s:+.2f} "
            f"vy={command.velocity_right_m_s:+.2f} "
            f"yaw={command.yaw_rate_rad_s:+.2f} "
            f"reason={command.reason}"
        ),
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"detections={detection_count} tracks={track_count}",
        (10, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 255, 200),
        2,
    )
    area_ratio_text = f"{area_ratio:.3f}" if area_ratio is not None else "n/a"
    cv2.putText(
        frame,
        f"area_ratio={area_ratio_text}",
        (10, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 220, 255),
        2,
    )


class AnnotatedClipRecorder:
    """Write annotated frames into fixed-duration video clips.

    Args:
        cv2: Imported OpenCV module used to create video writers.
        output_dir: Directory where clip files are written.
        clip_duration_s: Maximum duration of each clip in seconds.
        fps: Output frame rate for the encoded video.

    Raises:
        ValueError: If ``clip_duration_s`` or ``fps`` are not positive.
    """

    def __init__(
        self,
        cv2,
        output_dir: Path,
        clip_duration_s: float,
        fps: float,
    ) -> None:
        if clip_duration_s <= 0.0:
            raise ValueError("recording clip duration must be greater than zero")
        if fps <= 0.0:
            raise ValueError("recording fps must be greater than zero")

        self._cv2 = cv2
        self._output_dir = output_dir
        self._clip_duration_s = clip_duration_s
        self._fps = fps
        self._writer = None
        self._clip_started_s: float | None = None
        self._clip_index = 0
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write_frame(self, frame, now_s: float) -> None:
        """Append a frame to the current clip, rotating clips as needed.

        Args:
            frame: Annotated OpenCV image to encode.
            now_s: Monotonic timestamp for the current frame.
        """

        if self._needs_new_clip(now_s):
            self._start_new_clip(frame, now_s)
        self._writer.write(frame)

    def close(self) -> None:
        """Release the current writer, if one is active."""

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._clip_started_s = None

    def _needs_new_clip(self, now_s: float) -> bool:
        """Return ``True`` when a new output clip should be created."""

        if self._writer is None or self._clip_started_s is None:
            return True
        return (now_s - self._clip_started_s) >= self._clip_duration_s

    def _start_new_clip(self, frame, now_s: float) -> None:
        """Open a new writer sized to the current frame."""

        self.close()
        height, width = frame.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = self._output_dir / (
            f"annotated-{timestamp}-{self._clip_index:04d}.mp4"
        )
        fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
        writer = self._cv2.VideoWriter(
            str(output_path),
            fourcc,
            self._fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Unable to open annotated video writer: {output_path}")

        self._writer = writer
        self._clip_started_s = now_s
        self._clip_index += 1
        print(f"[recording] writing annotated clip {output_path}")


def run(config: AppConfig) -> int:
    """Run the main follow loop."""

    controller = FollowController(config.camera, config.tracking, config.control)
    detector = YoloPersonDetector(config.tracking, device=config.runtime.model_device)
    tracker = ByteTrackPersonTracker(config.tracking)
    selector = PrimaryTargetSelector(config=config.tracking)
    mavlink = None
    if not config.runtime.dry_run:
        print(
            "[mavlink] connecting to "
            f"{config.mavlink.describe_endpoint()}"
        )
        mavlink = MavlinkFollowerClient(config.mavlink)
        mavlink.connect()

    cv2, capture = open_video_source(
        config.runtime.video_source, config.runtime.video_backend
    )
    frame_interval_s = 1.0 / config.control.loop_rate_hz
    last_log_s = 0.0
    recorder = None
    if config.runtime.record_annotated_video:
        recorder = AnnotatedClipRecorder(
            cv2=cv2,
            output_dir=Path(config.runtime.recording_output_dir),
            clip_duration_s=config.runtime.recording_clip_duration_s,
            fps=config.control.loop_rate_hz,
        )

    try:
        while True:
            loop_started_s = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                print("Video source ended or failed.")
                break

            config.camera.width = int(frame.shape[1])
            config.camera.height = int(frame.shape[0])

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            observation = selector.select(
                tracks,
                detections,
                now_s=loop_started_s,
            )
            area_ratio = None
            if observation is not None:
                area_ratio = observation.bbox.area_ratio(
                    config.camera.width,
                    config.camera.height,
                )

            gate_text = "dry-run"
            follow_allowed = True
            if mavlink is not None:
                mavlink.poll_state()
                gate = mavlink.compute_follow_gate()
                if config.runtime.skip_rc_gate:
                    follow_allowed = gate.guided_mode
                    gate_text = "rc gate bypassed"
                else:
                    follow_allowed = gate.follow_allowed
                    gate_text = (
                        f"mode_guided={gate.guided_mode} rc_high={gate.rc_switch_high}"
                    )

            command = controller.step(
                observation, loop_started_s, follow_allowed=follow_allowed
            )

            if mavlink is not None:
                if follow_allowed:
                    mavlink.send_follow_command(command)
                else:
                    mavlink.send_zero_once(reason=command.reason)

            if loop_started_s - last_log_s >= 1.0:
                last_log_s = loop_started_s
                print(
                    f"[follow] active={command.active} reason={command.reason} "
                    f"detections={len(detections)} tracks={len(tracks)} "
                    f"target={'yes' if observation else 'no'} "
                    f"vx={command.velocity_forward_m_s:+.2f} "
                    f"vy={command.velocity_right_m_s:+.2f} "
                    f"yaw={command.yaw_rate_rad_s:+.2f} "
                    f"gate={gate_text}"
                )

            if config.runtime.visualize or recorder is not None:
                draw_overlay(
                    cv2,
                    frame,
                    observation,
                    command,
                    follow_allowed,
                    gate_text,
                    detection_count=len(detections),
                    track_count=len(tracks),
                    area_ratio=area_ratio,
                )

            if recorder is not None:
                recorder.write_frame(frame, now_s=loop_started_s)

            if config.runtime.visualize:
                cv2.imshow("drone-follower", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            elapsed_s = time.monotonic() - loop_started_s
            remaining_s = frame_interval_s - elapsed_s
            if remaining_s > 0.0:
                time.sleep(remaining_s)
    finally:
        capture.release()
        if recorder is not None:
            recorder.close()
        if config.runtime.visualize:
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    config = build_config(args)
    if args.print_config:
        print(json.dumps(config_to_dict(config), indent=2, sort_keys=True))
        return 0
    try:
        return run(config)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
