"""CLI entrypoint for SITL-first person-follow experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time

from autonomous_drone.config import AppConfig, config_to_dict, load_config_file
from autonomous_drone.control import FollowController
from autonomous_drone.mavlink import MavlinkFollowerClient
from autonomous_drone.perception import (
    ByteTrackPersonTracker,
    PrimaryTargetSelector,
    YoloPersonDetector,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Conservative ArduPilot person follower")
    parser.add_argument("--config")
    parser.add_argument("--connection")
    parser.add_argument("--baud", type=int)
    parser.add_argument("--video-source")
    parser.add_argument("--video-backend", choices=("auto", "gstreamer"))
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--mount-pitch-deg", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rc-gate", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    """Build an application config from CLI arguments."""

    config = load_config_file(args.config) if args.config else AppConfig()
    if args.connection is not None:
        config.mavlink.connection_string = args.connection
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
    return config


def open_video_source(video_source: str, backend: str):
    """Open an OpenCV video source."""

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
        raise RuntimeError(f"Unable to open video source: {video_source}")
    return cv2, capture


def draw_overlay(cv2, frame, observation, command, follow_allowed, gate_text: str) -> None:
    """Draw lightweight debugging overlays for local testing."""

    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(frame, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20)
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


def run(config: AppConfig) -> int:
    """Run the main follow loop."""

    controller = FollowController(config.camera, config.tracking, config.control)
    detector = YoloPersonDetector(config.tracking, device=config.runtime.model_device)
    tracker = ByteTrackPersonTracker(config.tracking)
    selector = PrimaryTargetSelector()
    mavlink = None
    if not config.runtime.dry_run:
        mavlink = MavlinkFollowerClient(config.mavlink)
        mavlink.connect()

    cv2, capture = open_video_source(config.runtime.video_source, config.runtime.video_backend)
    frame_interval_s = 1.0 / config.control.loop_rate_hz
    last_log_s = 0.0

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
        observation = selector.select(tracks, now_s=loop_started_s)

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
                gate_text = f"mode_guided={gate.guided_mode} rc_high={gate.rc_switch_high}"

        command = controller.step(observation, loop_started_s, follow_allowed=follow_allowed)

        if mavlink is not None:
            if follow_allowed:
                mavlink.send_follow_command(command)
            else:
                mavlink.send_zero_once(reason=command.reason)

        if loop_started_s - last_log_s >= 1.0:
            last_log_s = loop_started_s
            print(
                f"[follow] active={command.active} reason={command.reason} "
                f"target={'yes' if observation else 'no'} "
                f"vx={command.velocity_forward_m_s:+.2f} "
                f"vy={command.velocity_right_m_s:+.2f} "
                f"yaw={command.yaw_rate_rad_s:+.2f} "
                f"gate={gate_text}"
            )

        if config.runtime.visualize:
            draw_overlay(cv2, frame, observation, command, follow_allowed, gate_text)
            cv2.imshow("drone-follower", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        elapsed_s = time.monotonic() - loop_started_s
        remaining_s = frame_interval_s - elapsed_s
        if remaining_s > 0.0:
            time.sleep(remaining_s)

    capture.release()
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
