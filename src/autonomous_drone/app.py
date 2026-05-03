"""CLI entrypoint for SITL-first person-follow experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from autonomous_drone.config import AppConfig, config_to_dict, load_config_file
from autonomous_drone.control import (
    FollowController,
    GuidedNoGpsFollowController,
    OrbitStatus,
    OrbitController,
)
from autonomous_drone.mavlink import MavlinkFollowerClient
from autonomous_drone.metrics import CsvRunLogger, GateSnapshot, TimingSnapshot
from autonomous_drone.models import VehicleState
from autonomous_drone.perception import (
    ByteTrackPersonTracker,
    PrimaryTargetSelector,
    PoseGestureResult,
    YoloPersonDetector,
    YoloPoseEstimator,
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
    parser.add_argument("--enable-hand-raise-circle", action="store_true")
    parser.add_argument("--pose-model")
    parser.add_argument("--pose-interval-s", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rc-gate", action="store_true")
    parser.add_argument("--enable-guided-nogps-follow", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--log-data", action="store_true")
    parser.add_argument("--log-output-dir")
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
    if args.enable_hand_raise_circle:
        config.pose.hand_raise_circle_enabled = True
    if args.pose_model is not None:
        config.pose.model_path = args.pose_model
    if args.pose_interval_s is not None:
        config.pose.inference_interval_s = args.pose_interval_s
    if args.dry_run:
        config.runtime.dry_run = True
    if args.skip_rc_gate:
        config.runtime.skip_rc_gate = True
    if args.enable_guided_nogps_follow:
        config.runtime.enable_guided_nogps_follow = True
    if args.visualize:
        config.runtime.visualize = True
    if args.log_data:
        config.runtime.log_data = True
    if args.log_output_dir is not None:
        config.runtime.log_output_dir = args.log_output_dir
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


def format_command_text(command) -> str:
    """Return a concise human-readable command summary for logs and overlays."""

    if command.command_type == "attitude":
        roll_deg = (
            0.0
            if command.attitude_roll_rad is None
            else command.attitude_roll_rad * (180.0 / 3.141592653589793)
        )
        pitch_deg = (
            0.0
            if command.attitude_pitch_rad is None
            else command.attitude_pitch_rad * (180.0 / 3.141592653589793)
        )
        yaw_deg = (
            0.0
            if command.attitude_yaw_rad is None
            else command.attitude_yaw_rad * (180.0 / 3.141592653589793)
        )
        climb = (
            0.5
            if command.climb_rate_fraction is None
            else command.climb_rate_fraction
        )
        return (
            "att "
            f"roll={roll_deg:+.1f}deg "
            f"pitch={pitch_deg:+.1f}deg "
            f"yaw={yaw_deg:+.1f}deg "
            f"climb={climb:.2f} "
            f"reason={command.reason}"
        )

    return (
        "cmd "
        f"vx={command.velocity_forward_m_s:+.2f} "
        f"vy={command.velocity_right_m_s:+.2f} "
        f"yaw={command.yaw_rate_rad_s:+.2f} "
        f"reason={command.reason}"
    )


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
    pose_status_text: str,
    orbit_status_text: str,
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
        format_command_text(command),
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
    cv2.putText(
        frame,
        pose_status_text,
        (10, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 255),
        2,
    )
    cv2.putText(
        frame,
        orbit_status_text,
        (10, 154),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (150, 255, 150),
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

    use_guided_nogps_follow = config.runtime.enable_guided_nogps_follow
    if use_guided_nogps_follow:
        controller = GuidedNoGpsFollowController(
            config.camera,
            config.tracking,
            config.control,
        )
        orbit_controller = None
        print("[follow] using GUIDED_NOGPS attitude follower")
    else:
        controller = FollowController(config.camera, config.tracking, config.control)
        orbit_controller = OrbitController(
            config.camera,
            config.tracking,
            config.control,
            config.safety,
            config.orbit,
        )
        print("[follow] using GUIDED body-frame velocity follower")
    detector = YoloPersonDetector(config.tracking, device=config.runtime.model_device)
    tracker = ByteTrackPersonTracker(config.tracking)
    selector = PrimaryTargetSelector(config=config.tracking)
    pose_estimator = None
    orbit_supported = orbit_controller is not None
    if config.pose.hand_raise_circle_enabled and orbit_supported:
        print(
            "[pose] enabling right-hand orbit trigger with "
            f"{config.pose.model_path}"
        )
        pose_estimator = YoloPoseEstimator(
            config.pose, device=config.runtime.model_device
        )
    elif config.pose.hand_raise_circle_enabled and not orbit_supported:
        print("[pose] hand-raise orbit trigger disabled in GUIDED_NOGPS follow mode")

    mavlink = None
    if not config.runtime.dry_run:
        print(
            "[mavlink] connecting to "
            f"{config.mavlink.describe_endpoint()}"
        )
        mavlink = MavlinkFollowerClient(config.mavlink)
        mavlink.connect()
        if use_guided_nogps_follow:
            print(
                "[mavlink] expecting flight mode "
                f"{config.mavlink.guided_nogps_mode_name}"
            )
        else:
            print(
                "[mavlink] expecting flight mode "
                f"{config.mavlink.guided_mode_name}"
            )

    cv2, capture = open_video_source(
        config.runtime.video_source, config.runtime.video_backend
    )
    frame_interval_s = 1.0 / config.control.loop_rate_hz
    last_log_s = 0.0
    last_frame_started_s: float | None = None
    last_pose_sample_s = float("-inf")
    last_orbit_trigger_s = float("-inf")
    hand_raise_latched = False
    last_pose_result = PoseGestureResult(
        sampled=False,
        right_hand_up=None,
        reason="pose feature disabled"
        if not config.pose.hand_raise_circle_enabled
        else "pose idle",
    )
    recorder = None
    logger = None
    if config.runtime.log_data:
        logger = CsvRunLogger(
            output_dir=Path(config.runtime.log_output_dir),
            desired_area_ratio=config.tracking.desired_box_area_ratio,
        )
        print(f"[logging] writing CSV logs to {logger.session_dir}")
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
            frame_index = logger.next_frame_index() if logger is not None else 0
            ok, frame = capture.read()
            frame_grab_latency_s = time.monotonic() - loop_started_s
            if not ok:
                print("Video source ended or failed.")
                break

            config.camera.width = int(frame.shape[1])
            config.camera.height = int(frame.shape[0])

            detection_started_s = time.monotonic()
            detections = detector.detect(frame)
            detection_latency_s = time.monotonic() - detection_started_s
            tracking_started_s = time.monotonic()
            tracks = tracker.update(detections)
            tracking_latency_s = time.monotonic() - tracking_started_s
            selection_started_s = time.monotonic()
            observation = selector.select(
                tracks,
                detections,
                now_s=loop_started_s,
            )
            selection_latency_s = time.monotonic() - selection_started_s
            area_ratio = None
            target_angles = None
            if observation is not None:
                area_ratio = observation.bbox.area_ratio(
                    config.camera.width,
                    config.camera.height,
                )
                target_angles = controller.compute_target_angles(observation)
            pose_result = PoseGestureResult(
                sampled=False,
                right_hand_up=last_pose_result.right_hand_up,
                reason=last_pose_result.reason,
            )
            orbit_status = (
                orbit_controller.status
                if orbit_controller is not None
                else OrbitStatus(
                    active=False,
                    progress_rad=0.0,
                    reason="orbit unsupported in GUIDED_NOGPS follow mode",
                )
            )

            gate_snapshot = GateSnapshot(
                follow_allowed=True,
                guided_mode=True,
                rc_switch_high=True,
                rc_channel_pwm=None,
                gate_text="dry-run",
            )
            vehicle_state = (
                VehicleState(mode=config.mavlink.guided_nogps_mode_name)
                if use_guided_nogps_follow
                else None
            )
            if mavlink is not None:
                vehicle_state = mavlink.poll_state()
                expected_mode_name = (
                    config.mavlink.guided_nogps_mode_name
                    if use_guided_nogps_follow
                    else config.mavlink.guided_mode_name
                )
                gate = mavlink.compute_follow_gate(expected_mode_name=expected_mode_name)
                if config.runtime.skip_rc_gate:
                    gate_snapshot = GateSnapshot(
                        follow_allowed=gate.guided_mode,
                        guided_mode=gate.guided_mode,
                        rc_switch_high=gate.rc_switch_high,
                        rc_channel_pwm=vehicle_state.rc_channel_pwm(
                            config.mavlink.follow_enable_channel
                        ),
                        gate_text=f"rc gate bypassed mode={vehicle_state.mode}",
                    )
                else:
                    gate_snapshot = GateSnapshot(
                        follow_allowed=gate.follow_allowed,
                        guided_mode=gate.guided_mode,
                        rc_switch_high=gate.rc_switch_high,
                        rc_channel_pwm=vehicle_state.rc_channel_pwm(
                            config.mavlink.follow_enable_channel
                        ),
                        gate_text=(
                            f"mode_ok={gate.guided_mode} "
                            f"mode={vehicle_state.mode} "
                            f"rc_high={gate.rc_switch_high}"
                        ),
                    )

            if orbit_controller is not None and orbit_status.active and not gate_snapshot.follow_allowed:
                orbit_controller.abort("orbit aborted: follow disabled")
                orbit_status = orbit_controller.status

            if config.pose.hand_raise_circle_enabled and pose_estimator is not None:
                if orbit_status.active:
                    pose_result = PoseGestureResult(
                        sampled=False,
                        right_hand_up=last_pose_result.right_hand_up,
                        reason="pose paused during orbit",
                    )
                elif observation is None:
                    pose_result = PoseGestureResult(
                        sampled=False,
                        right_hand_up=last_pose_result.right_hand_up,
                        reason="pose waiting for target",
                    )
                elif (
                    loop_started_s - last_pose_sample_s
                    >= config.pose.inference_interval_s
                ):
                    last_pose_sample_s = loop_started_s
                    pose_result = pose_estimator.estimate_for_observation(
                        frame,
                        observation,
                        frame_width=config.camera.width,
                        frame_height=config.camera.height,
                    )
                    last_pose_result = pose_result
                    if pose_result.right_hand_up is False:
                        hand_raise_latched = False
                    if (
                        pose_result.right_hand_up is True
                        and not hand_raise_latched
                        and (
                            loop_started_s - last_orbit_trigger_s
                            >= config.pose.trigger_cooldown_s
                        )
                    ):
                        if orbit_controller is not None:
                            orbit_controller.start(loop_started_s)
                            orbit_status = orbit_controller.status
                            last_orbit_trigger_s = loop_started_s
                            hand_raise_latched = True
                            print("[orbit] triggered by raised right hand")

            control_started_s = time.monotonic()
            orbit_command = None
            if orbit_controller is not None:
                orbit_command = orbit_controller.step(
                    observation,
                    loop_started_s,
                    follow_allowed=gate_snapshot.follow_allowed,
                )
            if orbit_command is not None:
                if use_guided_nogps_follow:
                    command = controller.step(
                        observation,
                        loop_started_s,
                        follow_allowed=False,
                        vehicle_state=vehicle_state,
                    )
                else:
                    controller.step(
                        observation,
                        loop_started_s,
                        follow_allowed=False,
                    )
                    command = orbit_command
            elif use_guided_nogps_follow:
                command = controller.step(
                    observation,
                    loop_started_s,
                    follow_allowed=gate_snapshot.follow_allowed,
                    vehicle_state=vehicle_state,
                )
            else:
                command = controller.step(
                    observation,
                    loop_started_s,
                    follow_allowed=gate_snapshot.follow_allowed,
                )
            control_latency_s = time.monotonic() - control_started_s
            if orbit_controller is not None:
                orbit_status = orbit_controller.status

            mavlink_started_s = time.monotonic()
            if mavlink is not None:
                if gate_snapshot.follow_allowed:
                    mavlink.send_follow_command(command)
                else:
                    mavlink.send_zero_once(reason=command.reason)
            mavlink_latency_s = time.monotonic() - mavlink_started_s

            total_loop_latency_s = time.monotonic() - loop_started_s
            dt_s = (
                frame_interval_s
                if last_frame_started_s is None
                else max(loop_started_s - last_frame_started_s, 1e-3)
            )
            last_frame_started_s = loop_started_s

            if logger is not None:
                logger.log_detections(
                    frame_index=frame_index,
                    detections=detections,
                    frame_width=config.camera.width,
                    frame_height=config.camera.height,
                )
                logger.log_tracks(
                    frame_index=frame_index,
                    tracks=tracks,
                    frame_width=config.camera.width,
                    frame_height=config.camera.height,
                )
                logger.log_frame(
                    frame_index=frame_index,
                    now_s=loop_started_s,
                    dt_s=dt_s,
                    frame_width=config.camera.width,
                    frame_height=config.camera.height,
                    detections_count=len(detections),
                    tracks_count=len(tracks),
                    observation=observation,
                    target_angles=target_angles,
                    command=command,
                    gate=gate_snapshot,
                    vehicle_state=vehicle_state,
                    pose_sampled=pose_result.sampled,
                    pose_right_hand_up=pose_result.right_hand_up,
                    pose_reason=pose_result.reason,
                    orbit_active=orbit_status.active,
                    orbit_progress_deg=orbit_status.progress_rad * (180.0 / 3.141592653589793),
                    orbit_reason=orbit_status.reason,
                    timings=TimingSnapshot(
                        frame_grab_latency_s=frame_grab_latency_s,
                        detection_latency_s=detection_latency_s,
                        tracking_latency_s=tracking_latency_s,
                        selection_latency_s=selection_latency_s,
                        control_latency_s=control_latency_s,
                        mavlink_latency_s=mavlink_latency_s,
                        total_loop_latency_s=total_loop_latency_s,
                    ),
                )

            if loop_started_s - last_log_s >= 1.0:
                last_log_s = loop_started_s
                print(
                    f"[follow] active={command.active} "
                    f"detections={len(detections)} tracks={len(tracks)} "
                    f"target={'yes' if observation else 'no'} "
                    f"{format_command_text(command)} "
                    f"gate={gate_snapshot.gate_text} "
                    f"orbit_active={orbit_status.active}"
                )

            if config.runtime.visualize or recorder is not None:
                draw_overlay(
                    cv2,
                    frame,
                    observation,
                    command,
                    gate_snapshot.follow_allowed,
                    gate_snapshot.gate_text,
                    detection_count=len(detections),
                    track_count=len(tracks),
                    area_ratio=area_ratio,
                    pose_status_text=(
                        f"pose sampled={pose_result.sampled} "
                        f"hand_up={pose_result.right_hand_up} "
                        f"reason={pose_result.reason}"
                    ),
                    orbit_status_text=(
                        f"orbit active={orbit_status.active} "
                        f"progress_deg={orbit_status.progress_rad * (180.0 / 3.141592653589793):.0f} "
                        f"reason={orbit_status.reason}"
                    ),
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
        if logger is not None:
            logger.close()
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
