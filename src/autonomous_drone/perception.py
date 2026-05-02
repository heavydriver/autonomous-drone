"""YOLO detection, pose estimation, and ByteTrack tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from autonomous_drone.config import PoseConfig, TrackingConfig
from autonomous_drone.models import BoundingBox, Detection, TargetObservation, Track


_COCO_RIGHT_SHOULDER_INDEX = 6
_COCO_RIGHT_ELBOW_INDEX = 8
_COCO_RIGHT_WRIST_INDEX = 10


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    """Single pose keypoint in image pixel coordinates."""

    x_px: float
    y_px: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PoseGestureResult:
    """Result of one gesture-oriented pose estimate."""

    sampled: bool
    right_hand_up: bool | None
    reason: str


def detect_right_hand_up(
    keypoints: Sequence[PoseKeypoint],
    bbox: BoundingBox,
    config: PoseConfig,
) -> PoseGestureResult:
    """Classify whether the right hand is raised from pose landmarks.

    Args:
        keypoints: Pose keypoints in full-frame pixel coordinates.
        bbox: Bounding box used to normalize geometric thresholds.
        config: Pose-estimation thresholds for conservative classification.

    Returns:
        A gesture result describing whether the right hand is raised.
    """

    required_indices = (
        _COCO_RIGHT_SHOULDER_INDEX,
        _COCO_RIGHT_ELBOW_INDEX,
        _COCO_RIGHT_WRIST_INDEX,
    )
    if len(keypoints) <= max(required_indices):
        return PoseGestureResult(
            sampled=True,
            right_hand_up=None,
            reason="pose keypoints incomplete",
        )

    right_shoulder = keypoints[_COCO_RIGHT_SHOULDER_INDEX]
    right_elbow = keypoints[_COCO_RIGHT_ELBOW_INDEX]
    right_wrist = keypoints[_COCO_RIGHT_WRIST_INDEX]
    min_confidence = config.min_keypoint_confidence
    if (
        right_shoulder.confidence < min_confidence
        or right_elbow.confidence < min_confidence
        or right_wrist.confidence < min_confidence
    ):
        return PoseGestureResult(
            sampled=True,
            right_hand_up=None,
            reason="right arm keypoints low confidence",
        )

    shoulder_margin_px = bbox.height * config.wrist_above_shoulder_margin_ratio
    elbow_margin_px = bbox.height * config.wrist_above_elbow_margin_ratio
    max_horizontal_offset_px = bbox.width * config.max_wrist_shoulder_offset_ratio
    wrist_above_shoulder = (
        right_wrist.y_px + shoulder_margin_px < right_shoulder.y_px
    )
    wrist_above_elbow = right_wrist.y_px + elbow_margin_px < right_elbow.y_px
    aligned_with_shoulder = (
        abs(right_wrist.x_px - right_shoulder.x_px) <= max_horizontal_offset_px
    )
    if wrist_above_shoulder and wrist_above_elbow and aligned_with_shoulder:
        return PoseGestureResult(
            sampled=True,
            right_hand_up=True,
            reason="right hand raised",
        )
    return PoseGestureResult(
        sampled=True,
        right_hand_up=False,
        reason="right hand not raised",
    )


@dataclass(slots=True)
class PrimaryTargetSelector:
    """Keep target identity stable by preferring the current lock."""

    config: TrackingConfig
    active_track_id: int | None = None
    _last_bbox: BoundingBox | None = None
    _next_virtual_track_id: int = -1

    def select(
        self,
        tracks: Iterable[Track],
        detections: Iterable[Detection],
        now_s: float,
    ) -> TargetObservation | None:
        """Select the primary target from tracked people.

        Args:
            tracks: Current frame tracks.
            detections: Current frame raw detections.
            now_s: Monotonic timestamp in seconds.

        Returns:
            The selected target observation, or ``None`` if no plausible target exists.
        """

        track_list = list(tracks)
        detection_list = list(detections)
        if not track_list and not detection_list:
            return None
        if self.active_track_id is not None:
            for track in track_list:
                if track.track_id == self.active_track_id:
                    return self._build_observation(
                        bbox=track.bbox,
                        confidence=track.confidence,
                        now_s=now_s,
                    )

            reacquired_track = self._find_best_track_match(track_list)
            if reacquired_track is not None:
                return self._build_observation(
                    bbox=reacquired_track.bbox,
                    confidence=reacquired_track.confidence,
                    now_s=now_s,
                )

            fallback_detection = self._find_best_detection_match(detection_list)
            if fallback_detection is not None:
                return self._build_observation(
                    bbox=fallback_detection.bbox,
                    confidence=fallback_detection.confidence,
                    now_s=now_s,
                )

        if not track_list:
            if detection_list and self.active_track_id is None:
                selected_detection = max(detection_list, key=lambda detection: detection.bbox.area)
                self.active_track_id = self._allocate_virtual_track_id()
                return self._build_observation(
                    bbox=selected_detection.bbox,
                    confidence=selected_detection.confidence,
                    now_s=now_s,
                )
            return None

        selected = max(track_list, key=lambda track: track.bbox.area)
        self.active_track_id = selected.track_id
        return self._build_observation(
            bbox=selected.bbox,
            confidence=selected.confidence,
            now_s=now_s,
        )

    def _build_observation(
        self,
        bbox: BoundingBox,
        confidence: float,
        now_s: float,
    ) -> TargetObservation:
        """Store selector state and return a controller observation."""

        self._last_bbox = bbox
        if self.active_track_id is None:
            raise RuntimeError("active_track_id must be set before building an observation")
        return TargetObservation(
            track_id=self.active_track_id,
            bbox=bbox,
            confidence=confidence,
            timestamp_s=now_s,
        )

    def _allocate_virtual_track_id(self) -> int:
        """Return a synthetic track id for detection-only target locks."""

        track_id = self._next_virtual_track_id
        self._next_virtual_track_id -= 1
        return track_id

    def _find_best_track_match(self, tracks: Sequence[Track]) -> Track | None:
        """Return the track that best matches the last locked target footprint."""

        if self._last_bbox is None:
            return None
        best_match: Track | None = None
        best_score = -1.0
        for track in tracks:
            score = self._match_score(track.bbox)
            if score > best_score:
                best_match = track
                best_score = score
        return best_match

    def _find_best_detection_match(
        self, detections: Sequence[Detection]
    ) -> Detection | None:
        """Return a raw detection that plausibly continues the current target."""

        if self._last_bbox is None:
            return None
        best_match: Detection | None = None
        best_score = -1.0
        for detection in detections:
            if detection.confidence < self.config.detector_confidence:
                continue
            score = self._match_score(detection.bbox)
            if score > best_score:
                best_match = detection
                best_score = score
        return best_match

    def _match_score(self, candidate_bbox: BoundingBox) -> float:
        """Return a positive score for plausible target-continuation candidates."""

        if self._last_bbox is None:
            return -1.0
        overlap = self._last_bbox.intersection_over_union(candidate_bbox)
        center_shift_x = abs(candidate_bbox.center_x - self._last_bbox.center_x) / max(
            self._last_bbox.width, 1.0
        )
        center_shift_y = abs(candidate_bbox.center_y - self._last_bbox.center_y) / max(
            self._last_bbox.height, 1.0
        )
        if (
            overlap < self.config.selector_reacquire_min_iou
            and center_shift_x > self.config.selector_reacquire_max_center_shift_ratio
        ):
            return -1.0
        if (
            overlap < self.config.selector_reacquire_min_iou
            and center_shift_y > self.config.selector_reacquire_max_center_shift_ratio
        ):
            return -1.0
        return overlap - 0.1 * (center_shift_x + center_shift_y)


class YoloPersonDetector:
    """YOLO detector wrapper that returns only person detections."""

    def __init__(self, config: TrackingConfig, device: str = "cpu") -> None:
        """Load a YOLO model for person detection.

        Args:
            config: Tracking and detector thresholds.
            device: Inference device understood by Ultralytics.
        """

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "ultralytics is required for detection. Install dependencies first."
            ) from exc

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                "YOLO model weights were not found at "
                f"'{model_path}'. Provide a local weights file with "
                "--model /path/to/model.pt or set tracking.model_path in your config file."
            )

        self._config = config
        self._device = device
        self._model = YOLO(str(model_path))

    def detect(self, frame) -> List[Detection]:
        """Run the detector on one image frame."""

        results = self._model.predict(
            source=frame,
            conf=self._config.detector_confidence,
            verbose=False,
            device=self._device,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = [int(value) for value in boxes.cls.cpu().tolist()]
        for box, confidence, class_id in zip(xyxy, confidences, class_ids):
            if class_id != self._config.person_class_id:
                continue
            detections.append(
                Detection(
                    bbox=BoundingBox(*map(float, box)),
                    confidence=float(confidence),
                    class_id=class_id,
                )
            )
        return detections


class YoloPoseEstimator:
    """YOLO pose wrapper for conservative right-hand-raise detection."""

    def __init__(self, config: PoseConfig, device: str = "cpu") -> None:
        """Load a YOLO pose model for cropped single-person estimation.

        Args:
            config: Pose-estimation thresholds and model location.
            device: Inference device understood by Ultralytics.
        """

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "ultralytics is required for pose estimation. Install dependencies first."
            ) from exc

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                "YOLO pose weights were not found at "
                f"'{model_path}'. Provide a local weights file with "
                "--pose-model /path/to/model.pt or set pose.model_path in your config file."
            )

        self._config = config
        self._device = device
        self._model = YOLO(str(model_path))

    def estimate_for_observation(
        self,
        frame,
        observation: TargetObservation,
        frame_width: int,
        frame_height: int,
    ) -> PoseGestureResult:
        """Estimate whether the selected person's right hand is raised.

        Args:
            frame: Full RGB or BGR frame understood by the model.
            observation: Selected tracked person to crop and analyze.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            A gesture result for the selected person crop.
        """

        bbox = observation.bbox
        if bbox.area_ratio(frame_width, frame_height) < self._config.min_bbox_area_ratio:
            return PoseGestureResult(
                sampled=True,
                right_hand_up=None,
                reason="target too small for pose",
            )

        crop_bounds = self._expanded_crop_bounds(bbox, frame_width, frame_height)
        if crop_bounds is None:
            return PoseGestureResult(
                sampled=True,
                right_hand_up=None,
                reason="invalid pose crop",
            )
        x1, y1, x2, y2 = crop_bounds
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return PoseGestureResult(
                sampled=True,
                right_hand_up=None,
                reason="empty pose crop",
            )

        results = self._model.predict(
            source=crop,
            verbose=False,
            device=self._device,
        )
        if not results:
            return PoseGestureResult(
                sampled=True,
                right_hand_up=None,
                reason="pose model returned no result",
            )
        keypoints = results[0].keypoints
        if keypoints is None or keypoints.xy is None:
            return PoseGestureResult(
                sampled=True,
                right_hand_up=None,
                reason="pose keypoints unavailable",
            )

        xy = keypoints.xy.cpu().tolist()
        confidence_values = (
            keypoints.conf.cpu().tolist() if keypoints.conf is not None else None
        )
        if not xy:
            return PoseGestureResult(
                sampled=True,
                right_hand_up=None,
                reason="pose keypoints unavailable",
            )

        pose_index = 0
        if confidence_values is not None:
            pose_scores = [sum(pose_conf) for pose_conf in confidence_values]
            pose_index = max(range(len(pose_scores)), key=pose_scores.__getitem__)

        selected_xy = xy[pose_index]
        selected_confidence = (
            confidence_values[pose_index]
            if confidence_values is not None
            else [1.0] * len(selected_xy)
        )
        full_frame_keypoints = [
            PoseKeypoint(
                x_px=float(point_xy[0] + x1),
                y_px=float(point_xy[1] + y1),
                confidence=float(point_confidence),
            )
            for point_xy, point_confidence in zip(selected_xy, selected_confidence)
        ]
        return detect_right_hand_up(
            keypoints=full_frame_keypoints,
            bbox=bbox,
            config=self._config,
        )

    def _expanded_crop_bounds(
        self,
        bbox: BoundingBox,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        """Return an expanded crop around the tracked person."""

        margin_x = bbox.width * self._config.crop_margin_ratio
        margin_y = bbox.height * self._config.crop_margin_ratio
        x1 = max(0, int(bbox.x1 - margin_x))
        y1 = max(0, int(bbox.y1 - margin_y))
        x2 = min(frame_width, int(bbox.x2 + margin_x))
        y2 = min(frame_height, int(bbox.y2 + margin_y))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)


class ByteTrackPersonTracker:
    """ByteTrack wrapper built on the supervision package."""

    def __init__(self, config: TrackingConfig) -> None:
        """Create a ByteTrack tracker for person detections."""

        try:
            import numpy as np
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "supervision and numpy are required for ByteTrack."
            ) from exc

        self._config = config
        self._np = np
        self._sv = sv
        self._tracker = sv.ByteTrack(
            frame_rate=config.tracker_frame_rate,
            lost_track_buffer=config.tracker_track_buffer,
            minimum_matching_threshold=config.tracker_match_thresh,
        )

    def update(self, detections: Iterable[Detection]) -> List[Track]:
        """Update the tracker from a detector batch."""

        detection_list = list(detections)
        if not detection_list:
            return []

        xyxy = self._np.array(
            [[det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2] for det in detection_list],
            dtype=float,
        )
        confidence = self._np.array([det.confidence for det in detection_list], dtype=float)
        class_id = self._np.array([det.class_id for det in detection_list], dtype=int)
        sv_detections = self._sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        tracked = self._tracker.update_with_detections(sv_detections)

        tracks: List[Track] = []
        tracker_ids = tracked.tracker_id.tolist() if tracked.tracker_id is not None else []
        confidence_values = tracked.confidence.tolist() if tracked.confidence is not None else []
        class_ids = tracked.class_id.tolist() if tracked.class_id is not None else []
        for box, conf, cls_id, track_id in zip(
            tracked.xyxy.tolist(),
            confidence_values,
            class_ids,
            tracker_ids,
        ):
            if cls_id != self._config.person_class_id or track_id is None:
                continue
            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=BoundingBox(*map(float, box)),
                    confidence=float(conf),
                    class_id=int(cls_id),
                )
            )
        return tracks
