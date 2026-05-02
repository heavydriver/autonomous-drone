"""YOLO detection and ByteTrack tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from autonomous_drone.config import TrackingConfig
from autonomous_drone.models import BoundingBox, Detection, TargetObservation, Track


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
            device=self._device
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
