"""YOLO detection and ByteTrack tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from autonomous_drone.config import TrackingConfig
from autonomous_drone.models import BoundingBox, Detection, TargetObservation, Track


@dataclass(slots=True)
class PrimaryTargetSelector:
    """Keep target identity stable by preferring the current lock."""

    active_track_id: int | None = None

    def select(self, tracks: Iterable[Track], now_s: float) -> TargetObservation | None:
        """Select the primary target from tracked people.

        Args:
            tracks: Current frame tracks.
            now_s: Monotonic timestamp in seconds.

        Returns:
            The selected target observation, or ``None`` if no tracks exist.
        """

        track_list = list(tracks)
        if not track_list:
            return None
        if self.active_track_id is not None:
            for track in track_list:
                if track.track_id == self.active_track_id:
                    return TargetObservation(
                        track_id=track.track_id,
                        bbox=track.bbox,
                        confidence=track.confidence,
                        timestamp_s=now_s,
                    )
        selected = max(track_list, key=lambda track: track.bbox.area)
        self.active_track_id = selected.track_id
        return TargetObservation(
            track_id=selected.track_id,
            bbox=selected.bbox,
            confidence=selected.confidence,
            timestamp_s=now_s,
        )


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
