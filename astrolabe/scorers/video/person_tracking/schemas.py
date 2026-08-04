"""Validated, JSON-friendly schemas for person detection and tracking outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _validate_finite_sequence(name: str, values: Sequence[float], length: int) -> None:
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if not all(isfinite(float(value)) for value in values):
        raise ValueError(f"{name} values must be finite")


def _validate_detection(
    *,
    class_id: int,
    class_name: str,
    confidence: float,
    bbox_xyxy: Sequence[float],
    bbox_xywh: Sequence[float],
    bbox_xyxy_normalized: Sequence[float],
    bbox_area_ratio: float,
) -> None:
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
        raise ValueError("class_id must be a non-negative integer")
    if not class_name:
        raise ValueError("class_name must not be empty")
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    _validate_finite_sequence("bbox_xyxy", bbox_xyxy, 4)
    _validate_finite_sequence("bbox_xywh", bbox_xywh, 4)
    _validate_finite_sequence("bbox_xyxy_normalized", bbox_xyxy_normalized, 4)
    x1, y1, x2, y2 = bbox_xyxy
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must satisfy 0 <= x1 < x2 and 0 <= y1 < y2")
    x, y, width, height = bbox_xywh
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("bbox_xywh must use a non-negative top-left and positive size")
    nx1, ny1, nx2, ny2 = bbox_xyxy_normalized
    if not (0.0 <= nx1 < nx2 <= 1.0 and 0.0 <= ny1 < ny2 <= 1.0):
        raise ValueError("normalized bbox must be ordered and within [0, 1]")
    if not isfinite(bbox_area_ratio) or not 0.0 < bbox_area_ratio <= 1.0:
        raise ValueError("bbox_area_ratio must be in (0, 1]")


def _bbox_representations(
    bbox_xyxy: Sequence[float], image_width: int, image_height: int
) -> Tuple[List[float], List[float], List[float], float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    _validate_finite_sequence("bbox_xyxy", bbox_xyxy, 4)
    x1 = min(max(float(bbox_xyxy[0]), 0.0), float(image_width))
    y1 = min(max(float(bbox_xyxy[1]), 0.0), float(image_height))
    x2 = min(max(float(bbox_xyxy[2]), 0.0), float(image_width))
    y2 = min(max(float(bbox_xyxy[3]), 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox is empty after clipping to the original image")
    width, height = x2 - x1, y2 - y1
    xyxy = [x1, y1, x2, y2]
    xywh = [x1, y1, width, height]
    normalized = [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
    area_ratio = (width * height) / (image_width * image_height)
    return xyxy, xywh, normalized, area_ratio


@dataclass(frozen=True)
class RawDetection:
    """One YOLO person candidate before identity association."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    bbox_xyxy_normalized: List[float]
    bbox_area_ratio: float
    detection_index: int

    def __post_init__(self) -> None:
        _validate_detection(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            bbox_xyxy=self.bbox_xyxy,
            bbox_xywh=self.bbox_xywh,
            bbox_xyxy_normalized=self.bbox_xyxy_normalized,
            bbox_area_ratio=self.bbox_area_ratio,
        )
        if isinstance(self.detection_index, bool) or not isinstance(self.detection_index, int):
            raise ValueError("detection_index must be a non-negative integer")
        if self.detection_index < 0:
            raise ValueError("detection_index must be a non-negative integer")

    @classmethod
    def from_xyxy(
        cls,
        *,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox_xyxy: Sequence[float],
        image_width: int,
        image_height: int,
        detection_index: int,
    ) -> "RawDetection":
        xyxy, xywh, normalized, area_ratio = _bbox_representations(
            bbox_xyxy, image_width, image_height
        )
        return cls(
            class_id=class_id,
            class_name=class_name,
            confidence=float(confidence),
            bbox_xyxy=xyxy,
            bbox_xywh=xywh,
            bbox_xyxy_normalized=normalized,
            bbox_area_ratio=area_ratio,
            detection_index=detection_index,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackedDetection:
    """One ByteTrack-associated person observation."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    bbox_xyxy_normalized: List[float]
    bbox_area_ratio: float
    source_detection_index: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if self.source_detection_index is not None and (
            isinstance(self.source_detection_index, bool)
            or not isinstance(self.source_detection_index, int)
            or self.source_detection_index < 0
        ):
            raise ValueError("source_detection_index must be None or a non-negative integer")
        _validate_detection(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            bbox_xyxy=self.bbox_xyxy,
            bbox_xywh=self.bbox_xywh,
            bbox_xyxy_normalized=self.bbox_xyxy_normalized,
            bbox_area_ratio=self.bbox_area_ratio,
        )

    @classmethod
    def from_xyxy(
        cls,
        *,
        track_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox_xyxy: Sequence[float],
        image_width: int,
        image_height: int,
        source_detection_index: Optional[int] = None,
    ) -> "TrackedDetection":
        xyxy, xywh, normalized, area_ratio = _bbox_representations(
            bbox_xyxy, image_width, image_height
        )
        return cls(
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            confidence=float(confidence),
            bbox_xyxy=xyxy,
            bbox_xywh=xywh,
            bbox_xyxy_normalized=normalized,
            bbox_area_ratio=area_ratio,
            source_detection_index=source_detection_index,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Compatibility alias for milestone-1 callers.
Detection = TrackedDetection


@dataclass(frozen=True)
class FrameDetections:
    """Raw YOLO and successfully associated ByteTrack observations for one frame."""

    frame_index: int
    timestamp_sec: float
    raw_detections: List[RawDetection] = field(default_factory=list)
    tracked_detections: List[TrackedDetection] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not isfinite(self.timestamp_sec) or self.timestamp_sec < 0:
            raise ValueError("timestamp_sec must be finite and non-negative")

    @property
    def detections(self) -> List[TrackedDetection]:
        """Compatibility view of milestone-1 tracked detections."""
        return self.tracked_detections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "raw_detections": [detection.to_dict() for detection in self.raw_detections],
            "tracked_detections": [detection.to_dict() for detection in self.tracked_detections],
        }


@dataclass(frozen=True)
class TrackStatistics:
    """Aggregate statistics for one ByteTrack identity."""

    track_id: int
    start_frame: int
    end_frame: int
    num_observed_frames: int
    global_coverage: float
    span_coverage: float
    mean_confidence: float
    median_confidence: float
    mean_bbox_area_ratio: float
    median_bbox_area_ratio: float
    max_missing_gap: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionSummary:
    """Video-level counts and coverage for raw and associated people."""

    frames_with_raw_person: int
    frames_with_tracked_person: int
    raw_person_frame_coverage: float
    tracked_person_frame_coverage: float
    total_raw_detections: int
    total_tracked_detections: int
    untracked_raw_detections: int
    mean_raw_confidence: float
    median_raw_confidence: float
    min_raw_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    num_frames: int
    duration_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoTrackingResult:
    video: VideoInfo
    frames: List[FrameDetections]
    detector: Dict[str, Any]
    tracker: Dict[str, Any]
    processing: Dict[str, Any]
    tracks: List[TrackStatistics]
    detection_summary: DetectionSummary
    warnings: List[str] = field(default_factory=list)
    schema_version: str = "1.1"

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video": self.video.to_dict(),
            "detector": self.detector,
            "tracker": self.tracker,
            "processing": self.processing,
            "detection_summary": self.detection_summary.to_dict(),
            "tracks": [track.to_dict() for track in self.tracks],
            "warnings": self.warnings,
        }
