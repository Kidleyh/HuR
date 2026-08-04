"""Validated, JSON-friendly schemas for person tracking outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Dict, List, Sequence


def _validate_finite_sequence(name: str, values: Sequence[float], length: int) -> None:
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if not all(isfinite(float(value)) for value in values):
        raise ValueError(f"{name} values must be finite")


@dataclass(frozen=True)
class Detection:
    """One tracked person observation in original-image coordinates."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    bbox_xyxy_normalized: List[float]
    bbox_area_ratio: float

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError("class_id must be a non-negative integer")
        if not self.class_name:
            raise ValueError("class_name must not be empty")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        _validate_finite_sequence("bbox_xyxy", self.bbox_xyxy, 4)
        _validate_finite_sequence("bbox_xywh", self.bbox_xywh, 4)
        _validate_finite_sequence("bbox_xyxy_normalized", self.bbox_xyxy_normalized, 4)
        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must satisfy 0 <= x1 < x2 and 0 <= y1 < y2")
        x, y, width, height = self.bbox_xywh
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("bbox_xywh must use a non-negative top-left and positive size")
        nx1, ny1, nx2, ny2 = self.bbox_xyxy_normalized
        if not (0.0 <= nx1 < nx2 <= 1.0 and 0.0 <= ny1 < ny2 <= 1.0):
            raise ValueError("normalized bbox must be ordered and within [0, 1]")
        if not isfinite(self.bbox_area_ratio) or not 0.0 < self.bbox_area_ratio <= 1.0:
            raise ValueError("bbox_area_ratio must be in (0, 1]")

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
    ) -> "Detection":
        """Clip an xyxy box to the image and derive all exported representations."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        _validate_finite_sequence("bbox_xyxy", bbox_xyxy, 4)
        x1 = min(max(float(bbox_xyxy[0]), 0.0), float(image_width))
        y1 = min(max(float(bbox_xyxy[1]), 0.0), float(image_height))
        x2 = min(max(float(bbox_xyxy[2]), 0.0), float(image_width))
        y2 = min(max(float(bbox_xyxy[3]), 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox is empty after clipping to the original image")
        width = x2 - x1
        height = y2 - y1
        return cls(
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            confidence=float(confidence),
            bbox_xyxy=[x1, y1, x2, y2],
            bbox_xywh=[x1, y1, width, height],
            bbox_xyxy_normalized=[
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height,
            ],
            bbox_area_ratio=(width * height) / (image_width * image_height),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameDetections:
    """All active tracked-person observations for one decoded frame."""

    frame_index: int
    timestamp_sec: float
    detections: List[Detection] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not isfinite(self.timestamp_sec) or self.timestamp_sec < 0:
            raise ValueError("timestamp_sec must be finite and non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "detections": [detection.to_dict() for detection in self.detections],
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
class VideoInfo:
    """Metadata read from the input video and confirmed while decoding it."""

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
    """Complete in-memory representation of one processed video."""

    video: VideoInfo
    frames: List[FrameDetections]
    detector: Dict[str, Any]
    tracker: Dict[str, Any]
    processing: Dict[str, Any]
    tracks: List[TrackStatistics]
    warnings: List[str] = field(default_factory=list)
    schema_version: str = "1.0"

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video": self.video.to_dict(),
            "detector": self.detector,
            "tracker": self.tracker,
            "processing": self.processing,
            "tracks": [track.to_dict() for track in self.tracks],
            "warnings": self.warnings,
        }
