"""Data models for offline geometric tracklet stitching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional

from astrolabe.scorers.video.person_tracking.schemas import TrackedDetection


@dataclass(frozen=True)
class StitchingConfig:
    max_gap_frames: int = 5
    velocity_window: int = 5
    max_normalized_center_distance: float = 0.15
    max_area_ratio_change: float = 3.0
    max_aspect_ratio_change: float = 2.0
    time_tau: float = 3.0
    motion_sigma: float = 0.08
    merge_threshold: float = 0.75
    uncertain_threshold: float = 0.55
    minimum_assignment_margin: float = 0.08
    raw_bridge_max_center_distance: float = 0.12
    raw_bridge_max_area_ratio_change: float = 2.5
    raw_bridge_max_aspect_ratio_change: float = 2.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "time": 0.15, "motion": 0.35, "predicted_iou": 0.20,
        "scale": 0.10, "raw_bridge": 0.20,
    })

    def __post_init__(self) -> None:
        if self.max_gap_frames < 0 or self.velocity_window < 1:
            raise ValueError("max_gap_frames must be non-negative and velocity_window positive")
        if not 0 <= self.uncertain_threshold <= self.merge_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= uncertain <= merge <= 1")
        if self.time_tau <= 0 or self.motion_sigma <= 0:
            raise ValueError("time_tau and motion_sigma must be positive")
        if self.minimum_assignment_margin < 0:
            raise ValueError("minimum_assignment_margin must be non-negative")
        expected = {"time", "motion", "predicted_iou", "scale", "raw_bridge"}
        if set(self.weights) != expected or any(value < 0 for value in self.weights.values()):
            raise ValueError(f"weights must contain non-negative values for {sorted(expected)}")
        if sum(self.weights.values()) <= 0:
            raise ValueError("at least one score weight must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Tracklet:
    track_id: int
    frame_indices: List[int]
    detections: List[TrackedDetection]
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.track_id < 0 or not self.frame_indices:
            raise ValueError("tracklet must have a non-negative ID and at least one frame")
        if len(self.frame_indices) != len(self.detections):
            raise ValueError("frame_indices and detections lengths differ")
        if self.frame_indices != sorted(self.frame_indices):
            raise ValueError("tracklet observations must be sorted")
        if self.start_frame != self.frame_indices[0] or self.end_frame != self.frame_indices[-1]:
            raise ValueError("tracklet bounds do not match observations")

    @property
    def num_observed_frames(self) -> int:
        return len(self.frame_indices)

    @property
    def start_detection(self) -> TrackedDetection:
        return self.detections[0]

    @property
    def end_detection(self) -> TrackedDetection:
        return self.detections[-1]

    @property
    def median_area_ratio(self) -> float:
        return median(item.bbox_area_ratio for item in self.detections)

    @property
    def median_aspect_ratio(self) -> float:
        return median((item.bbox_xywh[2] / item.bbox_xywh[3]) for item in self.detections)


@dataclass(frozen=True)
class RawBridgeMatch:
    frame_index: int
    raw_detection_index: int
    compatibility: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateEdge:
    from_track_id: int
    to_track_id: int
    gap_frames: int
    score: float = 0.0
    time_score: float = 0.0
    motion_score: float = 0.0
    predicted_iou_score: float = 0.0
    scale_score: float = 0.0
    raw_bridge_score: float = 0.0
    raw_bridge_coverage: float = 0.0
    raw_bridge_compatibility: float = 0.0
    normalized_center_distance: float = 0.0
    area_ratio_change: float = 1.0
    aspect_ratio_change: float = 1.0
    outgoing_margin: Optional[float] = None
    incoming_margin: Optional[float] = None
    decision: str = "rejected"
    rejection_reasons: List[str] = field(default_factory=list)
    raw_bridge_matches: List[RawBridgeMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(frozen=True)
class LogicalTrackStatistics:
    logical_track_id: int
    source_track_ids: List[int]
    start_frame: int
    end_frame: int
    num_fragments: int
    num_observed_frames: int
    global_coverage: float
    span_coverage: float
    max_internal_gap: int
    mean_confidence: float
    median_confidence: float
    mean_bbox_area_ratio: float
    median_bbox_area_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StitchingResult:
    source_tracking_schema_version: str
    config: StitchingConfig
    track_id_to_logical_track_id: Dict[int, int]
    edges: List[CandidateEdge]
    logical_tracks: List[LogicalTrackStatistics]
    warnings: List[str] = field(default_factory=list)
    schema_version: str = "1.0"
    runtime_sec: float = 0.0

    @property
    def merged_edges(self) -> List[CandidateEdge]:
        return [edge for edge in self.edges if edge.decision == "merged"]

    @property
    def uncertain_edges(self) -> List[CandidateEdge]:
        return [edge for edge in self.edges if edge.decision == "uncertain"]

    @property
    def rejected_edges(self) -> List[CandidateEdge]:
        return [edge for edge in self.edges if edge.decision == "rejected"]
