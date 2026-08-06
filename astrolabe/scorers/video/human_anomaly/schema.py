"""Schemas and official thresholds for per-logical-track anomaly analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Dict, List, Optional, Sequence

OFFICIAL_THRESHOLDS: Dict[str, float] = {
    "human": 0.4545454545454546,
    "face": 0.30303030303030304,
    "hand": 0.3232,
}


@dataclass(frozen=True)
class HumanAnomalyInput:
    frame_index: int
    logical_track_id: int
    source_track_id: int
    bbox_xyxy: List[float]
    detection_confidence: float

    def __post_init__(self) -> None:
        if min(self.frame_index, self.logical_track_id, self.source_track_id) < 0:
            raise ValueError("frame and track IDs must be non-negative")
        if len(self.bbox_xyxy) != 4 or not all(isfinite(value) for value in self.bbox_xyxy):
            raise ValueError("bbox_xyxy must contain four finite values")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must satisfy 0 <= x1 < x2 and 0 <= y1 < y2")
        if not isfinite(self.detection_confidence) or not 0 <= self.detection_confidence <= 1:
            raise ValueError("detection_confidence must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestFailure:
    frame_index: Optional[int]
    logical_track_id: Optional[int]
    source_track_id: Optional[int]
    failure_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classifier_result(scores: Sequence[float], category: str) -> Dict[str, Any]:
    """Apply VBench's class-0 abnormal-probability threshold semantics."""
    if category not in OFFICIAL_THRESHOLDS:
        raise ValueError(f"Unknown anomaly category: {category}")
    if len(scores) < 2 or not all(isfinite(float(value)) for value in scores):
        raise ValueError("classifier scores must contain at least two finite values")
    values = [float(value) for value in scores]
    probability = values[0]
    return {
        "scored": True,
        "scores": values,
        "abnormal_probability": probability,
        "abnormal": probability > OFFICIAL_THRESHOLDS[category],
    }


def person_is_abnormal(
    human: Dict[str, Any], faces: Sequence[Dict[str, Any]], hands: Sequence[Dict[str, Any]]
) -> bool:
    """Missing face/hand detections are neutral; any scored anomaly is positive."""
    return bool(
        human.get("abnormal", False)
        or any(item.get("abnormal", False) for item in faces)
        or any(item.get("abnormal", False) for item in hands)
    )
