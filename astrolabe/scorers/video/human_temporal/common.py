"""Shared helpers for RTMPose temporal engines and metrics."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EPSILON = 1e-8


def p90(values: Sequence[float]) -> float:
    return float(np.percentile(values, 90, method="weibull"))


def pose_arrays(record: Mapping[str, Any], key: str) -> Tuple[np.ndarray, np.ndarray]:
    pose = record.get(key, {})
    xy = np.asarray(pose.get("keypoints_xy", []), dtype=np.float64)
    scores = np.asarray(pose.get("keypoint_scores", []), dtype=np.float64)
    if (
        xy.ndim != 2 or xy.shape[-1:] != (2,) or scores.ndim != 1
        or len(xy) != len(scores)
    ):
        return np.empty((0, 2)), np.empty(0)
    return xy, scores


def valid_point(index: int, xy: np.ndarray, scores: np.ndarray, threshold: float) -> bool:
    return bool(
        0 <= index < len(scores) and scores[index] >= threshold
        and np.isfinite(xy[index]).all()
    )


def summary(values: Sequence[float], prefix: str) -> Dict[str, Optional[float]]:
    if not values:
        return {
            f"{prefix}_mean": None, f"{prefix}_p90": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": float(mean(values)),
        f"{prefix}_p90": p90(values),
        f"{prefix}_max": float(max(values)),
    }


def worst(
    metrics: Sequence[Mapping[str, Any]], name: str, *, limit: int = 5
) -> List[Dict[str, Any]]:
    ranked = [
        {"frame_index": int(item["frame_index"]), "value": float(item[name])}
        for item in metrics if item.get(name) is not None
    ]
    ranked.sort(key=lambda item: (-item["value"], item["frame_index"]))
    return ranked[:limit]
