"""Bounding-box geometry and motion prediction features."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .schemas import Tracklet

BBox = List[float]


def bbox_center_size(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def ratio_change(first: float, second: float) -> float:
    if first <= 0 or second <= 0:
        return math.inf
    return max(first, second) / min(first, second)


def normalized_center_distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Distance for boxes already represented in normalized image coordinates."""
    ax, ay, _, _ = bbox_center_size(first)
    bx, by, _, _ = bbox_center_size(second)
    return math.hypot(ax - bx, ay - by) / math.sqrt(2.0)


def scale_score(first: Sequence[float], second: Sequence[float], epsilon: float = 1e-9) -> float:
    _, _, aw, ah = bbox_center_size(first)
    _, _, bw, bh = bbox_center_size(second)
    if min(aw, ah, bw, bh) <= 0:
        return 0.0
    area = math.exp(-abs(math.log((bw * bh + epsilon) / (aw * ah + epsilon))))
    aspect = math.exp(-abs(math.log((bw / bh + epsilon) / (aw / ah + epsilon))))
    return float((area + aspect) / 2.0)


def interpolate_bbox(first: Sequence[float], second: Sequence[float], alpha: float) -> BBox:
    return [float(a + alpha * (b - a)) for a, b in zip(first, second)]


def predict_tracklet_bbox(
    tracklet: Tracklet, target_frame: int, velocity_window: int = 5
) -> Optional[BBox]:
    """Regress normalized center/log-size over time and extrapolate to target_frame."""
    count = min(max(velocity_window, 1), tracklet.num_observed_frames)
    frames = np.asarray(tracklet.frame_indices[-count:], dtype=np.float64)
    values = []
    for detection in tracklet.detections[-count:]:
        cx, cy, width, height = bbox_center_size(detection.bbox_xyxy_normalized)
        if width <= 0 or height <= 0:
            return None
        values.append([cx, cy, math.log(width), math.log(height)])
    array = np.asarray(values, dtype=np.float64)
    if count == 1:
        predicted = array[0]
    else:
        centered = frames - frames.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= 0:
            predicted = array[-1]
        else:
            slopes = centered @ (array - array.mean(axis=0)) / denominator
            predicted = array.mean(axis=0) + slopes * (target_frame - frames.mean())
    if not np.all(np.isfinite(predicted)):
        return None
    cx, cy = float(predicted[0]), float(predicted[1])
    width, height = math.exp(float(predicted[2])), math.exp(float(predicted[3]))
    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in (cx, cy, width, height)):
        return None
    box = [max(0.0, cx - width / 2), max(0.0, cy - height / 2),
           min(1.0, cx + width / 2), min(1.0, cy + height / 2)]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box
