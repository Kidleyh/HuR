"""Similarity-invariant face shape and head motion temporal metrics."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .common import EPSILON, p90, pose_arrays, summary, valid_point, worst
from .schema import PartTemporalConfig


def _matched_landmarks(
    previous: Mapping[str, Any], current: Mapping[str, Any], config: PartTemporalConfig
) -> Tuple[np.ndarray, np.ndarray]:
    first_xy, first_scores = pose_arrays(previous, "face_pose")
    second_xy, second_scores = pose_arrays(current, "face_pose")
    count = min(len(first_xy), len(second_xy))
    valid = [
        index for index in range(count)
        if valid_point(index, first_xy, first_scores, config.keypoint_threshold)
        and valid_point(index, second_xy, second_scores, config.keypoint_threshold)
    ]
    if len(valid) < config.min_valid_keypoints:
        return np.empty((0, 2)), np.empty((0, 2))
    return first_xy[valid], second_xy[valid]


def face_shape_pair_metric(
    previous: Mapping[str, Any], current: Mapping[str, Any], config: PartTemporalConfig
) -> Tuple[Optional[float], int]:
    """Align translation, scale and 2D rotation before measuring residuals."""
    first, second = _matched_landmarks(previous, current, config)
    if not len(first):
        return None, 0
    first_centered = first - first.mean(axis=0)
    second_centered = second - second.mean(axis=0)
    first_norm = float(np.linalg.norm(first_centered))
    second_norm = float(np.linalg.norm(second_centered))
    if min(first_norm, second_norm) <= EPSILON:
        return None, len(first)
    first_unit = first_centered / first_norm
    second_unit = second_centered / second_norm
    u, _, vt = np.linalg.svd(first_unit.T @ second_unit)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = first_unit @ rotation
    residuals = np.linalg.norm(aligned - second_unit, axis=1) * math.sqrt(len(first))
    return p90(residuals.tolist()), len(first)


def _head_state(record: Mapping[str, Any], config: PartTemporalConfig) -> Optional[np.ndarray]:
    xy, scores = pose_arrays(record, "face_pose")
    valid = np.asarray([
        valid_point(i, xy, scores, config.keypoint_threshold) for i in range(len(xy))
    ])
    points = xy[valid]
    if len(points) < config.min_valid_keypoints:
        return None
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    scale = math.sqrt(max(float(values.sum()), EPSILON))
    axis = vectors[:, int(np.argmax(values))]
    orientation = math.atan2(float(axis[1]), float(axis[0]))
    x1, y1, x2, y2 = map(float, record["bbox_xyxy"])
    bbox_scale = max(math.hypot(x2 - x1, y2 - y1), EPSILON)
    return np.asarray([
        center[0] / bbox_scale, center[1] / bbox_scale,
        math.log(scale + EPSILON), orientation,
    ])


def _angle_difference(current: float, previous: float) -> float:
    # PCA axes are unoriented: theta and theta + pi describe the same axis.
    delta = current - previous
    return 0.5 * math.atan2(math.sin(2 * delta), math.cos(2 * delta))


def head_motion_triplet_metric(
    first: Mapping[str, Any], middle: Mapping[str, Any], last: Mapping[str, Any],
    config: PartTemporalConfig,
) -> Optional[float]:
    dt1 = int(middle["frame_index"]) - int(first["frame_index"])
    dt2 = int(last["frame_index"]) - int(middle["frame_index"])
    if min(dt1, dt2) <= 0 or max(dt1, dt2) > config.max_frame_gap:
        return None
    states = [_head_state(item, config) for item in (first, middle, last)]
    if any(state is None for state in states):
        return None
    first_state, middle_state, last_state = states
    v1 = (middle_state - first_state) / dt1
    v2 = (last_state - middle_state) / dt2
    v1[3] = _angle_difference(middle_state[3], first_state[3]) / dt1
    v2[3] = _angle_difference(last_state[3], middle_state[3]) / dt2
    return float(np.linalg.norm(v2 - v1) / ((dt1 + dt2) / 2))


def analyze_head_temporal(
    observations: Sequence[Mapping[str, Any]], config: PartTemporalConfig
) -> Dict[str, Any]:
    records = sorted(observations, key=lambda item: int(item["frame_index"]))
    metrics = {
        int(item["frame_index"]): {
            "frame_index": int(item["frame_index"]), "face_shape_jump": None,
            "head_motion_acceleration": None, "valid_landmarks": 0,
        } for item in records
    }
    for previous, current in zip(records, records[1:]):
        gap = int(current["frame_index"]) - int(previous["frame_index"])
        if gap <= 0 or gap > config.max_frame_gap:
            continue
        value, count = face_shape_pair_metric(previous, current, config)
        metrics[int(current["frame_index"])]["face_shape_jump"] = value
        metrics[int(current["frame_index"])]["valid_landmarks"] = count
    for first, middle, last in zip(records, records[1:], records[2:]):
        metrics[int(middle["frame_index"])]["head_motion_acceleration"] = (
            head_motion_triplet_metric(first, middle, last, config)
        )
    ordered = [metrics[int(item["frame_index"])] for item in records]
    shape = [float(item["face_shape_jump"]) for item in ordered if item["face_shape_jump"] is not None]
    motion = [float(item["head_motion_acceleration"]) for item in ordered if item["head_motion_acceleration"] is not None]
    coverages = []
    for item in records:
        xy, scores = pose_arrays(item, "face_pose")
        coverages.append(sum(score >= config.keypoint_threshold for score in scores) / len(xy) if len(xy) else 0.0)
    return {
        "valid": bool(shape or motion), "pose_frames": sum(value > 0 for value in coverages),
        "total_observed_frames": len(records),
        "mean_keypoint_coverage": float(mean(coverages)) if coverages else 0.0,
        "valid_shape_pairs": len(shape), "valid_motion_triplets": len(motion),
        "keypoint_threshold": config.keypoint_threshold,
        "metrics": {**summary(shape, "face_shape_jump"), **summary(motion, "head_motion_acceleration")},
        "frame_metrics": ordered,
        "worst_shape_frames": worst(ordered, "face_shape_jump"),
        "worst_motion_frames": worst(ordered, "head_motion_acceleration"),
        "score": None,
    }
