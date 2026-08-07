"""Body-structure and joint-motion temporal metrics."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schema import HumanTemporalConfig

BODY_BONES: Tuple[Tuple[str, str], ...] = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
)
BODY_JOINT_NAMES: Tuple[str, ...] = tuple(dict.fromkeys(
    name for bone in BODY_BONES for name in bone
))
EPSILON = 1e-8


def _p90(values: Sequence[float]) -> float:
    """Use a tail-sensitive percentile estimator so one body joint remains visible."""
    return float(np.percentile(values, 90, method="weibull"))


def _pose_arrays(frame: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pose = frame.get("human_pose", {})
    xy = np.asarray(pose.get("keypoints_xy", []), dtype=np.float64)
    scores = np.asarray(pose.get("keypoint_scores", []), dtype=np.float64)
    if xy.ndim != 2 or xy.shape[-1:] != (2,) or scores.ndim != 1:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    if len(xy) != len(scores):
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    return xy, scores


def _valid_joint(
    index: Optional[int], xy: np.ndarray, scores: np.ndarray, threshold: float
) -> bool:
    return bool(
        index is not None
        and 0 <= index < len(scores)
        and scores[index] >= threshold
        and np.isfinite(xy[index]).all()
    )


def _bone_lengths(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    name_to_index: Mapping[str, int],
    threshold: float,
) -> List[Tuple[float, float]]:
    previous_xy, previous_scores = _pose_arrays(previous)
    current_xy, current_scores = _pose_arrays(current)
    lengths: List[Tuple[float, float]] = []
    for first, second in BODY_BONES:
        first_index = name_to_index.get(first)
        second_index = name_to_index.get(second)
        if not all((
            _valid_joint(first_index, previous_xy, previous_scores, threshold),
            _valid_joint(second_index, previous_xy, previous_scores, threshold),
            _valid_joint(first_index, current_xy, current_scores, threshold),
            _valid_joint(second_index, current_xy, current_scores, threshold),
        )):
            continue
        previous_length = float(np.linalg.norm(
            previous_xy[first_index] - previous_xy[second_index]
        ))
        current_length = float(np.linalg.norm(
            current_xy[first_index] - current_xy[second_index]
        ))
        if previous_length > EPSILON and current_length > EPSILON:
            lengths.append((previous_length, current_length))
    return lengths


def structure_pair_metric(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    name_to_index: Mapping[str, int],
    config: HumanTemporalConfig,
) -> Tuple[Optional[float], int]:
    """Compute scale-compensated P90 bone-length change for a frame pair."""
    lengths = _bone_lengths(
        previous, current, name_to_index, config.keypoint_threshold
    )
    if len(lengths) < config.min_valid_bones:
        return None, len(lengths)
    ratios = [current / previous for previous, current in lengths]
    global_scale_ratio = float(np.median(ratios))
    jumps = []
    for previous, current in lengths:
        expected = global_scale_ratio * previous
        jumps.append(abs(current - expected) / (0.5 * (current + expected) + EPSILON))
    return _p90(jumps), len(lengths)


def _normalized_pose(
    frame: Mapping[str, Any],
    name_to_index: Mapping[str, int],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xy, scores = _pose_arrays(frame)
    if not len(xy):
        return xy, scores
    left_hip = name_to_index.get("left_hip")
    right_hip = name_to_index.get("right_hip")
    if _valid_joint(left_hip, xy, scores, threshold) and _valid_joint(
        right_hip, xy, scores, threshold
    ):
        center = 0.5 * (xy[left_hip] + xy[right_hip])
    else:
        x1, y1, x2, y2 = (float(value) for value in frame["bbox_xyxy"])
        center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float64)
    bbox_height = float(frame["bbox_xyxy"][3]) - float(frame["bbox_xyxy"][1])
    if bbox_height <= EPSILON:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    return (xy - center) / bbox_height, scores


def motion_triplet_metric(
    first: Mapping[str, Any],
    middle: Mapping[str, Any],
    last: Mapping[str, Any],
    name_to_index: Mapping[str, int],
    config: HumanTemporalConfig,
) -> Tuple[Optional[float], int]:
    """Compute P90 normalized joint acceleration using real frame intervals."""
    dt1 = int(middle["frame_index"]) - int(first["frame_index"])
    dt2 = int(last["frame_index"]) - int(middle["frame_index"])
    if min(dt1, dt2) <= 0 or max(dt1, dt2) > config.max_frame_gap:
        return None, 0
    poses = [
        _normalized_pose(frame, name_to_index, config.keypoint_threshold)
        for frame in (first, middle, last)
    ]
    accelerations: List[float] = []
    for name in BODY_JOINT_NAMES:
        index = name_to_index.get(name)
        if not all(
            _valid_joint(index, xy, scores, config.keypoint_threshold)
            for xy, scores in poses
        ):
            continue
        first_xy, middle_xy, last_xy = (pose[0][index] for pose in poses)
        first_velocity = (middle_xy - first_xy) / dt1
        second_velocity = (last_xy - middle_xy) / dt2
        acceleration = np.linalg.norm(second_velocity - first_velocity) / (
            (dt1 + dt2) / 2
        )
        accelerations.append(float(acceleration))
    if len(accelerations) < config.min_valid_joints:
        return None, len(accelerations)
    return _p90(accelerations), len(accelerations)


def _summary(values: Sequence[float], prefix: str) -> Dict[str, Optional[float]]:
    if not values:
        return {
            f"{prefix}_mean": None, f"{prefix}_p90": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": float(mean(values)),
        f"{prefix}_p90": _p90(values),
        f"{prefix}_max": float(max(values)),
    }


def _worst(
    frame_metrics: Sequence[Mapping[str, Any]], metric_name: str
) -> List[Dict[str, Any]]:
    ranked = [
        {"frame_index": int(item["frame_index"]), "value": float(item[metric_name])}
        for item in frame_metrics if item[metric_name] is not None
    ]
    ranked.sort(key=lambda item: (-item["value"], item["frame_index"]))
    return ranked[:5]


def analyze_person_temporal(
    person: Mapping[str, Any],
    name_to_index: Mapping[str, int],
    config: HumanTemporalConfig,
) -> Dict[str, Any]:
    """Aggregate pose coverage and temporal metrics for one logical person."""
    frames = sorted(person["frames"], key=lambda item: int(item["frame_index"]))
    frame_metrics = {
        int(frame["frame_index"]): {
            "frame_index": int(frame["frame_index"]),
            "bone_length_jump": None,
            "joint_acceleration": None,
            "valid_bones": 0,
            "valid_joints": 0,
        }
        for frame in frames
    }
    coverages: List[float] = []
    available_indices = [
        name_to_index[name] for name in BODY_JOINT_NAMES if name in name_to_index
    ]
    pose_frames = 0
    for frame in frames:
        xy, scores = _pose_arrays(frame)
        valid = sum(
            _valid_joint(index, xy, scores, config.keypoint_threshold)
            for index in available_indices
        )
        coverage = valid / len(available_indices) if available_indices else 0.0
        coverages.append(coverage)
        pose_frames += int(valid > 0)

    for previous, current in zip(frames, frames[1:]):
        gap = int(current["frame_index"]) - int(previous["frame_index"])
        if gap <= 0 or gap > config.max_frame_gap:
            continue
        value, valid_bones = structure_pair_metric(
            previous, current, name_to_index, config
        )
        target = frame_metrics[int(current["frame_index"])]
        target["bone_length_jump"] = value
        target["valid_bones"] = valid_bones

    for first, middle, last in zip(frames, frames[1:], frames[2:]):
        value, valid_joints = motion_triplet_metric(
            first, middle, last, name_to_index, config
        )
        target = frame_metrics[int(last["frame_index"])]
        target["joint_acceleration"] = value
        target["valid_joints"] = valid_joints

    ordered_metrics = [frame_metrics[int(frame["frame_index"])] for frame in frames]
    structure_values = [
        float(item["bone_length_jump"]) for item in ordered_metrics
        if item["bone_length_jump"] is not None
    ]
    motion_values = [
        float(item["joint_acceleration"]) for item in ordered_metrics
        if item["joint_acceleration"] is not None
    ]
    return {
        "valid": pose_frames > 0,
        "pose_frames": pose_frames,
        "total_observed_frames": len(frames),
        "mean_keypoint_coverage": float(mean(coverages)) if coverages else 0.0,
        "valid_structure_pairs": len(structure_values),
        "valid_motion_triplets": len(motion_values),
        "keypoint_threshold": config.keypoint_threshold,
        "keypoint_name_to_index": dict(name_to_index),
        "metrics": {
            **_summary(structure_values, "bone_length_jump"),
            **_summary(motion_values, "joint_acceleration"),
        },
        "frame_metrics": ordered_metrics,
        "worst_structure_frames": _worst(ordered_metrics, "bone_length_jump"),
        "worst_motion_frames": _worst(ordered_metrics, "joint_acceleration"),
        "score": None,
    }
