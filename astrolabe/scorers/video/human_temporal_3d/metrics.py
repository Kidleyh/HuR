"""Frame-gap-aware kinematic summaries for GVHMR 3D sequences."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .schema import GVHMRSequence, empty_metrics


def _p90_norm(vectors: np.ndarray, min_count: int) -> Optional[float]:
    finite = np.all(np.isfinite(vectors), axis=-1)
    values = np.linalg.norm(vectors[finite], axis=-1)
    if len(values) < min_count:
        return None
    return float(np.percentile(values, 90))


def _summary(
    values: Iterable[Optional[float]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    valid = np.asarray([value for value in values if value is not None])
    if valid.size == 0:
        return None, None, None
    return float(valid.mean()), float(np.percentile(valid, 90)), float(valid.max())


def analyze_3d_temporal(
    sequence: GVHMRSequence,
    *,
    fps: float,
    total_observed_frames: int,
    min_valid_joints: int = 1,
) -> Dict[str, Any]:
    """Compute root-relative joint and global-root derivatives per second."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    frames = sequence.frame_indices
    roots = sequence.root_translation
    joints = sequence.joints_3d - roots[:, None, :]
    count = len(frames)
    rows: Dict[int, Dict[str, Any]] = {
        int(frame): {
            "frame_index": int(frame),
            "joint_velocity": None,
            "joint_acceleration": None,
            "joint_jerk": None,
            "root_velocity": None,
            "root_acceleration": None,
            "valid_joints": 0,
        }
        for frame in frames
    }
    joint_velocities: List[Optional[np.ndarray]] = [None] * count
    root_velocities: List[Optional[np.ndarray]] = [None] * count
    velocity_times: List[Optional[float]] = [None] * count
    for index in range(1, count):
        dt = float(frames[index] - frames[index - 1]) / fps
        joint_velocity = (joints[index] - joints[index - 1]) / dt
        root_velocity = (roots[index] - roots[index - 1]) / dt
        row = rows[int(frames[index])]
        row["joint_velocity"] = _p90_norm(joint_velocity, min_valid_joints)
        row["root_velocity"] = (
            float(np.linalg.norm(root_velocity))
            if np.all(np.isfinite(root_velocity)) else None
        )
        row["valid_joints"] = int(
            np.all(np.isfinite(joint_velocity), axis=-1).sum()
        )
        joint_velocities[index] = joint_velocity
        root_velocities[index] = root_velocity
        velocity_times[index] = 0.5 * float(frames[index] + frames[index - 1]) / fps

    joint_accelerations: List[Optional[np.ndarray]] = [None] * count
    acceleration_times: List[Optional[float]] = [None] * count
    for index in range(2, count):
        dt = velocity_times[index] - velocity_times[index - 1]  # type: ignore[operator]
        joint_acceleration = (
            joint_velocities[index] - joint_velocities[index - 1]  # type: ignore[operator]
        ) / dt
        root_acceleration = (
            root_velocities[index] - root_velocities[index - 1]  # type: ignore[operator]
        ) / dt
        target = index - 1
        row = rows[int(frames[target])]
        row["joint_acceleration"] = _p90_norm(
            joint_acceleration, min_valid_joints
        )
        row["root_acceleration"] = (
            float(np.linalg.norm(root_acceleration))
            if np.all(np.isfinite(root_acceleration)) else None
        )
        row["valid_joints"] = max(
            row["valid_joints"],
            int(np.all(np.isfinite(joint_acceleration), axis=-1).sum()),
        )
        joint_accelerations[target] = joint_acceleration
        acceleration_times[target] = float(frames[target]) / fps

    for target in range(2, count - 1):
        previous = target - 1
        if joint_accelerations[previous] is None or joint_accelerations[target] is None:
            continue
        dt = acceleration_times[target] - acceleration_times[previous]  # type: ignore[operator]
        jerk = (
            joint_accelerations[target] - joint_accelerations[previous]  # type: ignore[operator]
        ) / dt
        rows[int(frames[target])]["joint_jerk"] = _p90_norm(
            jerk, min_valid_joints
        )

    frame_metrics = [rows[int(frame)] for frame in frames]
    metrics = empty_metrics()
    for name in (
        "joint_velocity", "joint_acceleration", "joint_jerk",
        "root_velocity", "root_acceleration",
    ):
        mean, p90, maximum = _summary(row[name] for row in frame_metrics)
        metrics[f"{name}_mean"] = mean
        metrics[f"{name}_p90"] = p90
        metrics[f"{name}_max"] = maximum

    def worst(name: str) -> List[Dict[str, Any]]:
        ranked = [
            {"frame_index": row["frame_index"], name: row[name]}
            for row in frame_metrics if row[name] is not None
        ]
        ranked.sort(key=lambda item: (-item[name], item["frame_index"]))
        return ranked[:5]

    return {
        "valid": True,
        "pose_frames": count,
        "total_observed_frames": int(total_observed_frames),
        "joint_count": int(sequence.joints_3d.shape[1]),
        "smpl_parameter_shapes": {
            key: list(np.asarray(value).shape)
            for key, value in sequence.smpl_params.items()
        },
        "metrics": metrics,
        "frame_metrics": frame_metrics,
        "worst_acceleration_frames": worst("joint_acceleration"),
        "worst_jerk_frames": worst("joint_jerk"),
        "score": None,
    }
