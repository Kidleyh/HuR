"""Schemas for the optional GVHMR Human Temporal V2 stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


METRIC_NAMES = (
    "joint_velocity", "joint_acceleration", "joint_jerk",
    "root_velocity", "root_acceleration",
)


@dataclass(frozen=True)
class GVHMRTemporalConfig:
    """Explicit local GVHMR resources and light metric controls."""

    gvhmr_root: Path
    checkpoint: Path
    min_valid_joints: int = 1

    def __post_init__(self) -> None:
        root = Path(self.gvhmr_root).expanduser().resolve()
        checkpoint = Path(self.checkpoint).expanduser().resolve()
        object.__setattr__(self, "gvhmr_root", root)
        object.__setattr__(self, "checkpoint", checkpoint)
        if not root.is_dir():
            raise NotADirectoryError(f"GVHMR root does not exist: {root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"GVHMR checkpoint does not exist: {checkpoint}")
        if self.min_valid_joints < 1:
            raise ValueError("min_valid_joints must be positive")


@dataclass(frozen=True)
class GVHMRSequence:
    """One logical person's GVHMR output aligned to observed HuR frames."""

    frame_indices: np.ndarray
    joints_3d: np.ndarray
    root_translation: np.ndarray
    smpl_params: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = np.asarray(self.frame_indices, dtype=np.int64)
        joints = np.asarray(self.joints_3d, dtype=np.float64)
        roots = np.asarray(self.root_translation, dtype=np.float64)
        if frames.ndim != 1 or len(frames) == 0:
            raise ValueError("GVHMR frame_indices must be a non-empty 1D array")
        if joints.ndim != 3 or joints.shape[0] != len(frames) or joints.shape[2] != 3:
            raise ValueError("GVHMR joints_3d must have shape (T, J, 3)")
        if roots.shape != (len(frames), 3):
            raise ValueError("GVHMR root_translation must have shape (T, 3)")
        if np.any(np.diff(frames) <= 0):
            raise ValueError("GVHMR frame_indices must be strictly increasing")
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "joints_3d", joints)
        object.__setattr__(self, "root_translation", roots)


def empty_metrics() -> Dict[str, Any]:
    return {
        f"{name}_{stat}": None
        for name in METRIC_NAMES
        for stat in ("mean", "p90", "max")
    }


def failed_human_3d_result(
    total_observed_frames: int, error: BaseException
) -> Dict[str, Any]:
    """Represent an auxiliary GVHMR failure without changing reward validity."""
    return {
        "valid": False,
        "error_type": type(error).__name__,
        "message": str(error),
        "pose_frames": 0,
        "total_observed_frames": int(total_observed_frames),
        "joint_count": 0,
        "smpl_parameter_shapes": {},
        "metrics": empty_metrics(),
        "frame_metrics": [],
        "worst_acceleration_frames": [],
        "worst_jerk_frames": [],
        "score": None,
    }
