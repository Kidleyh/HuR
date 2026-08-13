"""Configuration and stable result helpers for Human Temporal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class HumanTemporalConfig:
    """Local RTMPose resources and temporal metric validity controls."""

    pose_config: Path
    pose_checkpoint: Path
    keypoint_threshold: float = 0.3
    max_frame_gap: int = 2
    min_valid_bones: int = 4
    min_valid_joints: int = 6

    def __post_init__(self) -> None:
        for name in ("pose_config", "pose_checkpoint"):
            path = Path(getattr(self, name)).expanduser().resolve()
            object.__setattr__(self, name, path)
            if not path.is_file():
                raise FileNotFoundError(f"Human Temporal {name} does not exist: {path}")
        if not 0.0 <= self.keypoint_threshold <= 1.0:
            raise ValueError("keypoint_threshold must be in [0, 1]")
        if self.max_frame_gap < 1:
            raise ValueError("max_frame_gap must be positive")
        if self.min_valid_bones < 1 or self.min_valid_joints < 1:
            raise ValueError("minimum valid bone/joint counts must be positive")


def failed_human_temporal_result(
    total_observed_frames: int, error: BaseException
) -> Dict[str, Any]:
    """Represent an auxiliary temporal failure without inventing an anomaly score."""
    return {
        "valid": False,
        "error_type": type(error).__name__,
        "message": str(error),
        "pose_frames": 0,
        "total_observed_frames": int(total_observed_frames),
        "mean_keypoint_coverage": 0.0,
        "valid_structure_pairs": 0,
        "valid_motion_triplets": 0,
        "metrics": {
            "bone_length_jump_mean": None,
            "bone_length_jump_p90": None,
            "bone_length_jump_max": None,
            "joint_acceleration_mean": None,
            "joint_acceleration_p90": None,
            "joint_acceleration_max": None,
        },
        "frame_metrics": [],
        "worst_structure_frames": [],
        "worst_motion_frames": [],
        "score": None,
    }


@dataclass(frozen=True)
class PartTemporalConfig:
    """Local resources and validity controls for face landmark temporal analysis."""

    pose_config: Path
    pose_checkpoint: Path
    keypoint_threshold: float = 0.3
    max_frame_gap: int = 2
    min_valid_keypoints: int = 5

    def __post_init__(self) -> None:
        for name in ("pose_config", "pose_checkpoint"):
            path = Path(getattr(self, name)).expanduser().resolve()
            object.__setattr__(self, name, path)
            if not path.is_file():
                raise FileNotFoundError(f"Temporal {name} does not exist: {path}")
        if not 0.0 <= self.keypoint_threshold <= 1.0:
            raise ValueError("keypoint_threshold must be in [0, 1]")
        if self.max_frame_gap < 1 or self.min_valid_keypoints < 3:
            raise ValueError("invalid temporal gap or keypoint count")


@dataclass(frozen=True)
class HandTemporalConfig(PartTemporalConfig):
    """Hand pose resources, wrist association and metric validity controls."""

    wrist_threshold: float = 0.3
    max_wrist_distance: float = 1.5
    minimum_wrist_margin: float = 0.2
    min_valid_bones: int = 6
    min_valid_joints: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.wrist_threshold <= 1.0:
            raise ValueError("wrist_threshold must be in [0, 1]")
        if self.max_wrist_distance <= 0 or self.minimum_wrist_margin < 0:
            raise ValueError("invalid wrist association distance or margin")
        if self.min_valid_bones < 1 or self.min_valid_joints < 1:
            raise ValueError("minimum valid bone/joint counts must be positive")


def failed_part_temporal_result(error: BaseException) -> Dict[str, Any]:
    return {
        "valid": False, "error_type": type(error).__name__, "message": str(error),
        "metrics": {}, "frame_metrics": [], "score": None,
    }
