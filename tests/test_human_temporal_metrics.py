from pathlib import Path

import numpy as np

from astrolabe.scorers.video.human_temporal.metrics import (
    BODY_JOINT_NAMES,
    analyze_person_temporal,
)
from astrolabe.scorers.video.human_temporal.schema import HumanTemporalConfig

NAME_TO_INDEX = {name: index for index, name in enumerate(BODY_JOINT_NAMES)}
BASE = {
    "left_shoulder": (-1, 4), "right_shoulder": (1, 4),
    "left_elbow": (-2, 3), "right_elbow": (2, 3),
    "left_wrist": (-3, 2), "right_wrist": (3, 2),
    "left_hip": (-1, 0), "right_hip": (1, 0),
    "left_knee": (-1, -2), "right_knee": (1, -2),
    "left_ankle": (-1, -4), "right_ankle": (1, -4),
}


def _config(tmp_path: Path, threshold=0.3, max_gap=2):
    pose_config = tmp_path / "pose.py"
    checkpoint = tmp_path / "pose.pth"
    pose_config.write_text("config")
    checkpoint.write_bytes(b"checkpoint")
    return HumanTemporalConfig(
        pose_config, checkpoint, keypoint_threshold=threshold,
        max_frame_gap=max_gap,
    )


def _frame(frame_index, *, scale=1.0, shift=(0, 0), changes=None, low=()):
    changes = changes or {}
    points = []
    scores = []
    for name in BODY_JOINT_NAMES:
        x, y = changes.get(name, BASE[name])
        points.append([
            50 + scale * x + shift[0], 50 + scale * y + shift[1]
        ])
        scores.append(0.1 if name in low else 0.99)
    half_height = 10 * scale
    return {
        "frame_index": frame_index,
        "bbox_xyxy": [
            40 * scale + shift[0], 50 - half_height + shift[1],
            60 * scale + shift[0], 50 + half_height + shift[1],
        ],
        "human_pose": {"keypoints_xy": points, "keypoint_scores": scores},
    }


def _analyze(tmp_path, frames):
    return analyze_person_temporal(
        {"frames": frames}, NAME_TO_INDEX, _config(tmp_path)
    )


def test_static_pose_has_zero_structure_and_motion(tmp_path):
    result = _analyze(tmp_path, [_frame(0), _frame(1), _frame(2)])
    assert result["metrics"]["bone_length_jump_max"] == 0.0
    assert result["metrics"]["joint_acceleration_max"] == 0.0


def test_global_translation_and_scale_are_compensated(tmp_path):
    translated = _analyze(tmp_path, [
        _frame(0), _frame(1, shift=(5, 7)), _frame(2, shift=(10, 14)),
    ])
    scaled = _analyze(tmp_path, [
        _frame(0, scale=1.0), _frame(1, scale=1.5), _frame(2, scale=2.0),
    ])
    assert translated["metrics"]["bone_length_jump_max"] < 1e-12
    assert translated["metrics"]["joint_acceleration_max"] < 1e-12
    assert scaled["metrics"]["bone_length_jump_max"] < 1e-12


def test_single_arm_length_jump_and_joint_flicker_are_visible(tmp_path):
    changed = dict(BASE)
    changed["left_wrist"] = (-15, 2)
    result = _analyze(tmp_path, [
        _frame(0), _frame(1), _frame(2, changes=changed), _frame(3),
    ])
    assert result["metrics"]["bone_length_jump_max"] > 0.05
    assert result["metrics"]["joint_acceleration_max"] > 0.05
    assert result["worst_structure_frames"][0]["frame_index"] == 2
    assert {item["frame_index"] for item in result["worst_motion_frames"]} & {2, 3}


def test_low_confidence_joint_is_excluded_not_marked_abnormal(tmp_path):
    changed = dict(BASE)
    changed["left_wrist"] = (-15, 2)
    result = _analyze(tmp_path, [
        _frame(0, low=("left_wrist",)),
        _frame(1, low=("left_wrist",)),
        _frame(2, changes=changed, low=("left_wrist",)),
    ])
    assert result["score"] is None
    assert result["metrics"]["bone_length_jump_max"] == 0.0
    assert result["metrics"]["joint_acceleration_max"] == 0.0
    assert result["mean_keypoint_coverage"] < 1.0


def test_large_track_gap_breaks_structure_and_motion_sequence(tmp_path):
    result = _analyze(tmp_path, [_frame(1), _frame(2), _frame(10)])
    assert result["valid_structure_pairs"] == 1
    assert result["valid_motion_triplets"] == 0
    metrics = {item["frame_index"]: item for item in result["frame_metrics"]}
    assert metrics[10]["bone_length_jump"] is None
    assert metrics[10]["joint_acceleration"] is None
