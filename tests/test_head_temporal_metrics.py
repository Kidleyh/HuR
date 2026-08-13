import math
from pathlib import Path

import numpy as np

from astrolabe.scorers.video.human_temporal.head_metrics import analyze_head_temporal
from astrolabe.scorers.video.human_temporal.schema import PartTemporalConfig


def _config(tmp_path, threshold=0.3, gap=2):
    cfg, ckpt = tmp_path / "face.py", tmp_path / "face.pth"
    cfg.write_text("config")
    ckpt.write_bytes(b"weights")
    return PartTemporalConfig(cfg, ckpt, threshold, gap, min_valid_keypoints=5)


BASE = np.asarray([
    [-2, -1], [-1, -2], [0, -2.2], [1, -2], [2, -1],
    [-1.4, 0], [-0.5, 0.2], [0.5, 0.2], [1.4, 0],
    [-0.8, 1], [0, 1.3], [0.8, 1],
], dtype=float)


def _record(frame, *, scale=1.0, angle=0.0, shift=(50, 40), change=None, low=()):
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = BASE.copy()
    if change:
        points[change[0]] += change[1]
    points = points @ rotation.T * scale + np.asarray(shift)
    scores = [0.1 if i in low else 0.99 for i in range(len(points))]
    return {
        "frame_index": frame, "bbox_xyxy": [30, 20, 70, 60],
        "face_pose": {"keypoints_xy": points.tolist(), "keypoint_scores": scores},
    }


def test_similarity_transform_has_low_shape_jump(tmp_path):
    result = analyze_head_temporal([
        _record(0), _record(1, scale=1.5, angle=0.2, shift=(80, 20)),
    ], _config(tmp_path))
    assert result["metrics"]["face_shape_jump_max"] < 1e-12


def test_local_face_landmark_jump_is_visible_and_motion_targets_middle(tmp_path):
    result = analyze_head_temporal([
        _record(0), _record(1), _record(2, change=(0, np.asarray([8, 0]))),
        _record(3),
    ], _config(tmp_path))
    assert result["metrics"]["face_shape_jump_max"] > 0.05
    metrics = {item["frame_index"]: item for item in result["frame_metrics"]}
    assert metrics[1]["head_motion_acceleration"] is not None
    assert metrics[2]["head_motion_acceleration"] is not None
    assert metrics[3]["head_motion_acceleration"] is None
    assert {item["frame_index"] for item in result["worst_motion_frames"]} == {1, 2}


def test_low_confidence_and_large_gap_are_ignored(tmp_path):
    result = analyze_head_temporal([
        _record(0, low=tuple(range(10))), _record(1, low=tuple(range(10))),
        _record(10),
    ], _config(tmp_path))
    assert result["valid_shape_pairs"] == 0
    assert result["valid_motion_triplets"] == 0
    assert result["score"] is None
