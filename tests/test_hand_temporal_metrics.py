from pathlib import Path

import numpy as np

from astrolabe.scorers.video.human_temporal.hand_metrics import (
    analyze_hand_side, associate_hands_to_wrists,
)
from astrolabe.scorers.video.human_temporal.schema import HandTemporalConfig


def _config(tmp_path, **kwargs):
    cfg, ckpt = tmp_path / "hand.py", tmp_path / "hand.pth"
    cfg.write_text("config")
    ckpt.write_bytes(b"weights")
    return HandTemporalConfig(cfg, ckpt, min_valid_keypoints=5, **kwargs)


BASE = np.asarray([[0, 0]] + [
    [finger * 2 + step * 0.2, step] for finger in range(5) for step in range(1, 5)
], dtype=float)


def _hand(frame, *, scale=1.0, shift=(50, 50), changes=None, low=()):
    points = BASE.copy()
    for index, delta in (changes or {}).items():
        points[index] += delta
    points = points * scale + np.asarray(shift)
    return {
        "frame_index": frame, "bbox_xyxy": [40, 40, 70, 80],
        "hand_pose": {
            "keypoints_xy": points.tolist(),
            "keypoint_scores": [0.1 if i in low else 0.99 for i in range(21)],
        },
    }


def _person_frame(left=(20, 50), right=(80, 50), hands=()):
    return {
        "human_pose": {"keypoints_xy": [left, right], "keypoint_scores": [0.99, 0.99]},
        "hands": [{"bbox_xyxy": box} for box in hands],
    }


def test_wrist_association_is_one_to_one_and_unreliable_is_not_forced(tmp_path):
    config = _config(tmp_path, max_wrist_distance=1.5, minimum_wrist_margin=0.2)
    mapping = {"left_wrist": 0, "right_wrist": 1}
    frame = _person_frame(hands=([15, 45, 25, 55], [75, 45, 85, 55]))
    assert associate_hands_to_wrists(frame, mapping, config) == {"left": 0, "right": 1}
    ambiguous = _person_frame(left=(45, 50), right=(55, 50), hands=([45, 45, 55, 55],))
    assert associate_hands_to_wrists(ambiguous, mapping, config) == {"left": None, "right": None}


def test_scale_compensation_and_finger_jump(tmp_path):
    config = _config(tmp_path)
    scaled = analyze_hand_side([_hand(0), _hand(1, scale=2.0)], config)
    assert scaled["metrics"]["bone_length_jump_max"] < 1e-12
    changed = analyze_hand_side([
        _hand(0), _hand(1, changes={
            2: np.asarray([2, 0]), 3: np.asarray([6, 0]), 4: np.asarray([12, 0]),
        }),
    ], config)
    assert changed["metrics"]["bone_length_jump_max"] > 0.05


def test_joint_flicker_targets_middle_and_gap_breaks_sequence(tmp_path):
    config = _config(tmp_path)
    result = analyze_hand_side([
        _hand(0), _hand(1), _hand(2, changes={
            2: np.asarray([2, 0]), 3: np.asarray([6, 0]), 4: np.asarray([12, 0]),
        }), _hand(3),
    ], config)
    assert result["metrics"]["joint_acceleration_max"] > 0.05
    metrics = {item["frame_index"]: item for item in result["frame_metrics"]}
    assert metrics[1]["hand_joint_acceleration"] is not None
    assert metrics[2]["hand_joint_acceleration"] is not None
    assert metrics[3]["hand_joint_acceleration"] is None
    gap = analyze_hand_side([_hand(0), _hand(1), _hand(10)], config)
    assert gap["valid_motion_triplets"] == 0
