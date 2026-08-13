"""Left/right hand association and scale-invariant temporal metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .common import EPSILON, p90, pose_arrays, summary, valid_point
from .schema import HandTemporalConfig

HAND_BONES: Tuple[Tuple[int, int], ...] = tuple(
    (start, end)
    for finger in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                   (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    for start, end in zip(finger, finger[1:])
)


def associate_hands_to_wrists(
    frame: Mapping[str, Any], name_to_index: Mapping[str, int],
    config: HandTemporalConfig,
) -> Dict[str, Optional[int]]:
    """Reliably assign at most one bbox to each confident body wrist."""
    body_xy, body_scores = pose_arrays(frame, "human_pose")
    hands = frame.get("hands", [])
    candidates = []
    by_hand: Dict[int, List[Tuple[float, str]]] = {}
    for side in ("left", "right"):
        index = name_to_index.get(f"{side}_wrist")
        if index is None or not valid_point(index, body_xy, body_scores, config.wrist_threshold):
            continue
        wrist = body_xy[index]
        for hand_index, hand in enumerate(hands):
            x1, y1, x2, y2 = map(float, hand["bbox_xyxy"])
            diagonal = max(float(np.hypot(x2 - x1, y2 - y1)), EPSILON)
            center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2])
            distance = float(np.linalg.norm(center - wrist) / diagonal)
            if distance <= config.max_wrist_distance:
                candidates.append((distance, side, hand_index))
                by_hand.setdefault(hand_index, []).append((distance, side))
    ambiguous = {
        hand_index for hand_index, distances in by_hand.items()
        if len(distances) > 1
        and sorted(distance for distance, _ in distances)[1]
        - sorted(distance for distance, _ in distances)[0]
        < config.minimum_wrist_margin
    }
    assigned: Dict[str, Optional[int]] = {"left": None, "right": None}
    used = set()
    for _, side, hand_index in sorted(candidates):
        if hand_index in ambiguous:
            continue
        if assigned[side] is None and hand_index not in used:
            assigned[side] = hand_index
            used.add(hand_index)
    return assigned


def hand_structure_pair_metric(
    previous: Mapping[str, Any], current: Mapping[str, Any], config: HandTemporalConfig
) -> Tuple[Optional[float], int]:
    first_xy, first_scores = pose_arrays(previous, "hand_pose")
    second_xy, second_scores = pose_arrays(current, "hand_pose")
    lengths = []
    for start, end in HAND_BONES:
        if not all((
            valid_point(start, first_xy, first_scores, config.keypoint_threshold),
            valid_point(end, first_xy, first_scores, config.keypoint_threshold),
            valid_point(start, second_xy, second_scores, config.keypoint_threshold),
            valid_point(end, second_xy, second_scores, config.keypoint_threshold),
        )):
            continue
        before = float(np.linalg.norm(first_xy[start] - first_xy[end]))
        after = float(np.linalg.norm(second_xy[start] - second_xy[end]))
        if min(before, after) > EPSILON:
            lengths.append((before, after))
    if len(lengths) < config.min_valid_bones:
        return None, len(lengths)
    global_scale = float(np.median([after / before for before, after in lengths]))
    jumps = [
        abs(after - global_scale * before)
        / (0.5 * (after + global_scale * before) + EPSILON)
        for before, after in lengths
    ]
    return p90(jumps), len(lengths)


def _normalized_hand(record: Mapping[str, Any], config: HandTemporalConfig):
    xy, scores = pose_arrays(record, "hand_pose")
    if not len(xy) or not valid_point(0, xy, scores, config.keypoint_threshold):
        return np.empty((0, 2)), np.empty(0)
    x1, y1, x2, y2 = map(float, record["bbox_xyxy"])
    scale = max(float(np.hypot(x2 - x1, y2 - y1)), EPSILON)
    return (xy - xy[0]) / scale, scores


def hand_motion_triplet_metric(first, middle, last, config: HandTemporalConfig):
    dt1 = int(middle["frame_index"]) - int(first["frame_index"])
    dt2 = int(last["frame_index"]) - int(middle["frame_index"])
    if min(dt1, dt2) <= 0 or max(dt1, dt2) > config.max_frame_gap:
        return None, 0
    poses = [_normalized_hand(item, config) for item in (first, middle, last)]
    values = []
    for index in range(21):
        if not all(valid_point(index, xy, scores, config.keypoint_threshold) for xy, scores in poses):
            continue
        p0, p1, p2 = (pose[0][index] for pose in poses)
        v1, v2 = (p1 - p0) / dt1, (p2 - p1) / dt2
        values.append(float(np.linalg.norm(v2 - v1) / ((dt1 + dt2) / 2)))
    if len(values) < config.min_valid_joints:
        return None, len(values)
    return p90(values), len(values)


def analyze_hand_side(records: Sequence[Mapping[str, Any]], config: HandTemporalConfig) -> Dict[str, Any]:
    observations = sorted(records, key=lambda item: int(item["frame_index"]))
    metrics = {int(item["frame_index"]): {
        "frame_index": int(item["frame_index"]), "hand_bone_length_jump": None,
        "hand_joint_acceleration": None, "valid_bones": 0, "valid_joints": 0,
    } for item in observations}
    for previous, current in zip(observations, observations[1:]):
        gap = int(current["frame_index"]) - int(previous["frame_index"])
        if gap <= 0 or gap > config.max_frame_gap:
            continue
        value, count = hand_structure_pair_metric(previous, current, config)
        metrics[int(current["frame_index"])]["hand_bone_length_jump"] = value
        metrics[int(current["frame_index"])]["valid_bones"] = count
    for first, middle, last in zip(observations, observations[1:], observations[2:]):
        value, count = hand_motion_triplet_metric(first, middle, last, config)
        metrics[int(middle["frame_index"])]["hand_joint_acceleration"] = value
        metrics[int(middle["frame_index"])]["valid_joints"] = count
    ordered = [metrics[int(item["frame_index"])] for item in observations]
    structure = [float(item["hand_bone_length_jump"]) for item in ordered if item["hand_bone_length_jump"] is not None]
    motion = [float(item["hand_joint_acceleration"]) for item in ordered if item["hand_joint_acceleration"] is not None]
    return {
        "valid": bool(structure or motion), "observed_frames": len(observations),
        "valid_structure_pairs": len(structure), "valid_motion_triplets": len(motion),
        "metrics": {**summary(structure, "bone_length_jump"), **summary(motion, "joint_acceleration")},
        "frame_metrics": ordered, "score": None,
    }


def aggregate_hand_temporal(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    frame_metrics = []
    structure, motion = [], []
    for side, result in (("left", left), ("right", right)):
        for item in result["frame_metrics"]:
            tagged = {**item, "side": side}
            frame_metrics.append(tagged)
            if item["hand_bone_length_jump"] is not None:
                structure.append(float(item["hand_bone_length_jump"]))
            if item["hand_joint_acceleration"] is not None:
                motion.append(float(item["hand_joint_acceleration"]))
    frame_metrics.sort(key=lambda item: (item["frame_index"], item["side"]))
    def worst_with_side(name: str):
        ranked = [
            {"frame_index": int(item["frame_index"]), "side": item["side"],
             "value": float(item[name])}
            for item in frame_metrics if item.get(name) is not None
        ]
        ranked.sort(key=lambda item: (-item["value"], item["frame_index"], item["side"]))
        return ranked[:5]

    return {
        "left": left, "right": right, "valid": left["valid"] or right["valid"],
        "metrics": {**summary(structure, "bone_length_jump"), **summary(motion, "joint_acceleration")},
        "worst_structure_frames": worst_with_side("hand_bone_length_jump"),
        "worst_motion_frames": worst_with_side("hand_joint_acceleration"),
        "score": None,
    }
