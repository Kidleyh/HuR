"""Candidate generation, hard gates, and explainable edge scoring."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from astrolabe.scorers.video.person_tracking.schemas import FrameDetections, RawDetection

from .features import (
    bbox_center_size, bbox_iou, interpolate_bbox, normalized_center_distance,
    predict_tracklet_bbox, ratio_change, scale_score,
)
from .schemas import (
    CandidateEdge,
    RawBridgeMatch,
    RawBridgeResult,
    StitchingConfig,
    Tracklet,
)


def raw_bridge_score(
    first: Tracklet,
    second: Tracklet,
    frames_by_index: Dict[int, FrameDetections],
    config: StitchingConfig,
) -> RawBridgeResult:
    gap = second.start_frame - first.end_frame - 1
    if gap == 0:
        return RawBridgeResult(1.0, 1.0, 1.0, [], 0)
    if gap < 0:
        return RawBridgeResult(0.0, 0.0, 0.0, [], 0)
    start_box = first.end_detection.bbox_xyxy_normalized
    end_box = second.start_detection.bbox_xyxy_normalized
    matches: List[RawBridgeMatch] = []
    excluded_associated_count = 0
    for offset, frame_index in enumerate(range(first.end_frame + 1, second.start_frame), 1):
        expected = interpolate_bbox(start_box, end_box, offset / (gap + 1))
        best: Tuple[float, RawDetection] | None = None
        frame = frames_by_index.get(frame_index)
        associated_raw_indices = {
            detection.source_detection_index
            for detection in frame.tracked_detections
            if detection.source_detection_index is not None
        } if frame else set()
        for raw in frame.raw_detections if frame else []:
            if (
                not config.raw_bridge_allow_associated_raw
                and raw.detection_index in associated_raw_indices
            ):
                excluded_associated_count += 1
                continue
            box = raw.bbox_xyxy_normalized
            distance = normalized_center_distance(expected, box)
            _, _, ew, eh = bbox_center_size(expected)
            _, _, rw, rh = bbox_center_size(box)
            area_change = ratio_change(ew * eh, rw * rh)
            aspect_change = ratio_change(ew / eh, rw / rh) if min(eh, rh) > 0 else math.inf
            if (distance > config.raw_bridge_max_center_distance or
                    area_change > config.raw_bridge_max_area_ratio_change or
                    aspect_change > config.raw_bridge_max_aspect_ratio_change):
                continue
            center_score = math.exp(-(distance ** 2) / (2 * config.motion_sigma ** 2))
            compatibility = 0.6 * center_score + 0.25 * scale_score(expected, box) + 0.15 * bbox_iou(expected, box)
            if best is None or (compatibility, -raw.detection_index) > (best[0], -best[1].detection_index):
                best = (compatibility, raw)
        if best is not None:
            matches.append(RawBridgeMatch(frame_index, best[1].detection_index, float(best[0])))
    coverage = len(matches) / gap
    compatibility = sum(item.compatibility for item in matches) / len(matches) if matches else 0.0
    return RawBridgeResult(
        score=0.5 * coverage + 0.5 * compatibility,
        coverage=coverage,
        compatibility=compatibility,
        matches=matches,
        excluded_associated_count=excluded_associated_count,
    )


def score_candidate(
    first: Tracklet,
    second: Tracklet,
    frames_by_index: Dict[int, FrameDetections],
    config: StitchingConfig,
) -> CandidateEdge:
    gap = second.start_frame - first.end_frame - 1
    edge = CandidateEdge(first.track_id, second.track_id, gap)
    if first.end_frame >= second.start_frame:
        edge.rejection_reasons.append("time_overlap")
    if gap > config.max_gap_frames:
        edge.rejection_reasons.append("gap_too_large")
    predicted = predict_tracklet_bbox(first, second.start_frame, config.velocity_window)
    if predicted is None:
        edge.rejection_reasons.append("invalid_predicted_bbox")
        return edge
    target = second.start_detection.bbox_xyxy_normalized
    distance = normalized_center_distance(predicted, target)
    _, _, pw, ph = bbox_center_size(predicted)
    _, _, tw, th = bbox_center_size(target)
    area_change = ratio_change(pw * ph, tw * th)
    aspect_change = ratio_change(pw / ph, tw / th) if min(ph, th) > 0 else math.inf
    edge.normalized_center_distance = distance
    edge.area_ratio_change = area_change
    edge.aspect_ratio_change = aspect_change
    if distance > config.max_normalized_center_distance:
        edge.rejection_reasons.append("center_distance_too_large")
    if area_change > config.max_area_ratio_change:
        edge.rejection_reasons.append("area_ratio_change_too_large")
    if aspect_change > config.max_aspect_ratio_change:
        edge.rejection_reasons.append("aspect_ratio_change_too_large")
    if edge.rejection_reasons:
        return edge
    edge.time_score = math.exp(-gap / config.time_tau)
    edge.motion_score = math.exp(-(distance ** 2) / (2 * config.motion_sigma ** 2))
    edge.predicted_iou_score = bbox_iou(predicted, target)
    edge.scale_score = scale_score(predicted, target)
    bridge = raw_bridge_score(first, second, frames_by_index, config)
    edge.raw_bridge_score = bridge.score
    edge.raw_bridge_coverage = bridge.coverage
    edge.raw_bridge_compatibility = bridge.compatibility
    edge.raw_bridge_matches = bridge.matches
    edge.raw_bridge_excluded_associated_count = bridge.excluded_associated_count
    components = {
        "time": edge.time_score, "motion": edge.motion_score,
        "predicted_iou": edge.predicted_iou_score, "scale": edge.scale_score,
        "raw_bridge": edge.raw_bridge_score,
    }
    total_weight = sum(config.weights.values())
    edge.score = sum(config.weights[name] * value for name, value in components.items()) / total_weight
    if edge.score >= config.merge_threshold:
        edge.decision = "eligible"
    elif edge.score >= config.uncertain_threshold:
        edge.decision = "uncertain"
    else:
        edge.decision = "rejected"
        edge.rejection_reasons.append("score_below_uncertain_threshold")
    return edge


def generate_candidates(
    tracklets: Sequence[Tracklet], frames: Sequence[FrameDetections], config: StitchingConfig
) -> List[CandidateEdge]:
    """Score each deterministic chronological pair exactly once."""
    ordered = sorted(tracklets, key=lambda item: (item.start_frame, item.track_id))
    frames_by_index = {frame.frame_index: frame for frame in frames}
    return [
        score_candidate(first, second, frames_by_index, config)
        for left, first in enumerate(ordered)
        for second in ordered[left + 1:]
    ]
