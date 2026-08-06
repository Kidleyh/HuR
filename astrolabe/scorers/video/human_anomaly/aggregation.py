"""Track-level observation-quality and video-level anatomy aggregation."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, List, Sequence, Tuple

from .schema import HumanAnomalyInput


def aggregate_human_anomaly(
    entries: Sequence[HumanAnomalyInput],
    frame_results: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Aggregate person-frame records without treating missing parts as anomalies."""
    results_by_key = {
        (int(item["frame_index"]), int(item["logical_track_id"])): item
        for item in frame_results
    }
    by_track: Dict[int, List[HumanAnomalyInput]] = defaultdict(list)
    for entry in entries:
        by_track[entry.logical_track_id].append(entry)
    tracks: List[Dict[str, Any]] = []
    total_scored = total_abnormal = total_observed = 0
    for logical_id in sorted(by_track):
        observations = by_track[logical_id]
        results = [
            results_by_key.get((entry.frame_index, logical_id)) for entry in observations
        ]
        scored = [
            item for item in results
            if item is not None and item.get("human", {}).get("scored") is True
        ]
        abnormal = sum(bool(item.get("person_abnormal")) for item in scored)
        human_abnormal = sum(bool(item.get("human", {}).get("abnormal")) for item in scored)
        face_detected = sum(bool(item.get("faces")) for item in scored)
        hand_detected = sum(bool(item.get("hands")) for item in scored)
        face_abnormal = sum(
            any(face.get("abnormal", False) for face in item.get("faces", []))
            for item in scored
        )
        hand_abnormal = sum(
            any(hand.get("abnormal", False) for hand in item.get("hands", []))
            for item in scored
        )
        areas = []
        boundary_hits = 0
        for entry in observations:
            x1, y1, x2, y2 = entry.bbox_xyxy
            areas.append(((x2 - x1) * (y2 - y1)) / (width * height))
            boundary_hits += int(x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height)
        observed_count, scored_count = len(observations), len(scored)
        quality = 1 - abnormal / scored_count if scored_count else None
        track = {
            "logical_track_id": logical_id,
            "observed_frames": observed_count,
            "scored_frames": scored_count,
            "scored_frame_coverage": scored_count / observed_count if observed_count else 0.0,
            "abnormal_frames": abnormal,
            "anatomy_quality_score": quality,
            "human_anomaly_rate": human_abnormal / scored_count if scored_count else 0.0,
            "face_anomaly_rate": face_abnormal / scored_count if scored_count else 0.0,
            "hand_anomaly_rate": hand_abnormal / scored_count if scored_count else 0.0,
            "face_detected_frames": face_detected,
            "face_detection_coverage": face_detected / scored_count if scored_count else 0.0,
            "hand_detected_frames": hand_detected,
            "hand_detection_coverage": hand_detected / scored_count if scored_count else 0.0,
            "median_bbox_area_ratio": median(areas) if areas else 0.0,
            "boundary_truncation_rate": boundary_hits / observed_count if observed_count else 0.0,
        }
        tracks.append(track)
        total_observed += observed_count
        total_scored += scored_count
        total_abnormal += abnormal
    valid_quality = [
        item["anatomy_quality_score"]
        for item in tracks
        if item["anatomy_quality_score"] is not None
    ]
    summary = {
        "logical_track_count": len(tracks),
        "observed_person_frames": total_observed,
        "scored_person_frames": total_scored,
        "abnormal_person_frames": total_abnormal,
        "video_micro_score": 1 - total_abnormal / total_scored if total_scored else None,
        "video_macro_score": mean(valid_quality) if valid_quality else None,
        "failed_person_frames": total_observed - total_scored,
    }
    return tracks, summary
