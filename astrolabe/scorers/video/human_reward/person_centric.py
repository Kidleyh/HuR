"""Person-centric assembly and lightweight frame indexing for Human Reward."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from astrolabe.scorers.video.human_anomaly.schema import HumanAnomalyInput

FrameToPersonRefs = Dict[int, List[Tuple[int, int]]]


def _track_dict(track: Any) -> Dict[str, Any]:
    if isinstance(track, Mapping):
        return dict(track)
    to_dict = getattr(track, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if is_dataclass(track):
        return asdict(track)
    raise TypeError(f"Unsupported logical track metadata: {type(track).__name__}")


def _fallback_track(logical_track_id: int, frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Provide compatibility metadata when an older/mock stitcher omits summaries."""
    frame_indices = [int(frame["frame_index"]) for frame in frames]
    source_ids = sorted({int(frame["source_track_id"]) for frame in frames})
    gaps = [
        current - previous - 1
        for previous, current in zip(frame_indices, frame_indices[1:])
    ]
    return {
        "logical_track_id": logical_track_id,
        "source_track_ids": source_ids,
        "start_frame": frame_indices[0],
        "end_frame": frame_indices[-1],
        "num_observed_frames": len(frame_indices),
        "num_fragments": len(source_ids),
        "max_internal_gap": max(gaps, default=0),
    }


def build_person_centric_result(
    *,
    video: Mapping[str, Any],
    entries: Sequence[HumanAnomalyInput],
    frame_results: Sequence[Mapping[str, Any]],
    logical_tracks: Sequence[Any],
    track_scores: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Join flat anomaly results with stitching metadata, storing each frame once."""
    entries_by_key = {
        (entry.frame_index, entry.logical_track_id): entry for entry in entries
    }
    frames_by_person: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for result in frame_results:
        key = (int(result["frame_index"]), int(result["logical_track_id"]))
        entry = entries_by_key[key]
        person_frame = dict(result)
        person_frame.pop("logical_track_id", None)
        person_frame.update({
            "frame_index": entry.frame_index,
            "source_track_id": entry.source_track_id,
            "bbox_xyxy": list(entry.bbox_xyxy),
            "detection_confidence": entry.detection_confidence,
            "human": dict(result.get("human", {})),
            "faces": [dict(item) for item in result.get("faces", [])],
            "hands": [dict(item) for item in result.get("hands", [])],
            "person_abnormal": bool(result.get("person_abnormal", False)),
            "failure_reason": result.get("failure_reason"),
            "failures": [dict(item) for item in result.get("failures", [])],
        })
        frames_by_person[entry.logical_track_id].append(person_frame)

    metadata = {
        int(data["logical_track_id"]): data
        for data in (_track_dict(track) for track in logical_tracks)
    }
    scores = {
        int(score["logical_track_id"]): score for score in track_scores
    }
    persons: List[Dict[str, Any]] = []
    for logical_track_id in sorted(frames_by_person):
        frames = sorted(
            frames_by_person[logical_track_id], key=lambda item: int(item["frame_index"])
        )
        track = dict(
            metadata.get(
                logical_track_id, _fallback_track(logical_track_id, frames)
            )
        )
        track.pop("logical_track_id", None)
        aggregate = scores[logical_track_id]
        observed = int(aggregate["observed_frames"])
        scored = int(aggregate["scored_frames"])
        abnormal = int(aggregate["abnormal_frames"])
        persons.append({
            "logical_track_id": logical_track_id,
            "track": track,
            "frames": frames,
            "score": {
                "binary_score": aggregate["anatomy_quality_score"],
                "observed_frames": observed,
                "scored_frames": scored,
                "failed_frames": observed - scored,
                "scored_frame_coverage": aggregate["scored_frame_coverage"],
                "abnormal_frames": abnormal,
                "human_anomaly_rate": aggregate["human_anomaly_rate"],
                "face_anomaly_rate": aggregate["face_anomaly_rate"],
                "hand_anomaly_rate": aggregate["hand_anomaly_rate"],
                "face_detected_frames": aggregate["face_detected_frames"],
                "face_detection_coverage": aggregate["face_detection_coverage"],
                "hand_detected_frames": aggregate["hand_detected_frames"],
                "hand_detection_coverage": aggregate["hand_detection_coverage"],
                "median_bbox_area_ratio": aggregate["median_bbox_area_ratio"],
                "boundary_truncation_rate": aggregate["boundary_truncation_rate"],
            },
            "temporal": {},
        })

    micro = summary["video_micro_score"]
    macro = summary["video_macro_score"]
    return {
        "video": dict(video),
        "persons": persons,
        "video_score": {
            "reward": float(micro),
            "micro_score": float(micro),
            "macro_score": float(macro),
        },
    }


def build_frame_to_person_refs(persons: Sequence[Mapping[str, Any]]) -> FrameToPersonRefs:
    """Index person frames without copying any frame result dictionaries."""
    index: FrameToPersonRefs = defaultdict(list)
    for person in persons:
        logical_track_id = int(person["logical_track_id"])
        for person_frame_index, frame in enumerate(person["frames"]):
            index[int(frame["frame_index"])].append(
                (logical_track_id, person_frame_index)
            )
    for refs in index.values():
        refs.sort()
    return dict(index)
