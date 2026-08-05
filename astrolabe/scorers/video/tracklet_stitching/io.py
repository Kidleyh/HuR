"""Read schema-1.1 tracking outputs and discover result directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from astrolabe.scorers.video.person_tracking.schemas import FrameDetections, RawDetection, TrackedDetection

from .schemas import Tracklet


@dataclass(frozen=True)
class TrackingInput:
    source_dir: Path
    summary: Dict[str, Any]
    frames: List[FrameDetections]


def discover_tracking_results(input_path: Path, recursive: bool = False) -> List[Path]:
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        raise ValueError(f"Input must be a tracking result directory: {path}")
    if (path / "detections.jsonl").is_file() and (path / "tracks_summary.json").is_file():
        return [path]
    iterator = path.rglob("tracks_summary.json") if recursive else path.glob("*/tracks_summary.json")
    results = sorted(item.parent for item in iterator if (item.parent / "detections.jsonl").is_file())
    if not results:
        raise FileNotFoundError(f"No tracking result directories found in: {path}")
    return results


def output_dir_for_result(source_dir: Path, input_root: Path, output_root: Path) -> Path:
    source, root, output = source_dir.resolve(), input_root.resolve(), output_root.resolve()
    if (root / "detections.jsonl").is_file():
        return output / source.name
    return output / source.relative_to(root)


def _raw(data: Dict[str, Any]) -> RawDetection:
    allowed = {"class_id", "class_name", "confidence", "bbox_xyxy", "bbox_xywh",
               "bbox_xyxy_normalized", "bbox_area_ratio", "detection_index"}
    return RawDetection(**{key: data[key] for key in allowed})


def _tracked(data: Dict[str, Any]) -> TrackedDetection:
    allowed = {"track_id", "class_id", "class_name", "confidence", "bbox_xyxy", "bbox_xywh",
               "bbox_xyxy_normalized", "bbox_area_ratio", "source_detection_index"}
    return TrackedDetection(**{key: data.get(key) for key in allowed})


def load_tracking_input(source_dir: Path) -> TrackingInput:
    source = source_dir.expanduser().resolve()
    jsonl_path, summary_path = source / "detections.jsonl", source / "tracks_summary.json"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Missing detections.jsonl: {jsonl_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing tracks_summary.json: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "1.1":
        raise ValueError(f"Expected source tracking schema 1.1, got {summary.get('schema_version')!r}")
    frames: List[FrameDetections] = []
    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        data = json.loads(line)
        frames.append(FrameDetections(
            frame_index=int(data["frame_index"]), timestamp_sec=float(data["timestamp_sec"]),
            raw_detections=[_raw(item) for item in data.get("raw_detections", [])],
            tracked_detections=[_tracked(item) for item in data.get("tracked_detections", [])],
        ))
    expected = int(summary.get("video", {}).get("num_frames", len(frames)))
    if len(frames) != expected:
        raise ValueError(f"detections.jsonl has {len(frames)} frames; summary declares {expected}")
    return TrackingInput(source, summary, frames)


def build_tracklets(frames: Sequence[FrameDetections]) -> List[Tracklet]:
    observations: Dict[int, Dict[int, TrackedDetection]] = {}
    for frame in frames:
        for detection in frame.tracked_detections:
            by_frame = observations.setdefault(detection.track_id, {})
            previous = by_frame.get(frame.frame_index)
            if previous is None or detection.confidence > previous.confidence:
                by_frame[frame.frame_index] = detection
    tracklets = []
    for track_id in sorted(observations):
        by_frame = observations[track_id]
        indices = sorted(by_frame)
        tracklets.append(Tracklet(track_id, indices, [by_frame[i] for i in indices], indices[0], indices[-1]))
    return tracklets
