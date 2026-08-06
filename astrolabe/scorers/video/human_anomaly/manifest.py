"""Convert HuR stitched detections into the VBench worker manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

from .schema import HumanAnomalyInput, ManifestFailure


def _video_dimensions(video_path: Path) -> Tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Input video cannot be opened: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Input video has invalid dimensions: {width}x{height}")
    return width, height


def _clipped_bbox(values: Sequence[float], width: int, height: int) -> List[float]:
    if len(values) != 4:
        raise ValueError("bbox_xyxy must have four values")
    x1 = min(max(float(values[0]), 0.0), float(width))
    y1 = min(max(float(values[1]), 0.0), float(height))
    x2 = min(max(float(values[2]), 0.0), float(width))
    y2 = min(max(float(values[3]), 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox is empty after clipping to the video frame")
    return [x1, y1, x2, y2]


def build_human_anomaly_manifest(
    video_path: Path, stitching_dir: Path
) -> Tuple[List[HumanAnomalyInput], List[ManifestFailure], int, int]:
    """Build a stable, deduplicated per-person-frame manifest from HuR schema."""
    video = video_path.expanduser().resolve()
    source = stitching_dir.expanduser().resolve() / "stitched_detections.jsonl"
    if not video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video}")
    if not source.is_file():
        raise FileNotFoundError(f"Missing stitched detections: {source}")
    width, height = _video_dimensions(video)
    selected: Dict[Tuple[int, int], HumanAnomalyInput] = {}
    failures: List[ManifestFailure] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        frame = json.loads(line)
        frame_index = frame.get("frame_index")
        for detection in frame.get("tracked_detections", []):
            logical_id = detection.get("logical_track_id")
            source_id = detection.get("track_id")
            try:
                if not isinstance(frame_index, int) or not isinstance(logical_id, int) or not isinstance(source_id, int):
                    raise ValueError("frame_index, logical_track_id and track_id must be integers")
                entry = HumanAnomalyInput(
                    frame_index=frame_index,
                    logical_track_id=logical_id,
                    source_track_id=source_id,
                    bbox_xyxy=_clipped_bbox(detection.get("bbox_xyxy", []), width, height),
                    detection_confidence=float(detection.get("confidence")),
                )
            except (TypeError, ValueError) as error:
                failures.append(ManifestFailure(
                    frame_index=frame_index if isinstance(frame_index, int) else None,
                    logical_track_id=logical_id if isinstance(logical_id, int) else None,
                    source_track_id=source_id if isinstance(source_id, int) else None,
                    failure_reason=f"line {line_number}: {error}",
                ))
                continue
            key = (entry.frame_index, entry.logical_track_id)
            previous = selected.get(key)
            if previous is None or (
                entry.detection_confidence,
                -entry.source_track_id,
            ) > (
                previous.detection_confidence,
                -previous.source_track_id,
            ):
                discarded = previous
                selected[key] = entry
            else:
                discarded = entry
            if previous is not None:
                failures.append(ManifestFailure(
                    frame_index=discarded.frame_index,
                    logical_track_id=discarded.logical_track_id,
                    source_track_id=discarded.source_track_id,
                    failure_reason="duplicate person-frame detection discarded by confidence",
                ))
    entries = [selected[key] for key in sorted(selected)]
    return entries, failures, width, height


def write_input_manifest(entries: Sequence[HumanAnomalyInput], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
