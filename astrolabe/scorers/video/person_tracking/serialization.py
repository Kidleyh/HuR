"""Serialization helpers for the person-tracking schema."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

from .schemas import VideoTrackingResult


class TrackingSerializationError(RuntimeError):
    """Raised when a tracking artifact cannot be serialized."""


RAW_CSV_FIELDS = [
    "video_name", "frame_index", "timestamp_sec", "detection_index", "class_id",
    "class_name", "confidence", "x1", "y1", "x2", "y2", "width", "height",
    "bbox_area_ratio",
]
TRACKED_CSV_FIELDS = [
    "video_name", "frame_index", "timestamp_sec", "track_id", "source_detection_index",
    "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "width",
    "height", "bbox_area_ratio",
]


def _bbox_row(detection: Any) -> Dict[str, float]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    _, _, width, height = detection.bbox_xywh
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": width,
        "height": height,
        "bbox_area_ratio": detection.bbox_area_ratio,
    }


def write_tracking_outputs(
    result: VideoTrackingResult,
    output_dir: Path,
    *,
    save_raw_csv: bool = True,
) -> None:
    """Write per-frame JSONL, distinct raw/tracked CSVs, and schema-1.1 summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (output_dir / "detections.jsonl").open("w", encoding="utf-8") as handle:
            for frame in result.frames:
                handle.write(json.dumps(frame.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")

        video_name = Path(result.video.path).name
        if save_raw_csv:
            with (output_dir / "raw_detections.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=RAW_CSV_FIELDS)
                writer.writeheader()
                for frame in result.frames:
                    for detection in frame.raw_detections:
                        writer.writerow(
                            {
                                "video_name": video_name,
                                "frame_index": frame.frame_index,
                                "timestamp_sec": frame.timestamp_sec,
                                "detection_index": detection.detection_index,
                                "class_id": detection.class_id,
                                "class_name": detection.class_name,
                                "confidence": detection.confidence,
                                **_bbox_row(detection),
                            }
                        )

        tracked_rows = []
        for frame in result.frames:
            for detection in frame.tracked_detections:
                tracked_rows.append(
                    {
                        "video_name": video_name,
                        "frame_index": frame.frame_index,
                        "timestamp_sec": frame.timestamp_sec,
                        "track_id": detection.track_id,
                        "source_detection_index": (
                            "" if detection.source_detection_index is None
                            else detection.source_detection_index
                        ),
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        **_bbox_row(detection),
                    }
                )
        for filename in ("tracked_detections.csv", "detections.csv"):
            with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRACKED_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(tracked_rows)

        with (output_dir / "tracks_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(result.summary_dict(), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as error:
        raise TrackingSerializationError(
            f"Failed to serialize tracking outputs in {output_dir}: {error}"
        ) from error


def write_error_json(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (output_dir / "error.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as error:
        raise TrackingSerializationError(f"Failed to write {output_dir / 'error.json'}: {error}") from error
