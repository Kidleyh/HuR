"""Atomic serialization of offline stitching artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

from astrolabe.scorers.video.person_tracking.schemas import FrameDetections

from .schemas import StitchingResult
from .visualization import write_stitched_video

GENERATED_FILENAMES = frozenset(
    {
        "tracklet_stitching.json",
        "stitched_detections.jsonl",
        "stitched_tracks_summary.json",
        "stitched.mp4",
        "stitching_error.json",
    }
)


def _main_payload(result: StitchingResult) -> Dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "source_tracking_schema_version": result.source_tracking_schema_version,
        "config": result.config.to_dict(),
        "track_id_to_logical_track_id": {
            str(key): value
            for key, value in sorted(result.track_id_to_logical_track_id.items())
        },
        "merged_edges": [edge.to_dict() for edge in result.merged_edges],
        "uncertain_edges": [edge.to_dict() for edge in result.uncertain_edges],
        "rejected_edges": [edge.to_dict() for edge in result.rejected_edges],
        "logical_tracks": [track.to_dict() for track in result.logical_tracks],
        "warnings": result.warnings,
        "runtime_sec": result.runtime_sec,
    }


def _summary_payload(result: StitchingResult) -> Dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "source_tracking_schema_version": result.source_tracking_schema_version,
        "num_source_tracks": len(result.track_id_to_logical_track_id),
        "num_logical_tracks": len(result.logical_tracks),
        "num_candidate_edges": len(result.edges),
        "num_merged_edges": len(result.merged_edges),
        "num_uncertain_edges": len(result.uncertain_edges),
        "num_rejected_edges": len(result.rejected_edges),
        "logical_tracks": [track.to_dict() for track in result.logical_tracks],
        "warnings": result.warnings,
        "runtime_sec": result.runtime_sec,
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _promote_atomic(staging: Path, destination: Path) -> None:
    """Promote managed files and roll back if any replacement fails."""
    destination.mkdir(parents=True, exist_ok=True)
    produced = sorted(
        item for item in staging.iterdir() if item.name in GENERATED_FILENAMES
    )
    backup = staging / ".backup"
    backup.mkdir()
    changed = []
    try:
        for item in produced:
            target = destination / item.name
            if target.exists():
                shutil.copy2(target, backup / item.name)
            os.replace(item, target)
            changed.append(target)
        produced_names = {item.name for item in produced}
        for name in GENERATED_FILENAMES - produced_names:
            stale = destination / name
            if stale.exists():
                shutil.copy2(stale, backup / name)
                stale.unlink()
                changed.append(stale)
    except Exception:
        for target in changed:
            saved = backup / target.name
            if saved.exists():
                os.replace(saved, target)
            else:
                target.unlink(missing_ok=True)
        raise


def write_stitching_outputs(
    result: StitchingResult,
    frames: Sequence[FrameDetections],
    source_video: Path,
    output_dir: Path,
    save_visualization: bool,
) -> None:
    """Write a complete result in a temporary directory before promotion."""
    output = output_dir.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.tmp-"
    ) as name:
        staging = Path(name)
        if save_visualization:
            if source_video.is_file():
                write_stitched_video(
                    source_video,
                    frames,
                    result.track_id_to_logical_track_id,
                    staging / "stitched.mp4",
                )
            else:
                result.warnings.append(
                    "Original video is unavailable; visualization skipped: "
                    f"{source_video}"
                )
        _write_json(staging / "tracklet_stitching.json", _main_payload(result))
        _write_json(staging / "stitched_tracks_summary.json", _summary_payload(result))
        with (staging / "stitched_detections.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for frame in frames:
                payload = frame.to_dict()
                for detection in payload["tracked_detections"]:
                    detection["logical_track_id"] = (
                        result.track_id_to_logical_track_id[detection["track_id"]]
                    )
                handle.write(
                    json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
                )
        _promote_atomic(staging, output)


def write_stitching_error(output_dir: Path, payload: Dict[str, Any]) -> None:
    """Atomically record one result's error without touching prior good output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".stitching_error.{os.getpid()}.tmp"
    _write_json(temporary, payload)
    os.replace(temporary, output_dir / "stitching_error.json")
