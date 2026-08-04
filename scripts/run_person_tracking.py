#!/usr/bin/env python3
"""Run YOLOv8x person detection and ByteTrack on videos."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKER_CONFIG = PROJECT_ROOT / "configs/bytetrack_person.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrolabe.scorers.video.person_tracking.serialization import write_error_json
from astrolabe.scorers.video.person_tracking.tracker import (
    YOLOByteTrackPersonTracker,
    discover_videos,
    resolve_yolo_weights,
)

LOGGER = logging.getLogger("run_person_tracking")


def output_dir_for_video(video_path: Path, input_root: Path, output_root: Path) -> Path:
    """Preserve a video's relative parent path and remove only its final suffix."""
    video = video_path.expanduser().resolve()
    root = input_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if root.is_file() or root == video:
        return output / video.stem
    try:
        relative = video.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Video {video} is not under input root {root}") from error
    return output / relative.parent / relative.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Video file or directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--tracker-config", default=str(DEFAULT_TRACKER_CONFIG))
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-visualization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-raw-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--track-high-thresh", type=float)
    parser.add_argument("--track-low-thresh", type=float)
    parser.add_argument("--new-track-thresh", type=float)
    parser.add_argument("--track-buffer", type=int)
    parser.add_argument("--match-thresh", type=float)
    parser.add_argument("--fuse-score", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _effective_tracker_config(args: argparse.Namespace, output_root: Path) -> Path:
    config_path = Path(args.tracker_config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"ByteTrack config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"ByteTrack config must be a YAML mapping: {config_path}")
    overrides = {
        "track_high_thresh": args.track_high_thresh,
        "track_low_thresh": args.track_low_thresh,
        "new_track_thresh": args.new_track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "fuse_score": args.fuse_score,
    }
    supplied = {key: value for key, value in overrides.items() if value is not None}
    if not supplied:
        return config_path
    config.update(supplied)
    runtime_dir = output_root / "_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / "bytetrack_effective.yaml"
    with runtime_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    LOGGER.info("Using overridden ByteTrack config: %s", runtime_path)
    return runtime_path


def _is_complete(output_dir: Path, visualization: bool, raw_csv: bool) -> bool:
    required = [
        "detections.jsonl",
        "detections.csv",
        "tracked_detections.csv",
        "tracks_summary.json",
    ]
    if raw_csv:
        required.append("raw_detections.csv")
    if visualization:
        required.append("tracked.mp4")
    return all(
        (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
        for name in required
    )


def _record_error(video: Path, output_dir: Path, error: BaseException) -> None:
    write_error_json(
        output_dir,
        {
            "video": str(video),
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        },
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.max_videos is not None and args.max_videos <= 0:
        LOGGER.error("--max-videos must be positive")
        return 2
    try:
        videos = discover_videos(args.input, recursive=args.recursive)
    except Exception as error:
        LOGGER.error("Input discovery failed: %s", error)
        return 2
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        weights = resolve_yolo_weights(
            args.weights, allow_download=args.allow_download, project_root=PROJECT_ROOT
        )
        tracker_config = _effective_tracker_config(args, output_root)
        tracker = YOLOByteTrackPersonTracker(
            weights=weights,
            tracker_config=str(tracker_config),
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            half=args.half,
            allow_download=args.allow_download,
        )
    except Exception as error:
        LOGGER.error("Tracker setup failed: %s", error)
        for video in videos:
            try:
                _record_error(
                    video,
                    output_dir_for_video(video, input_root, output_root),
                    error,
                )
            except Exception as report_error:
                LOGGER.error("Could not write error report for %s: %s", video, report_error)
        return 1

    succeeded = skipped = failed = 0
    for index, video in enumerate(videos, start=1):
        video_output = output_dir_for_video(video, input_root, output_root)
        if (
            _is_complete(video_output, args.save_visualization, args.save_raw_csv)
            and not args.overwrite
        ):
            skipped += 1
            LOGGER.info("[%d/%d] Skipping complete result: %s", index, len(videos), video)
            continue
        LOGGER.info("[%d/%d] Processing %s", index, len(videos), video)
        try:
            result = tracker.track_video(
                str(video),
                str(video_output),
                save_visualization=args.save_visualization,
                save_raw_csv=args.save_raw_csv,
            )
            succeeded += 1
            summary = result.detection_summary
            LOGGER.info(
                "Completed %s: %d frames, %d raw detections, %d tracked detections, "
                "%d tracks, raw coverage=%.3f, tracked coverage=%.3f, %.2fs (%.2f FPS)",
                video.name,
                result.processing["processed_frames"],
                summary.total_raw_detections,
                summary.total_tracked_detections,
                len(result.tracks),
                summary.raw_person_frame_coverage,
                summary.tracked_person_frame_coverage,
                result.processing["runtime_sec"],
                result.processing["fps_effective"],
            )
        except Exception as error:
            failed += 1
            LOGGER.exception("Failed to process %s", video)
            try:
                _record_error(video, video_output, error)
            except Exception as report_error:
                LOGGER.error("Could not write error report: %s", report_error)

    LOGGER.info("Summary: success=%d skipped=%d failed=%d", succeeded, skipped, failed)
    return 1 if succeeded == 0 and failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
