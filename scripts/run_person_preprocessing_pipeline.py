#!/usr/bin/env python3
"""Run all implemented human preprocessing stages for one input video."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKER_CONFIG = PROJECT_ROOT / "configs/bytetrack_person.yaml"
DEFAULT_STITCHING_CONFIG = PROJECT_ROOT / "configs/tracklet_stitching.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrolabe.scorers.video.person_tracking.tracker import (
    YOLOByteTrackPersonTracker,
    VIDEO_EXTENSIONS,
    resolve_yolo_weights,
)
from astrolabe.scorers.video.tracklet_stitching.io import load_tracking_input
from astrolabe.scorers.video.tracklet_stitching.schemas import StitchingConfig
from astrolabe.scorers.video.tracklet_stitching.serialization import (
    write_stitching_outputs,
)
from astrolabe.scorers.video.tracklet_stitching.stitcher import stitch_tracking
from scripts.run_tracklet_stitching import is_complete_result

LOGGER = logging.getLogger("run_person_preprocessing_pipeline")


def pipeline_output_dirs(
    name: str, output_root: Path
) -> Tuple[Path, Path]:
    """Return exact stage directories derived from an explicit safe run name."""
    validate_run_name(name)
    root = output_root.expanduser().resolve()
    return root / f"{name}_person_tracking", root / f"{name}_tracklet_stitching"


def validate_run_name(name: str) -> None:
    """Reject empty names and path components that could escape output_root."""
    if (
        not isinstance(name, str)
        or not name.strip()
        or name != name.strip()
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError(
            "--name must be a non-empty single directory name without surrounding "
            "whitespace, path separators, '.' or '..'"
        )


def _tracking_complete(output_dir: Path, visualization: bool, raw_csv: bool) -> bool:
    names = [
        "detections.jsonl",
        "detections.csv",
        "tracked_detections.csv",
        "tracks_summary.json",
    ]
    if raw_csv:
        names.append("raw_detections.csv")
    if visualization:
        names.append("tracked.mp4")
    try:
        return all(
            (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
            for name in names
        )
    except OSError:
        return False


def _load_stitching_config(args: argparse.Namespace) -> StitchingConfig:
    path = Path(args.stitching_config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stitching config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Stitching config must be a YAML mapping: {path}")
    overrides: Dict[str, object] = {
        "max_gap_frames": args.max_gap_frames,
        "merge_threshold": args.merge_threshold,
        "uncertain_threshold": args.uncertain_threshold,
        "minimum_assignment_margin": args.minimum_assignment_margin,
        "raw_bridge_allow_associated_raw": args.raw_bridge_allow_associated_raw,
    }
    data.update({key: value for key, value in overrides.items() if value is not None})
    return StitchingConfig(**data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One input video file")
    parser.add_argument(
        "--name", required=True, help="Run name used in each stage output directory"
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--weights")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--tracker-config", default=str(DEFAULT_TRACKER_CONFIG))
    parser.add_argument("--stitching-config", default=str(DEFAULT_STITCHING_CONFIG))
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-visualization", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-raw-csv", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-gap-frames", type=int)
    parser.add_argument("--merge-threshold", type=float)
    parser.add_argument("--uncertain-threshold", type=float)
    parser.add_argument("--minimum-assignment-margin", type=float)
    parser.add_argument(
        "--raw-bridge-allow-associated-raw",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    video = Path(args.input).expanduser().resolve()
    if not video.is_file():
        LOGGER.error("Input video does not exist: %s", video)
        return 2
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        LOGGER.error("Unsupported video extension %s", video.suffix)
        return 2
    try:
        validate_run_name(args.name)
    except ValueError as error:
        LOGGER.error("Invalid run name: %s", error)
        return 2
    output_root = Path(args.output_root).expanduser().resolve()
    tracking_output, stitching_output = pipeline_output_dirs(args.name, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if _tracking_complete(
            tracking_output, args.save_visualization, args.save_raw_csv
        ) and not args.overwrite:
            LOGGER.info("Stage person_tracking: skipped complete output %s", tracking_output)
        else:
            weights = resolve_yolo_weights(
                args.weights,
                allow_download=args.allow_download,
                project_root=PROJECT_ROOT,
            )
            tracker = YOLOByteTrackPersonTracker(
                weights=weights,
                tracker_config=str(Path(args.tracker_config).expanduser().resolve()),
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                half=args.half,
                allow_download=args.allow_download,
            )
            tracker.track_video(
                str(video),
                str(tracking_output),
                save_visualization=args.save_visualization,
                save_raw_csv=args.save_raw_csv,
            )
            LOGGER.info("Stage person_tracking: completed %s", tracking_output)

        if is_complete_result(stitching_output, args.save_visualization) and not args.overwrite:
            LOGGER.info("Stage tracklet_stitching: skipped complete output %s", stitching_output)
        else:
            tracking = load_tracking_input(tracking_output)
            stitching = stitch_tracking(tracking, _load_stitching_config(args))
            write_stitching_outputs(
                stitching,
                tracking.frames,
                video,
                stitching_output,
                args.save_visualization,
            )
            LOGGER.info("Stage tracklet_stitching: completed %s", stitching_output)
        LOGGER.info(
            "Pipeline completed in %.2fs\nperson_tracking=%s\ntracklet_stitching=%s",
            time.perf_counter() - started,
            tracking_output,
            stitching_output,
        )
        return 0
    except Exception:
        LOGGER.exception("Person preprocessing pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
