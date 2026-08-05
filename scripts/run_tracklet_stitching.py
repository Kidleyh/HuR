#!/usr/bin/env python3
"""Run offline geometric and raw-detection-based tracklet stitching."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/tracklet_stitching.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrolabe.scorers.video.tracklet_stitching.io import (
    discover_tracking_results,
    load_tracking_input,
    output_dir_for_result,
)
from astrolabe.scorers.video.tracklet_stitching.schemas import StitchingConfig
from astrolabe.scorers.video.tracklet_stitching.serialization import (
    write_stitching_error,
    write_stitching_outputs,
)
from astrolabe.scorers.video.tracklet_stitching.stitcher import stitch_tracking

LOGGER = logging.getLogger("run_tracklet_stitching")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Tracking result directory or root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-visualization", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-gap-frames", type=int)
    parser.add_argument("--merge-threshold", type=float)
    parser.add_argument("--uncertain-threshold", type=float)
    parser.add_argument("--minimum-assignment-margin", type=float)
    parser.add_argument("--max-videos", type=int)
    return parser


def load_config(args: argparse.Namespace) -> StitchingConfig:
    path = Path(args.config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stitching config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Stitching config must be a YAML mapping: {path}")
    overrides = {
        "max_gap_frames": args.max_gap_frames,
        "merge_threshold": args.merge_threshold,
        "uncertain_threshold": args.uncertain_threshold,
        "minimum_assignment_margin": args.minimum_assignment_margin,
    }
    data.update({key: value for key, value in overrides.items() if value is not None})
    return StitchingConfig(**data)


def _is_complete(output_dir: Path, visualization: bool) -> bool:
    names = [
        "tracklet_stitching.json",
        "stitched_detections.jsonl",
        "stitched_tracks_summary.json",
    ]
    if visualization:
        names.append("stitched.mp4")
    return all(
        (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
        for name in names
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.max_videos is not None and args.max_videos <= 0:
        LOGGER.error("--max-videos must be positive")
        return 2
    try:
        config = load_config(args)
        input_root = Path(args.input).expanduser().resolve()
        output_root = Path(args.output_dir).expanduser().resolve()
        sources = discover_tracking_results(input_root, recursive=args.recursive)
        if args.max_videos is not None:
            sources = sources[: args.max_videos]
    except Exception as error:
        LOGGER.error("Setup failed: %s", error)
        return 2

    succeeded = skipped = failed = 0
    for index, source in enumerate(sources, 1):
        destination = output_dir_for_result(source, input_root, output_root)
        if _is_complete(destination, args.save_visualization) and not args.overwrite:
            skipped += 1
            LOGGER.info("[%d/%d] Skipping complete result: %s", index, len(sources), source)
            continue
        LOGGER.info("[%d/%d] Stitching %s", index, len(sources), source)
        try:
            tracking = load_tracking_input(source)
            result = stitch_tracking(tracking, config)
            video_path = Path(str(tracking.summary["video"]["path"])).expanduser()
            write_stitching_outputs(
                result,
                tracking.frames,
                video_path,
                destination,
                args.save_visualization,
            )
            succeeded += 1
            LOGGER.info(
                "Completed %s: source tracks=%d, candidate edges=%d, merged edges=%d, "
                "uncertain edges=%d, logical tracks=%d, runtime=%.3fs",
                source.name,
                len(result.track_id_to_logical_track_id),
                len(result.edges),
                len(result.merged_edges),
                len(result.uncertain_edges),
                len(result.logical_tracks),
                result.runtime_sec,
            )
        except Exception as error:
            failed += 1
            LOGGER.exception("Failed to stitch %s", source)
            try:
                write_stitching_error(
                    destination,
                    {
                        "source": str(source),
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": "".join(
                            traceback.format_exception(
                                type(error), error, error.__traceback__
                            )
                        ),
                    },
                )
            except Exception as report_error:
                LOGGER.error("Could not write stitching error: %s", report_error)
    LOGGER.info("Summary: success=%d skipped=%d failed=%d", succeeded, skipped, failed)
    return 1 if succeeded == 0 and failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
