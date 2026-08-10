#!/usr/bin/env python3
"""Score a directory of positive gt.mp4 / negative render.mp4 video pairs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrolabe.scorers.video.human_reward import HumanRewardConfig, HumanRewardModel

LOGGER = logging.getLogger("human_reward_pairs")
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class VideoPair:
    """One named positive/negative pair discovered under the input root."""

    name: str
    positive: Path
    negative: Path


def discover_video_pairs(input_dir: Path) -> List[VideoPair]:
    """Discover complete immediate-child pairs in stable Unicode name order."""
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Pair input directory does not exist: {root}")
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not directories:
        raise ValueError(f"Pair input directory contains no sample folders: {root}")
    pairs: List[VideoPair] = []
    incomplete = []
    for directory in directories:
        positive = directory / "gt.mp4"
        negative = directory / "render.mp4"
        missing = [
            filename
            for filename, path in (("gt.mp4", positive), ("render.mp4", negative))
            if not path.is_file()
        ]
        if missing:
            incomplete.append(f"{directory.name}: missing {', '.join(missing)}")
            continue
        pairs.append(VideoPair(directory.name, positive.resolve(), negative.resolve()))
    if incomplete:
        raise ValueError("Incomplete video pair folders: " + "; ".join(incomplete))
    return pairs


def build_paired_result(
    input_dir: Path,
    pairs: Sequence[VideoPair],
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Map ordered batch outputs back to named positive/negative samples."""
    expected = 2 * len(pairs)
    if len(results) != expected:
        raise RuntimeError(
            f"Human Reward returned {len(results)} results for {expected} videos"
        )
    output_pairs = []
    for index, pair in enumerate(pairs):
        positive_result = results[2 * index]
        negative_result = results[2 * index + 1]
        output_pairs.append({
            "name": pair.name,
            "positive": {
                "kind": "gt",
                "video_path": str(pair.positive),
                "result": positive_result,
            },
            "negative": {
                "kind": "render",
                "video_path": str(pair.negative),
                "result": negative_result,
            },
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(Path(input_dir).expanduser().resolve()),
        "pair_count": len(pairs),
        "video_count": expected,
        "pairs": output_pairs,
    }


def write_json_atomic(path: Path, data: Any) -> None:
    """Write the sole aggregate artifact only after serialization succeeds."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    defaults = HumanRewardConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", default=str(defaults.yolo_weights))
    parser.add_argument("--tracker-config", default=str(defaults.tracker_config))
    parser.add_argument("--stitching-config", default=str(defaults.stitching_config))
    parser.add_argument("--vbench-root", default=str(defaults.vbench_root))
    parser.add_argument("--vbench-cache-dir", default=str(defaults.vbench_cache_dir))
    parser.add_argument("--vbench-clip-model", default=str(defaults.vbench_clip_model))
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-batch-size", type=int, default=128)
    parser.add_argument("--human-temporal", action="store_true")
    parser.add_argument("--human-temporal-pose-config")
    parser.add_argument("--human-temporal-pose-checkpoint")
    parser.add_argument(
        "--human-temporal-keypoint-threshold", type=float, default=0.3
    )
    parser.add_argument("--human-temporal-max-frame-gap", type=int, default=2)
    return parser


def _config_from_args(args: argparse.Namespace) -> HumanRewardConfig:
    return HumanRewardConfig(
        yolo_weights=Path(args.weights),
        tracker_config=Path(args.tracker_config),
        stitching_config=Path(args.stitching_config),
        vbench_root=Path(args.vbench_root),
        vbench_cache_dir=Path(args.vbench_cache_dir),
        vbench_clip_model=Path(args.vbench_clip_model),
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=args.half,
        crop_batch_size=args.crop_batch_size,
        human_temporal=args.human_temporal,
        human_temporal_pose_config=(
            Path(args.human_temporal_pose_config)
            if args.human_temporal_pose_config else None
        ),
        human_temporal_pose_checkpoint=(
            Path(args.human_temporal_pose_checkpoint)
            if args.human_temporal_pose_checkpoint else None
        ),
        human_temporal_keypoint_threshold=(
            args.human_temporal_keypoint_threshold
        ),
        human_temporal_max_frame_gap=args.human_temporal_max_frame_gap,
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    pairs = discover_video_pairs(Path(args.input_dir))
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    video_paths = [
        path for pair in pairs for path in (pair.positive, pair.negative)
    ]
    LOGGER.info("Scoring %d pairs (%d videos)", len(pairs), len(video_paths))
    results = HumanRewardModel(_config_from_args(args)).score_batch(video_paths)
    aggregate = build_paired_result(Path(args.input_dir), pairs, results)
    write_json_atomic(Path(args.output), aggregate)
    valid = sum(
        result.get("valid") is True
        for pair in aggregate["pairs"]
        for result in (pair["positive"]["result"], pair["negative"]["result"])
    )
    LOGGER.info(
        "Completed %d/%d valid videos; output=%s",
        valid,
        len(video_paths),
        Path(args.output).expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
