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
from astrolabe.scorers.video.human_reward.visualization import (
    write_human_reward_visualization,
)

LOGGER = logging.getLogger("human_reward_pairs")
SCHEMA_VERSION = "1.0"
FULL_RESULT_FILENAME = "human_reward_pairs_full.json"
SCORES_RESULT_FILENAME = "human_reward_pairs_scores.json"


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


def build_scores_result(full_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact score-only view without person-frame payloads."""
    pairs = []
    for pair in full_result["pairs"]:
        compact_pair = {"name": pair["name"]}
        for side in ("positive", "negative"):
            source = pair[side]
            result = source["result"]
            persons = []
            for person in result.get("persons", []):
                temporal = person.get("temporal", {})
                human_temporal = temporal.get("human")
                if isinstance(human_temporal, dict):
                    human_temporal = {
                        key: value for key, value in human_temporal.items()
                        if key not in ("frame_metrics", "keypoint_name_to_index")
                    }
                persons.append({
                    "logical_track_id": person.get("logical_track_id"),
                    "track": person.get("track", {}),
                    "score": person.get("score", {}),
                    "human_temporal": human_temporal,
                })
            compact_pair[side] = {
                "kind": source["kind"],
                "video_path": source["video_path"],
                "valid": result.get("valid"),
                "reason": result.get("reason"),
                "reward": result.get("reward"),
                "micro_score": result.get("micro_score"),
                "macro_score": result.get("macro_score"),
                "logical_track_count": result.get("logical_track_count", 0),
                "observed_person_frames": result.get("observed_person_frames", 0),
                "scored_person_frames": result.get("scored_person_frames", 0),
                "abnormal_person_frames": result.get("abnormal_person_frames", 0),
                "failed_person_frames": result.get("failed_person_frames", 0),
                "visualization": result.get("visualization"),
                "persons": persons,
            }
        pairs.append(compact_pair)
    return {
        "schema_version": full_result["schema_version"],
        "input_dir": full_result["input_dir"],
        "pair_count": full_result["pair_count"],
        "video_count": full_result["video_count"],
        "pairs": pairs,
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


def write_pair_visualizations(
    pairs: Sequence[VideoPair],
    results: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    """Render results under ``<output>/<pair name>/{gt,render}.mp4``."""
    expected = 2 * len(pairs)
    if len(results) != expected:
        raise RuntimeError(
            f"Human Reward returned {len(results)} results for {expected} videos"
        )
    root = Path(output_dir).expanduser().resolve()
    destinations: List[Path] = []
    for index, pair in enumerate(pairs):
        for offset, (stem, video) in enumerate((
            ("gt", pair.positive),
            ("render", pair.negative),
        )):
            result = results[2 * index + offset]
            destination = root / pair.name / f"{stem}.mp4"
            write_human_reward_visualization(video, result, destination)
            result["visualization"] = str(destination)
            destinations.append(destination)
    return destinations


def build_parser() -> argparse.ArgumentParser:
    defaults = HumanRewardConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output directory containing human_reward_pairs_full.json and "
            "human_reward_pairs_scores.json"
        ),
    )
    parser.add_argument(
        "--visualization-dir",
        help=(
            "Optional root for <pair-name>/gt.mp4 and "
            "<pair-name>/render.mp4 visualizations"
        ),
    )
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
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output must be a directory path: {output_dir}")
    pairs = discover_video_pairs(Path(args.input_dir))
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    video_paths = [
        path for pair in pairs for path in (pair.positive, pair.negative)
    ]
    LOGGER.info("Scoring %d pairs (%d videos)", len(pairs), len(video_paths))
    results = HumanRewardModel(_config_from_args(args)).score_batch(video_paths)
    if args.visualization_dir:
        visualizations = write_pair_visualizations(
            pairs, results, Path(args.visualization_dir)
        )
        LOGGER.info(
            "Generated %d visualizations under %s",
            len(visualizations),
            Path(args.visualization_dir).expanduser().resolve(),
        )
    aggregate = build_paired_result(Path(args.input_dir), pairs, results)
    scores = build_scores_result(aggregate)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_output = output_dir / FULL_RESULT_FILENAME
    scores_output = output_dir / SCORES_RESULT_FILENAME
    write_json_atomic(full_output, aggregate)
    write_json_atomic(scores_output, scores)
    valid = sum(
        result.get("valid") is True
        for pair in aggregate["pairs"]
        for result in (pair["positive"]["result"], pair["negative"]["result"])
    )
    LOGGER.info(
        "Completed %d/%d valid videos; full=%s; scores=%s",
        valid,
        len(video_paths),
        full_output,
        scores_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
