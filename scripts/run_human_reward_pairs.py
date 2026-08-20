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
from astrolabe.scorers.video.human_reward.pair_evaluation import (
    build_pair_frame_evaluation,
    build_pair_score_summary,
)
from astrolabe.scorers.video.human_reward.visualization import (
    write_human_reward_visualization,
)

LOGGER = logging.getLogger("human_reward_pairs")
SCHEMA_VERSION = "1.0"
FULL_RESULT_FILENAME = "human_reward_pairs_full.json"
SCORES_RESULT_FILENAME = "human_reward_pairs_scores.json"
FRAME_EVALUATION_FILENAME = "human_reward_pair_frame_evaluation.json"
PAIR_EVALUATION_FILENAME = "human_reward_pair_evaluation.json"


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


def load_selected_video_pairs(manifest_path: Path) -> tuple[Path, List[VideoPair]]:
    """Load exactly the ordered pairs recorded by a selection manifest."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Selection manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read selection manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Selection manifest root must be a JSON object")
    data_root_value = manifest.get("data_root")
    selected = manifest.get("selected_pairs")
    if not isinstance(data_root_value, str) or not data_root_value.strip():
        raise ValueError("Selection manifest data_root must be a non-empty string")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Selection manifest selected_pairs must be a non-empty list")
    data_root = Path(data_root_value).expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Selection manifest data_root is missing: {data_root}")

    pairs = []
    seen = set()
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise ValueError(f"selected_pairs[{index}] must be an object")
        folder = item.get("folder")
        if not isinstance(folder, str) or not folder or Path(folder).name != folder:
            raise ValueError(
                f"selected_pairs[{index}].folder must be one directory name"
            )
        if folder in seen:
            raise ValueError(f"Duplicate selected pair folder: {folder}")
        seen.add(folder)
        directory = data_root / folder
        gt = directory / "gt.mp4"
        render = directory / "render.mp4"
        missing = [str(video) for video in (gt, render) if not video.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Selected pair {folder} is missing video(s): {', '.join(missing)}"
            )
        pairs.append(VideoPair(folder, gt.resolve(), render.resolve()))
    return data_root, pairs


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
    def compact_temporal(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: compact_temporal(item)
                for key, item in value.items()
                if key not in ("frame_metrics", "keypoint_name_to_index")
            }
        if isinstance(value, list):
            return [compact_temporal(item) for item in value]
        return value

    pairs = []
    for pair in full_result["pairs"]:
        compact_pair = {"name": pair["name"]}
        for side in ("positive", "negative"):
            source = pair[side]
            result = source["result"]
            persons = []
            for person in result.get("persons", []):
                temporal = person.get("temporal", {})
                persons.append({
                    "logical_track_id": person.get("logical_track_id"),
                    "track": person.get("track", {}),
                    "score": person.get("score", {}),
                    "human_temporal": compact_temporal(temporal.get("human")),
                    "head_temporal": compact_temporal(temporal.get("head")),
                    "hand_temporal": compact_temporal(temporal.get("hand")),
                    "human_temporal_3d": compact_temporal(
                        temporal.get("human_3d")
                    ),
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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir", help="Directory whose immediate children are pair folders"
    )
    input_group.add_argument(
        "--selection-manifest",
        help="JSON manifest containing data_root and ordered selected_pairs",
    )
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
    parser.add_argument(
        "--tie-epsilon", type=float, default=1e-12,
        help="Absolute GT/render quality difference treated as a tie",
    )
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
    for prefix in ("head", "hand"):
        parser.add_argument(f"--{prefix}-temporal", action="store_true")
        parser.add_argument(f"--{prefix}-temporal-pose-config")
        parser.add_argument(f"--{prefix}-temporal-pose-checkpoint")
        parser.add_argument(f"--{prefix}-temporal-keypoint-threshold", type=float, default=0.3)
        parser.add_argument(f"--{prefix}-temporal-max-frame-gap", type=int, default=2)
    parser.add_argument("--hand-temporal-wrist-threshold", type=float, default=0.3)
    parser.add_argument("--hand-temporal-max-wrist-distance", type=float, default=1.5)
    parser.add_argument("--human-temporal-3d", action="store_true")
    parser.add_argument("--gvhmr-root")
    parser.add_argument("--gvhmr-checkpoint")
    parser.add_argument("--human-temporal-3d-min-valid-joints", type=int, default=1)
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
        head_temporal=args.head_temporal,
        head_temporal_pose_config=(Path(args.head_temporal_pose_config) if args.head_temporal_pose_config else None),
        head_temporal_pose_checkpoint=(Path(args.head_temporal_pose_checkpoint) if args.head_temporal_pose_checkpoint else None),
        head_temporal_keypoint_threshold=args.head_temporal_keypoint_threshold,
        head_temporal_max_frame_gap=args.head_temporal_max_frame_gap,
        hand_temporal=args.hand_temporal,
        hand_temporal_pose_config=(Path(args.hand_temporal_pose_config) if args.hand_temporal_pose_config else None),
        hand_temporal_pose_checkpoint=(Path(args.hand_temporal_pose_checkpoint) if args.hand_temporal_pose_checkpoint else None),
        hand_temporal_keypoint_threshold=args.hand_temporal_keypoint_threshold,
        hand_temporal_max_frame_gap=args.hand_temporal_max_frame_gap,
        hand_temporal_wrist_threshold=args.hand_temporal_wrist_threshold,
        hand_temporal_max_wrist_distance=args.hand_temporal_max_wrist_distance,
        human_temporal_3d=args.human_temporal_3d,
        gvhmr_root=Path(args.gvhmr_root) if args.gvhmr_root else None,
        gvhmr_checkpoint=(
            Path(args.gvhmr_checkpoint) if args.gvhmr_checkpoint else None
        ),
        human_temporal_3d_min_valid_joints=(
            args.human_temporal_3d_min_valid_joints
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    if args.tie_epsilon < 0:
        parser.error("--tie-epsilon must be non-negative")
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output must be a directory path: {output_dir}")
    if args.selection_manifest:
        input_root, pairs = load_selected_video_pairs(
            Path(args.selection_manifest)
        )
    else:
        input_root = Path(args.input_dir).expanduser().resolve()
        pairs = discover_video_pairs(input_root)
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
    aggregate = build_paired_result(input_root, pairs, results)
    if args.selection_manifest:
        aggregate["selection_manifest"] = str(
            Path(args.selection_manifest).expanduser().resolve()
        )
    scores = build_scores_result(aggregate)
    frame_evaluation = build_pair_frame_evaluation(
        aggregate, tie_epsilon=args.tie_epsilon
    )
    frame_evaluation["selection_manifest"] = (
        str(Path(args.selection_manifest).expanduser().resolve())
        if args.selection_manifest else None
    )
    pair_evaluation = build_pair_score_summary(frame_evaluation)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_output = output_dir / FULL_RESULT_FILENAME
    scores_output = output_dir / SCORES_RESULT_FILENAME
    frame_evaluation_output = output_dir / FRAME_EVALUATION_FILENAME
    pair_evaluation_output = output_dir / PAIR_EVALUATION_FILENAME
    write_json_atomic(full_output, aggregate)
    write_json_atomic(scores_output, scores)
    write_json_atomic(frame_evaluation_output, frame_evaluation)
    write_json_atomic(pair_evaluation_output, pair_evaluation)
    valid = sum(
        result.get("valid") is True
        for pair in aggregate["pairs"]
        for result in (pair["positive"]["result"], pair["negative"]["result"])
    )
    LOGGER.info(
        "Completed %d/%d valid videos; full=%s; scores=%s; evaluation=%s",
        valid,
        len(video_paths),
        full_output,
        scores_output,
        pair_evaluation_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
