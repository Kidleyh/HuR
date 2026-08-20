#!/usr/bin/env python3
"""Run the single-process, in-memory HuR Human Reward pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrolabe.scorers.video.human_reward import HumanRewardConfig, HumanRewardModel


def build_parser() -> argparse.ArgumentParser:
    defaults = HumanRewardConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True)
    parser.add_argument("--output")
    parser.add_argument("--visualization-output")
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


def _write_atomic(path: Path, result: Any) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.video) > 1 and args.visualization_output:
        parser.error(
            "--visualization-output supports single-video mode only; "
            "use one --video"
        )
    config = HumanRewardConfig(
        yolo_weights=Path(args.weights), tracker_config=Path(args.tracker_config),
        stitching_config=Path(args.stitching_config),
        vbench_root=Path(args.vbench_root),
        vbench_cache_dir=Path(args.vbench_cache_dir), device=args.device,
        vbench_clip_model=Path(args.vbench_clip_model),
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, half=args.half,
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
    model = HumanRewardModel(config)
    result = (
        model.score(
            args.video[0], visualization_output=args.visualization_output
        )
        if len(args.video) == 1
        else model.score_batch(args.video)
    )
    if args.output:
        _write_atomic(Path(args.output), result)
    else:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
