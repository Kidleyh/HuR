#!/usr/bin/env python3
"""Run the single-process, in-memory HuR Human Reward pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrolabe.scorers.video.human_reward import HumanRewardConfig, HumanRewardModel


def build_parser() -> argparse.ArgumentParser:
    defaults = HumanRewardConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output")
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
    return parser


def _write_atomic(path: Path, result: dict) -> None:
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
    args = build_parser().parse_args(argv)
    config = HumanRewardConfig(
        yolo_weights=Path(args.weights), tracker_config=Path(args.tracker_config),
        stitching_config=Path(args.stitching_config),
        vbench_root=Path(args.vbench_root),
        vbench_cache_dir=Path(args.vbench_cache_dir), device=args.device,
        vbench_clip_model=Path(args.vbench_clip_model),
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, half=args.half,
        crop_batch_size=args.crop_batch_size,
    )
    result = HumanRewardModel(config).score(args.video)
    if args.output:
        _write_atomic(Path(args.output), result)
    else:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
