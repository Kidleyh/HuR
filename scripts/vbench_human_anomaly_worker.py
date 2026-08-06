#!/usr/bin/env python3
"""CLI wrapper for the in-process HuR HumanAnomalyEngine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

HUR_ROOT = Path(__file__).resolve().parents[1]
if str(HUR_ROOT) not in sys.path:
    sys.path.insert(0, str(HUR_ROOT))

from astrolabe.scorers.video.human_anomaly.engine import (
    FaceHandDetector, HumanAnomalyEngine, _is_cuda_failure, _part_result,
    _process_person,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--runtime-info-json", required=True)
    parser.add_argument("--vbench-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-batch-size", type=int, default=128)
    return parser


def _read_entries(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = Path(os.environ["VBENCH2_CACHE_DIR"]).expanduser().resolve()
    engine = HumanAnomalyEngine(
        vbench_root=args.vbench_root,
        cache_dir=cache_dir,
        device=args.device,
        crop_batch_size=args.crop_batch_size,
    )
    try:
        entries = _read_entries(Path(args.input_manifest).expanduser().resolve())
        results = engine.score_video(args.video, entries)
        runtime = engine.runtime_info()
    finally:
        engine.close()

    output = Path(args.output_jsonl).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    runtime_path = Path(args.runtime_info_json).expanduser().resolve()
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
