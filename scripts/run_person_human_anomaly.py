#!/usr/bin/env python3
"""Analyze every HuR logical-track person frame with VBench Human Anomaly."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrolabe.scorers.video.human_anomaly.aggregation import aggregate_human_anomaly
from astrolabe.scorers.video.human_anomaly.manifest import (
    build_human_anomaly_manifest,
    write_input_manifest,
)
from astrolabe.scorers.video.human_anomaly.schema import OFFICIAL_THRESHOLDS
from astrolabe.scorers.video.human_anomaly.subprocess_backend import (
    VBenchWorkerError,
    build_worker_command,
    run_vbench_worker,
)
from astrolabe.scorers.video.human_anomaly.visualization import (
    write_anomaly_visualization,
)

LOGGER = logging.getLogger("run_person_human_anomaly")
CORE_OUTPUTS = (
    "human_anomaly_input.jsonl", "human_anomaly_frames.jsonl",
    "human_anomaly_tracks.json", "human_anomaly_summary.json", "run_manifest.json",
    "worker_stdout.log", "worker_stderr.log",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--stitching-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vbench-root", required=True)
    parser.add_argument("--vbench-cache-dir", required=True)
    parser.add_argument("--vbench-clip-model", required=True)
    parser.add_argument("--vbench-conda-env", default="vbench2-human-anomaly")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-batch-size", type=int, default=128)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_commit(root: Path) -> Optional[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _path_info(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _is_complete(output: Path) -> bool:
    try:
        return all((output / name).is_file() and (output / name).stat().st_size >= 0 for name in CORE_OUTPUTS)
    except OSError:
        return False


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _promote(staging: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    generated = set(CORE_OUTPUTS) | {"human_anomaly.mp4"}
    for name in generated:
        source, target = staging / name, output / name
        if source.exists():
            os.replace(source, target)
        elif target.exists():
            target.unlink()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    video = Path(args.video).expanduser().resolve()
    stitching = Path(args.stitching_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    vbench_root = Path(args.vbench_root).expanduser().resolve()
    cache_dir = Path(args.vbench_cache_dir).expanduser().resolve()
    clip_model = Path(args.vbench_clip_model).expanduser().resolve()
    if args.crop_batch_size <= 0:
        LOGGER.error("--crop-batch-size must be positive")
        return 2
    for path, label in ((video, "video"), (stitching, "stitching-dir"),
                        (vbench_root, "vbench-root"), (cache_dir, "vbench-cache-dir"),
                        (clip_model, "vbench-clip-model")):
        if not path.exists():
            LOGGER.error("%s does not exist: %s", label, path)
            return 2
    if _is_complete(output) and not args.overwrite:
        LOGGER.info("Skipping complete Human Anomaly output: %s", output)
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.tmp-") as name:
        stage = Path(name)
        run_manifest: Dict[str, Any] = {
            "status": "running", "input_video": str(video),
            "stitching_input": str(stitching), "hur_commit": _git_commit(PROJECT_ROOT),
            "vbench_root": str(vbench_root), "vbench_commit": _git_commit(vbench_root),
            "vbench_cache_dir": str(cache_dir), "clip_model": _path_info(clip_model),
            "device": args.device, "thresholds": OFFICIAL_THRESHOLDS,
            "model_weights": [
                _path_info(cache_dir / "YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"),
                *[
                    _path_info(cache_dir / f"anomaly_detector/{category}.pth")
                    for category in ("human", "face", "hand")
                ],
            ],
        }
        try:
            entries, manifest_failures, width, height = build_human_anomaly_manifest(video, stitching)
            input_manifest = stage / "human_anomaly_input.jsonl"
            write_input_manifest(entries, input_manifest)
            worker_output = stage / "human_anomaly_frames.jsonl"
            runtime_info_path = stage / "worker_runtime.json"
            command = build_worker_command(
                conda_env=args.vbench_conda_env,
                worker_script=PROJECT_ROOT / "scripts/vbench_human_anomaly_worker.py",
                video=video, input_manifest=input_manifest, output_jsonl=worker_output,
                runtime_info_json=runtime_info_path, vbench_root=vbench_root,
                cache_dir=cache_dir, clip_model=clip_model, hur_root=PROJECT_ROOT,
                device=args.device, crop_batch_size=args.crop_batch_size,
            )
            run_manifest["command"] = command
            backend = run_vbench_worker(
                command, stage / "worker_stdout.log", stage / "worker_stderr.log",
                working_directory=vbench_root,
            )
            results = _jsonl(worker_output)
            tracks, summary = aggregate_human_anomaly(entries, results, width, height)
            summary["manifest_failures"] = [item.to_dict() for item in manifest_failures]
            _write_json(stage / "human_anomaly_tracks.json", tracks)
            _write_json(stage / "human_anomaly_summary.json", summary)
            runtime_info = json.loads(runtime_info_path.read_text(encoding="utf-8"))
            weight_paths = [runtime_info["detector_weight"]] + [
                config["weight_path"] for config in runtime_info["classifier_models"].values()
            ]
            run_manifest.update({
                "status": "success", "runtime_sec": time.perf_counter() - started,
                "worker_runtime_sec": backend.runtime_sec,
                "worker_exit_code": backend.returncode,
                "torch_version": runtime_info["torch_version"],
                "mmcv_version": runtime_info["mmcv_version"],
                "model_weights": [_path_info(Path(path)) for path in weight_paths],
                "input_person_frames": len(entries),
                "invalid_or_duplicate_inputs": len(manifest_failures),
            })
            if args.visualize:
                write_anomaly_visualization(video, results, stage / "human_anomaly.mp4")
            _write_json(stage / "run_manifest.json", run_manifest)
            _promote(stage, output)
            LOGGER.info(
                "Human Anomaly completed: tracks=%d observed=%d scored=%d abnormal=%d output=%s",
                summary["logical_track_count"], summary["observed_person_frames"],
                summary["scored_person_frames"], summary["abnormal_person_frames"], output,
            )
            return 0
        except Exception as error:
            run_manifest.update({
                "status": "failed", "runtime_sec": time.perf_counter() - started,
                "error_type": type(error).__name__, "message": str(error),
            })
            if isinstance(error, VBenchWorkerError):
                run_manifest.update({
                    "worker_exit_code": error.result.returncode,
                    "worker_runtime_sec": error.result.runtime_sec,
                })
            output.mkdir(parents=True, exist_ok=True)
            for log_name in ("worker_stdout.log", "worker_stderr.log"):
                if (stage / log_name).exists():
                    shutil.copy2(stage / log_name, output / log_name)
            _write_json(output / "run_manifest.json", run_manifest)
            LOGGER.exception("Human Anomaly processing failed")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
