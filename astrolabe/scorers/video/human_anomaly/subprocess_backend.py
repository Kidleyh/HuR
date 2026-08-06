"""Isolated invocation of the HuR-owned worker in the VBench Conda environment."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class BackendResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    runtime_sec: float


class VBenchWorkerError(RuntimeError):
    def __init__(self, result: BackendResult):
        super().__init__(
            f"VBench worker failed with exit code {result.returncode}: {result.stderr.strip()}"
        )
        self.result = result


def _cuda_visible_devices(device: str) -> str:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    return ""


def _worker_device(device: str) -> str:
    """Translate a host GPU selection into the worker's remapped device."""
    if device == "cuda" or device.startswith("cuda:"):
        return "cuda:0"
    return device


def build_worker_command(
    *,
    conda_env: str,
    worker_script: Path,
    video: Path,
    input_manifest: Path,
    output_jsonl: Path,
    runtime_info_json: Path,
    vbench_root: Path,
    cache_dir: Path,
    clip_model: Path,
    hur_root: Path,
    device: str,
    crop_batch_size: int,
) -> List[str]:
    pythonpath = os.pathsep.join(
        [str(hur_root.resolve()), str(vbench_root.resolve()), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return [
        "conda", "run", "-n", conda_env, "env",
        f"VBENCH2_CACHE_DIR={cache_dir.resolve()}",
        f"VBENCH2_CLIP_TEXT_MODEL={clip_model.resolve()}",
        f"HF_HOME={(cache_dir.resolve().parent / 'huggingface')}",
        f"CUDA_VISIBLE_DEVICES={_cuda_visible_devices(device)}",
        f"PYTHONPATH={pythonpath}",
        "python", str(worker_script.resolve()),
        "--video", str(video.resolve()),
        "--input-manifest", str(input_manifest.resolve()),
        "--output-jsonl", str(output_jsonl.resolve()),
        "--runtime-info-json", str(runtime_info_json.resolve()),
        "--vbench-root", str(vbench_root.resolve()),
        "--device", _worker_device(device),
        "--crop-batch-size", str(crop_batch_size),
    ]


def run_vbench_worker(
    command: Sequence[str], stdout_path: Path, stderr_path: Path,
    working_directory: Optional[Path] = None,
    runner=subprocess.run,
) -> BackendResult:
    """Run without a shell, persist complete logs, and fail on nonzero status."""
    started = time.perf_counter()
    completed = runner(
        list(command), capture_output=True, text=True, check=False, shell=False,
        cwd=str(working_directory.resolve()) if working_directory is not None else None,
    )
    result = BackendResult(
        command=list(command), returncode=int(completed.returncode),
        stdout=completed.stdout or "", stderr=completed.stderr or "",
        runtime_sec=time.perf_counter() - started,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise VBenchWorkerError(result)
    return result
