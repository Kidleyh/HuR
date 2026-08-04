"""Environment-check strictness tests for CPU CI without model weights."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/check_tracking_env.py"


def run_check(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_environment_check_allows_missing_optional_runtime_resources() -> None:
    completed = run_check()
    assert completed.returncode == 0
    assert "Environment status: READY" in completed.stderr


def test_require_weights_fails_when_weights_are_missing(monkeypatch) -> None:
    monkeypatch.delenv("YOLO_WEIGHTS", raising=False)
    monkeypatch.delenv("GVHMR_ROOT", raising=False)
    if (PROJECT_ROOT / "checkpoints/yolo/yolov8x.pt").is_file():
        pytest.skip("project-local YOLO weights are installed")
    completed = run_check("--require-weights")
    assert completed.returncode == 1
    assert "Environment status: NOT READY" in completed.stderr


def test_require_cuda_fails_on_cpu_runner() -> None:
    completed = run_check("--require-cuda")
    expected = 0 if torch.cuda.is_available() else 1
    assert completed.returncode == expected
    expected_status = "READY" if expected == 0 else "NOT READY"
    assert f"Environment status: {expected_status}" in completed.stderr
