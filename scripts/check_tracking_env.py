#!/usr/bin/env python3
"""Report whether the isolated person-tracking runtime is ready."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("check_tracking_env")


def _version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        return f"NOT AVAILABLE ({type(error).__name__}: {error})"
    return str(getattr(module, "__version__", "unknown"))


def _find_weights() -> Optional[Path]:
    candidates = []
    if os.environ.get("YOLO_WEIGHTS"):
        candidates.append(Path(os.environ["YOLO_WEIGHTS"]).expanduser())
    if os.environ.get("GVHMR_ROOT"):
        candidates.append(
            Path(os.environ["GVHMR_ROOT"]).expanduser() / "inputs/checkpoints/yolo/yolov8x.pt"
        )
    candidates.append(PROJECT_ROOT / "checkpoints/yolo/yolov8x.pt")
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def main() -> int:
    LOGGER.info("Python version: %s", platform.python_version())
    LOGGER.info("Python executable: %s", sys.executable)
    LOGGER.info("PyTorch version: %s", _version("torch"))
    LOGGER.info("torchvision version: %s", _version("torchvision"))
    LOGGER.info("Ultralytics version: %s", _version("ultralytics"))
    LOGGER.info("OpenCV version: %s", _version("cv2"))
    LOGGER.info("NumPy version: %s", _version("numpy"))
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        LOGGER.info("CUDA available: %s", cuda_available)
        if cuda_available:
            device = torch.cuda.current_device()
            LOGGER.info("Current CUDA device: %s", device)
            LOGGER.info("GPU name: %s", torch.cuda.get_device_name(device))
        else:
            LOGGER.warning("CUDA is unavailable; tracking may still run with --device cpu.")
            LOGGER.info("Current CUDA device: none")
            LOGGER.info("GPU name: none")
    except Exception as error:
        LOGGER.warning("CUDA inspection failed: %s", error)
    try:
        importlib.import_module("lap")
        LOGGER.info("lap module importable: yes")
    except Exception as error:
        LOGGER.error("lap module importable: no (%s)", error)
    weights = _find_weights()
    LOGGER.info("YOLO weights exist: %s", "yes" if weights else "no")
    LOGGER.info("YOLO weights path: %s", weights or "not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
