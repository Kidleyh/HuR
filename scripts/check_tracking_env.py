#!/usr/bin/env python3
"""Report whether the isolated person-tracking runtime is ready."""

from __future__ import annotations

import importlib
import argparse
import logging
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("check_tracking_env")


def _version(module_name: str) -> Tuple[str, bool]:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        return f"NOT AVAILABLE ({type(error).__name__}: {error})", False
    return str(getattr(module, "__version__", "unknown")), True


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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    ready = True
    LOGGER.info("Python version: %s", platform.python_version())
    LOGGER.info("Python executable: %s", sys.executable)
    modules = [
        ("PyTorch", "torch"),
        ("torchvision", "torchvision"),
        ("Ultralytics", "ultralytics"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("PyYAML", "yaml"),
        ("lap", "lap"),
    ]
    for label, module_name in modules:
        version, available = _version(module_name)
        LOGGER.info("%s version: %s", label, version)
        ready = ready and available

    cuda_available = False
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
        LOGGER.error("CUDA inspection failed: %s", error)
        ready = False
    if args.require_cuda and not cuda_available:
        LOGGER.error("CUDA is required but unavailable.")
        ready = False

    weights = _find_weights()
    LOGGER.info("YOLO weights exist: %s", "yes" if weights else "no")
    LOGGER.info("YOLO weights path: %s", weights or "not found")
    if weights is None:
        if args.require_weights:
            LOGGER.error("YOLO weights are required but were not found.")
            ready = False
        else:
            LOGGER.warning("YOLO weights were not found; real inference is unavailable.")
    LOGGER.info("Environment status: %s", "READY" if ready else "NOT READY")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
