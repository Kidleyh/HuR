#!/usr/bin/env python3
"""Filter Guangdian V2 samples by small-clear faces in the video first frame."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.filter_small_clear_faces import (
    DEFAULT_VBENCH_ROOT,
    FaceHandDetector,
    FilterThresholds,
    _environment_variable,
    _model_paths,
    _working_directory,
    analyze_frame,
    write_json_atomic,
)

LOGGER = logging.getLogger("filter_guangdian_small_clear_faces")


def iter_dataset(
    input_root: Path,
) -> Iterator[Tuple[Optional[Tuple[Path, Path, Path]], Optional[Dict[str, Any]]]]:
    """Stream paired triples or skipped entries without materializing the dataset."""
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    children = sorted((item for item in root.iterdir() if item.is_dir()), key=str)
    for folder in children:
        version = folder / "v2.0.0"
        if not version.is_dir():
            yield None, {
                "folder": str(folder), "reason": "version_directory_missing",
                "message": f"Missing directory: {version}",
            }
            continue
        caption_dir = version / "caption_v2"
        if not caption_dir.is_dir():
            yield None, {
                "folder": str(folder), "reason": "caption_directory_missing",
                "message": f"Missing directory: {caption_dir}",
            }
            continue
        caption_paths = sorted(caption_dir.glob("*.json"), key=lambda item: item.name)
        if not caption_paths:
            yield None, {
                "folder": str(folder), "reason": "caption_directory_empty",
                "message": f"No JSON files in: {caption_dir}",
            }
            continue
        for caption_path in caption_paths:
            video_path = version / "video" / "data" / f"{caption_path.stem}.mp4"
            label_path = version / "label" / f"{caption_path.stem}.json"
            missing = []
            if not video_path.is_file():
                missing.append("video")
            if not label_path.is_file():
                missing.append("label")
            if missing:
                yield None, {
                    "folder": str(folder),
                    "video_caption_path": str(caption_path),
                    "file_path": str(video_path),
                    "label_path": str(label_path),
                    "reason": "paired_file_missing",
                    "missing": missing,
                }
                continue
            yield (caption_path, video_path, label_path), None


def discover_dataset(
    input_root: Path,
) -> Tuple[List[Tuple[Path, Path, Path]], List[Dict[str, Any]]]:
    """Materialize discovery for tests and small callers; main uses streaming."""
    triples: List[Tuple[Path, Path, Path]] = []
    skipped: List[Dict[str, Any]] = []
    for triple, skipped_entry in iter_dataset(input_root):
        if triple is not None:
            triples.append(triple)
        if skipped_entry is not None:
            skipped.append(skipped_entry)
    return triples, skipped


def analyze_video_first_frame(
    video_path: Path,
    detector: Any,
    thresholds: FilterThresholds,
    detection_options: Dict[str, Any],
) -> Dict[str, Any]:
    """Decode and analyze exactly the first video frame."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Video cannot be opened: {video_path}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Video first frame cannot be decoded: {video_path}")
    result = analyze_frame(frame, 0, detector, thresholds, **detection_options)
    result["width"] = int(frame.shape[1])
    result["height"] = int(frame.shape[0])
    result["selected"] = any(
        face.get("qualified") is True for face in result.get("faces", [])
    )
    return result


def selected_manifest_entry(
    caption_path: Path, video_path: Path, label_path: Path
) -> Dict[str, Any]:
    """Build the exact downstream manifest record requested by the dataset."""
    return {
        "video_caption_path": str(caption_path.resolve()),
        "file_path": str(video_path.resolve()),
        "label_path": str(label_path.resolve()),
        "lipsync": {},
    }


def default_skipped_output(output_path: Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    return output.with_name(f"{output.stem}_skipped{output.suffix or '.json'}")


def _load_existing(
    output_path: Path, skipped_path: Path, resume: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set[str]]:
    if not resume:
        if output_path.exists() or skipped_path.exists():
            raise FileExistsError(
                "Output already exists; pass --resume to continue or choose a new path"
            )
        return [], [], set()
    selected = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else []
    skipped_payload = (
        json.loads(skipped_path.read_text(encoding="utf-8"))
        if skipped_path.is_file() else {"entries": []}
    )
    if not isinstance(selected, list) or not isinstance(skipped_payload, dict):
        raise ValueError("Existing resume outputs have an invalid schema")
    skipped = skipped_payload.get("entries", [])
    if not isinstance(skipped, list):
        raise ValueError("Existing skipped output entries must be a list")
    processed = {
        str(item["video_caption_path"])
        for item in [*selected, *skipped]
        if isinstance(item, dict) and item.get("video_caption_path")
    }
    return selected, skipped, processed


def _write_outputs(
    output_path: Path,
    skipped_path: Path,
    selected: Sequence[Dict[str, Any]],
    skipped: Sequence[Dict[str, Any]],
    input_root: Path,
    paired_candidates_seen: int,
) -> None:
    write_json_atomic(output_path, list(selected))
    write_json_atomic(skipped_path, {
        "schema_version": "1.0",
        "input_root": str(input_root),
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "paired_candidates_seen": paired_candidates_seen,
        "entries": list(skipped),
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--skipped-output-path",
        help="Defaults to <output stem>_skipped.json next to --output-path",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vbench-root", default=str(DEFAULT_VBENCH_ROOT))
    parser.add_argument("--vbench-cache-dir")
    parser.add_argument("--vbench-clip-model")
    parser.add_argument("--detector-config")
    parser.add_argument("--detector-checkpoint")
    parser.add_argument("--detector-threshold", type=float, default=0.10)
    parser.add_argument(
        "--tiled-inference", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.15)
    parser.add_argument(
        "--whole-image-detection", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--nms-iou-threshold", type=float, default=0.50)
    parser.add_argument("--area-threshold", type=float, default=0.01)
    parser.add_argument("--min-face-short-side", type=float, default=32.0)
    parser.add_argument("--laplacian-threshold", type=float, default=100.0)
    parser.add_argument("--tenengrad-threshold", type=float, default=1000.0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be positive")
    input_root = Path(args.input_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    skipped_path = (
        Path(args.skipped_output_path).expanduser().resolve()
        if args.skipped_output_path else default_skipped_output(output_path)
    )
    try:
        thresholds = FilterThresholds(
            area_threshold=args.area_threshold,
            min_face_short_side=args.min_face_short_side,
            laplacian_threshold=args.laplacian_threshold,
            tenengrad_threshold=args.tenengrad_threshold,
        )
        if not input_root.is_dir():
            raise FileNotFoundError(f"Input root does not exist: {input_root}")
        selected, skipped, processed = _load_existing(
            output_path, skipped_path, args.resume
        )
        def discovery_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
            caption = item.get("video_caption_path")
            return (
                ("caption", caption, item.get("reason")) if caption else
                ("folder", item.get("folder"), item.get("reason"))
            )

        existing_discovery_keys = {
            discovery_key(item) for item in skipped if isinstance(item, dict)
        }
        root, config, checkpoint, clip_model = _model_paths(args)
        for model_path in (config, checkpoint):
            if not model_path.is_file():
                raise FileNotFoundError(f"YOLO-World resource is missing: {model_path}")
        if not clip_model.is_dir():
            raise FileNotFoundError(f"Local CLIP text model is missing: {clip_model}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    paired_candidates_seen = 0
    _write_outputs(output_path, skipped_path, selected, skipped, input_root, 0)

    yolo_world_root = root / "vbench2/third_party/YOLO-World"
    if str(yolo_world_root) not in sys.path:
        sys.path.insert(0, str(yolo_world_root))
    with ExitStack() as stack:
        stack.enter_context(_environment_variable("VBENCH2_CLIP_TEXT_MODEL", str(clip_model)))
        stack.enter_context(_environment_variable("HF_HUB_OFFLINE", "1"))
        stack.enter_context(_environment_variable("TRANSFORMERS_OFFLINE", "1"))
        stack.enter_context(_working_directory(root))
        detector = FaceHandDetector(config, checkpoint, args.device)

    detection_options = {
        "detector_threshold": args.detector_threshold,
        "tiled_inference": args.tiled_inference,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "whole_image_detection": args.whole_image_detection,
        "nms_iou_threshold": args.nms_iou_threshold,
    }
    failures = 0
    processed_this_run = 0
    for triple, discovery_skip in iter_dataset(input_root):
        if discovery_skip is not None:
            key = discovery_key(discovery_skip)
            if key not in existing_discovery_keys:
                skipped.append(discovery_skip)
                existing_discovery_keys.add(key)
            continue
        assert triple is not None
        caption_path, video_path, label_path = triple
        paired_candidates_seen += 1
        if str(caption_path) in processed:
            continue
        if (
            args.max_videos is not None
            and processed_this_run >= args.max_videos
        ):
            break
        processed_this_run += 1
        LOGGER.info("Processing %d: %s", processed_this_run, video_path)
        try:
            frame_result = analyze_video_first_frame(
                video_path, detector, thresholds, detection_options
            )
            if frame_result["selected"]:
                selected.append(selected_manifest_entry(
                    caption_path, video_path, label_path
                ))
            else:
                skipped.append({
                    "video_caption_path": str(caption_path),
                    "file_path": str(video_path),
                    "label_path": str(label_path),
                    "reason": "first_frame_not_qualified",
                    "first_frame": frame_result,
                })
        except Exception as exc:
            failures += 1
            LOGGER.exception("Failed to process %s", video_path)
            skipped.append({
                "video_caption_path": str(caption_path),
                "file_path": str(video_path),
                "label_path": str(label_path),
                "reason": "processing_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
        _write_outputs(
            output_path, skipped_path, selected, skipped, input_root,
            paired_candidates_seen,
        )
        LOGGER.info("Checkpointed selected=%d skipped=%d", len(selected), len(skipped))
    _write_outputs(
        output_path, skipped_path, selected, skipped, input_root,
        paired_candidates_seen,
    )
    LOGGER.info(
        "Completed processed=%d selected=%d skipped=%d failures=%d output=%s",
        processed_this_run, len(selected), len(skipped), failures, output_path,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
