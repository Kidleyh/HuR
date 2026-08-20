#!/usr/bin/env python3
"""Filter images/videos containing small yet locally sharp detected faces."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import imageio_ffmpeg
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrolabe.scorers.video.human_anomaly.engine import FaceHandDetector

LOGGER = logging.getLogger("filter_small_clear_faces")
DEFAULT_VBENCH_ROOT = Path(os.environ.get(
    "VBENCH_ROOT",
    "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0",
))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
QUALIFIED_COLOR = (0, 200, 0)
UNQUALIFIED_COLOR = (0, 215, 255)


@dataclass(frozen=True)
class FilterThresholds:
    """Direct, deliberately untuned V1 face-selection thresholds."""

    area_threshold: float = 0.01
    min_face_short_side: float = 32.0
    laplacian_threshold: float = 100.0
    tenengrad_threshold: float = 1000.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if not 0.0 < self.area_threshold <= 1.0:
            raise ValueError("area_threshold must be in (0, 1]")
        for name in (
            "min_face_short_side", "laplacian_threshold", "tenengrad_threshold"
        ):
            value = values[name]
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _environment_variable(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def tile_origins(length: int, tile_size: int, overlap: float) -> List[int]:
    """Return deterministic tile starts and always cover the trailing edge."""
    if length <= 0:
        raise ValueError("image dimension must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("tile_overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    origins = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if origins[-1] != final:
        origins.append(final)
    return origins


def iter_tiles(
    image: np.ndarray, tile_size: int, overlap: float
) -> Iterable[Tuple[int, int, np.ndarray]]:
    """Yield ``(x_offset, y_offset, original-resolution tile)`` tuples."""
    height, width = image.shape[:2]
    for y in tile_origins(height, tile_size, overlap):
        for x in tile_origins(width, tile_size, overlap):
            yield x, y, image[y:min(y + tile_size, height), x:min(x + tile_size, width)]


def map_detection_to_image(
    detection: Dict[str, Any],
    x_offset: int,
    y_offset: int,
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, Any]]:
    """Map one tile-local face detection into clipped image coordinates."""
    if int(detection.get("label", -1)) != 0:
        return None
    values = detection.get("bbox_xyxy")
    if not isinstance(values, Sequence) or len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in values]
        score = float(detection["detector_score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)):
        return None
    x1 = min(max(x1 + x_offset, 0.0), float(image_width))
    y1 = min(max(y1 + y_offset, 0.0), float(image_height))
    x2 = min(max(x2 + x_offset, 0.0), float(image_width))
    y2 = min(max(y2 + y_offset, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "detector_score": score,
    }


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def non_maximum_suppression(
    detections: Sequence[Dict[str, Any]], iou_threshold: float
) -> List[Dict[str, Any]]:
    """Apply deterministic score-first NMS to globally mapped face boxes."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be in [0, 1]")
    ordered = sorted(
        enumerate(detections),
        key=lambda item: (-float(item[1]["detector_score"]), item[0]),
    )
    kept: List[Dict[str, Any]] = []
    for _, detection in ordered:
        if all(
            bbox_iou(detection["bbox_xyxy"], item["bbox_xyxy"]) <= iou_threshold
            for item in kept
        ):
            kept.append(dict(detection))
    return kept


def detect_faces(
    image: np.ndarray,
    detector: Any,
    *,
    detector_threshold: float,
    tiled_inference: bool,
    tile_size: int,
    tile_overlap: float,
    whole_image_detection: bool,
    nms_iou_threshold: float,
) -> List[Dict[str, Any]]:
    """Detect faces over full image and/or tiles, then globally deduplicate."""
    if not tiled_inference and not whole_image_detection:
        raise ValueError("At least one detection mode must be enabled")
    if not 0.0 <= detector_threshold <= 1.0:
        raise ValueError("detector_threshold must be in [0, 1]")
    height, width = image.shape[:2]
    mapped: List[Dict[str, Any]] = []

    def add(source: np.ndarray, x_offset: int, y_offset: int) -> None:
        for detection in detector.detect(source, score_threshold=detector_threshold):
            result = map_detection_to_image(
                detection, x_offset, y_offset, width, height
            )
            if result is not None:
                mapped.append(result)

    if whole_image_detection:
        add(image, 0, 0)
    if tiled_inference:
        for x_offset, y_offset, tile in iter_tiles(image, tile_size, tile_overlap):
            add(tile, x_offset, y_offset)
    return non_maximum_suppression(mapped, nms_iou_threshold)


def clarity_metrics(face_crop: np.ndarray) -> Tuple[float, float]:
    """Compute Laplacian variance and mean Sobel gradient energy, without resize."""
    if face_crop.size == 0:
        raise ValueError("face crop is empty")
    gray = (
        cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        if face_crop.ndim == 3 else face_crop
    )
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y))
    return laplacian_variance, tenengrad


def measure_face(
    image: np.ndarray,
    detection: Dict[str, Any],
    thresholds: FilterThresholds,
) -> Dict[str, Any]:
    """Measure one mapped detection using its original-resolution image crop."""
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
    crop_x1 = max(0, int(math.floor(x1)))
    crop_y1 = max(0, int(math.floor(y1)))
    crop_x2 = min(image_width, int(math.ceil(x2)))
    crop_y2 = min(image_height, int(math.ceil(y2)))
    width = x2 - x1
    height = y2 - y1
    short_side = min(width, height)
    area_ratio = (width * height) / float(image_width * image_height)
    laplacian, tenengrad = clarity_metrics(
        image[crop_y1:crop_y2, crop_x1:crop_x2]
    )
    qualified = (
        area_ratio < thresholds.area_threshold
        and short_side >= thresholds.min_face_short_side
        and laplacian >= thresholds.laplacian_threshold
        and tenengrad >= thresholds.tenengrad_threshold
    )
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "detector_score": float(detection["detector_score"]),
        "width_px": width,
        "height_px": height,
        "short_side_px": short_side,
        "area_ratio": area_ratio,
        "laplacian_variance": laplacian,
        "tenengrad": tenengrad,
        "qualified": qualified,
    }


def analyze_frame(
    image: np.ndarray,
    frame_index: int,
    detector: Any,
    thresholds: FilterThresholds,
    **detection_options: Any,
) -> Dict[str, Any]:
    detections = detect_faces(image, detector, **detection_options)
    faces = [measure_face(image, item, thresholds) for item in detections]
    return {
        "frame_index": int(frame_index),
        "face_count": len(faces),
        "qualified_face_count": sum(face["qualified"] for face in faces),
        "faces": faces,
    }


def build_media_result(
    path: Path,
    media_type: str,
    width: int,
    height: int,
    frames: Sequence[Dict[str, Any]],
    min_qualified_frames: int,
) -> Dict[str, Any]:
    qualified_frames = sum(
        any(face.get("qualified") is True for face in frame.get("faces", []))
        for frame in frames
    )
    required = 1 if media_type == "image" else min_qualified_frames
    return {
        "path": str(Path(path).expanduser().resolve()),
        "media_type": media_type,
        "width": int(width),
        "height": int(height),
        "selected": qualified_frames >= required,
        "qualified_frame_count": qualified_frames,
        "sampled_frame_count": len(frames),
        "frames": list(frames),
    }


def process_image(
    path: Path,
    detector: Any,
    thresholds: FilterThresholds,
    detection_options: Dict[str, Any],
    min_qualified_frames: int,
) -> Dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Image cannot be decoded: {path}")
    height, width = image.shape[:2]
    frame = analyze_frame(
        image, 0, detector, thresholds, **detection_options
    )
    return build_media_result(
        path, "image", width, height, [frame], min_qualified_frames
    )


def process_video(
    path: Path,
    detector: Any,
    thresholds: FilterThresholds,
    detection_options: Dict[str, Any],
    frame_stride: int,
    min_qualified_frames: int,
) -> Dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Video cannot be opened: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid dimensions: {path}")
    frames = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                frames.append(analyze_frame(
                    frame, frame_index, detector, thresholds, **detection_options
                ))
            frame_index += 1
    finally:
        capture.release()
    if frame_index == 0:
        raise RuntimeError(f"Video contains no decodable frames: {path}")
    result = build_media_result(
        path, "video", width, height, frames, min_qualified_frames
    )
    result["decoded_frame_count"] = frame_index
    result["frame_stride"] = frame_stride
    return result


def discover_media(input_path: Path, recursive: bool) -> Tuple[Path, List[Path]]:
    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported media extension: {source}")
        return source.parent, [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    iterator = source.rglob("*") if recursive else source.iterdir()
    media = sorted(
        (
            path.resolve() for path in iterator
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(source)),
    )
    if not media:
        raise ValueError(f"Input directory contains no supported media: {source}")
    return source, media


def copy_selected(
    path: Path,
    input_root: Path,
    destination_root: Path,
    overwrite: bool,
) -> Path:
    relative = path.name if input_root == path.parent else path.relative_to(input_root)
    destination = destination_root.expanduser().resolve() / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Selected-media destination already exists: {destination}"
        )
    shutil.copy2(path, destination)
    return destination


def visualization_path_for_media(
    path: Path,
    input_root: Path,
    visualization_root: Path,
    media_type: str,
) -> Path:
    """Return a collision-free visualization path preserving input layout."""
    source = Path(path).expanduser().resolve()
    root = Path(input_root).expanduser().resolve()
    relative = source.relative_to(root)
    destination = Path(visualization_root).expanduser().resolve() / relative
    return destination.with_suffix(".mp4") if media_type == "video" else destination


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, 1, cv2.LINE_AA,
    )


def draw_face_visualization(
    image: np.ndarray, frame_result: Dict[str, Any]
) -> np.ndarray:
    """Draw already-computed face measurements without running detection again."""
    rendered = image.copy()
    image_height, image_width = rendered.shape[:2]
    scale = max(0.35, min(0.65, image_width / 1600.0))
    for face_index, face in enumerate(frame_result.get("faces", [])):
        values = face.get("bbox_xyxy", [])
        if not isinstance(values, Sequence) or len(values) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in values]
        x1 = min(max(x1, 0), max(0, image_width - 1))
        y1 = min(max(y1, 0), max(0, image_height - 1))
        x2 = min(max(x2, 0), max(0, image_width - 1))
        y2 = min(max(y2, 0), max(0, image_height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        qualified = face.get("qualified") is True
        color = QUALIFIED_COLOR if qualified else UNQUALIFIED_COLOR
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        first = (
            f"F{face_index} {'QUAL' if qualified else 'NO'} "
            f"conf={float(face.get('detector_score', 0.0)):.2f} "
            f"area={float(face.get('area_ratio', 0.0)):.4f} "
            f"short={float(face.get('short_side_px', 0.0)):.0f}px"
        )
        second = (
            f"lap={float(face.get('laplacian_variance', 0.0)):.1f} "
            f"ten={float(face.get('tenengrad', 0.0)):.1f}"
        )
        baseline_y = y1 - 8
        if baseline_y < 34:
            baseline_y = min(image_height - 24, y2 + 18)
        _draw_label(rendered, first, (x1, baseline_y), color, scale)
        _draw_label(rendered, second, (x1, baseline_y + 18), color, scale)
    return rendered


def write_image_visualization(
    source: Path, result: Dict[str, Any], destination: Path
) -> Path:
    """Atomically write one annotated image at its original resolution."""
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Image cannot be decoded for visualization: {source}")
    frame_result = result.get("frames", [{}])[0] if result.get("frames") else {}
    rendered = draw_face_visualization(image, frame_result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix or ".jpg"
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{suffix}"
    )
    try:
        if not cv2.imwrite(str(temporary), rendered):
            raise RuntimeError(f"OpenCV failed to write visualization: {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_video_visualization(
    source: Path, result: Dict[str, Any], destination: Path
) -> Path:
    """Write an original-FPS/resolution H.264 visualization atomically."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Video cannot be opened for visualization: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise RuntimeError(f"Video has invalid visualization metadata: {source}")
    by_frame = {
        int(frame["frame_index"]): frame for frame in result.get("frames", [])
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.stem}.visualization-", dir=destination.parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        intermediate = temporary_root / "annotated.mp4"
        encoded = temporary_root / "encoded.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            writer.release()
            raise RuntimeError(f"VideoWriter cannot be opened: {intermediate}")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                annotation = by_frame.get(frame_index)
                writer.write(
                    draw_face_visualization(frame, annotation)
                    if annotation is not None else frame
                )
                frame_index += 1
        finally:
            capture.release()
            writer.release()
        if frame_index == 0:
            raise RuntimeError(f"Video contains no visualizable frames: {source}")
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-i", str(intermediate), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0 or not encoded.is_file() or encoded.stat().st_size == 0:
            message = completed.stderr.strip() or "ffmpeg produced no output"
            raise RuntimeError(f"H.264 visualization encoding failed: {message}")
        os.replace(encoded, destination)
    return destination


def write_media_visualization(
    source: Path,
    result: Dict[str, Any],
    input_root: Path,
    visualization_root: Path,
) -> Path:
    """Dispatch visualization writing using the result's media type."""
    media_type = str(result.get("media_type", ""))
    destination = visualization_path_for_media(
        source, input_root, visualization_root, media_type
    )
    if media_type == "image":
        return write_image_visualization(source, result, destination)
    if media_type == "video":
        return write_video_visualization(source, result, destination)
    raise ValueError(f"Unsupported result media_type: {media_type!r}")


def write_json_atomic(path: Path, payload: Any) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _model_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    root = Path(args.vbench_root).expanduser().resolve()
    cache = (
        Path(args.vbench_cache_dir).expanduser().resolve()
        if args.vbench_cache_dir else root / ".cache/vbench2"
    )
    config = (
        Path(args.detector_config).expanduser().resolve()
        if args.detector_config else root / (
            "vbench2/third_party/YOLO-World/"
            "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_"
            "obj365v1_goldg_train_lvis_minival.py"
        )
    )
    checkpoint = (
        Path(args.detector_checkpoint).expanduser().resolve()
        if args.detector_checkpoint else cache / (
            "YOLO-World/"
            "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
        )
    )
    clip_model = (
        Path(args.vbench_clip_model).expanduser().resolve()
        if args.vbench_clip_model else root / (
            ".cache/huggingface/openai/clip-vit-base-patch32"
        )
    )
    return root, config, checkpoint, clip_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Aggregate output JSON")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--min-qualified-frames", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vbench-root", default=str(DEFAULT_VBENCH_ROOT))
    parser.add_argument("--vbench-cache-dir")
    parser.add_argument(
        "--vbench-clip-model",
        help="Required local CLIP text-model directory used by YOLO-World",
    )
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
    parser.add_argument("--copy-selected-to")
    parser.add_argument(
        "--visualization-dir",
        help="Write annotated images/videos while preserving input layout",
    )
    parser.add_argument("--overwrite-copies", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.min_qualified_frames <= 0:
        parser.error("--min-qualified-frames must be positive")
    try:
        thresholds = FilterThresholds(
            area_threshold=args.area_threshold,
            min_face_short_side=args.min_face_short_side,
            laplacian_threshold=args.laplacian_threshold,
            tenengrad_threshold=args.tenengrad_threshold,
        )
        input_root, media = discover_media(Path(args.input), args.recursive)
        (
            vbench_root, detector_config, detector_checkpoint, clip_model
        ) = _model_paths(args)
        for model_path in (detector_config, detector_checkpoint):
            if not model_path.is_file():
                raise FileNotFoundError(f"YOLO-World resource is missing: {model_path}")
        if not clip_model.is_dir():
            raise FileNotFoundError(
                f"Local YOLO-World CLIP text model is missing: {clip_model}"
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    yolo_world_root = vbench_root / "vbench2/third_party/YOLO-World"
    if str(yolo_world_root) not in sys.path:
        sys.path.insert(0, str(yolo_world_root))
    with _environment_variable("VBENCH2_CLIP_TEXT_MODEL", str(clip_model)):
        with _environment_variable("HF_HUB_OFFLINE", "1"):
            with _environment_variable("TRANSFORMERS_OFFLINE", "1"):
                with _working_directory(vbench_root):
                    detector = FaceHandDetector(
                        detector_config, detector_checkpoint, args.device
                    )
    detection_options = {
        "detector_threshold": args.detector_threshold,
        "tiled_inference": args.tiled_inference,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "whole_image_detection": args.whole_image_detection,
        "nms_iou_threshold": args.nms_iou_threshold,
    }
    items = []
    selected_paths = []
    visualization_failures = 0
    for index, path in enumerate(media, start=1):
        LOGGER.info("Processing %d/%d: %s", index, len(media), path)
        try:
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                result = process_image(
                    path, detector, thresholds, detection_options,
                    args.min_qualified_frames,
                )
            else:
                result = process_video(
                    path, detector, thresholds, detection_options,
                    args.frame_stride, args.min_qualified_frames,
                )
            result["error"] = None
            result["visualization"] = None
            result["visualization_error"] = None
            if args.visualization_dir:
                try:
                    visualization = write_media_visualization(
                        path, result, input_root, Path(args.visualization_dir)
                    )
                    result["visualization"] = str(visualization)
                except Exception as exc:
                    visualization_failures += 1
                    LOGGER.exception("Failed to visualize %s", path)
                    result["visualization_error"] = {
                        "error_type": type(exc).__name__, "message": str(exc),
                    }
            if result["selected"] and args.copy_selected_to:
                copied = copy_selected(
                    path, input_root, Path(args.copy_selected_to),
                    args.overwrite_copies,
                )
                result["copied_to"] = str(copied)
            else:
                result["copied_to"] = None
            if result["selected"]:
                selected_paths.append(str(path))
        except Exception as exc:
            LOGGER.exception("Failed to process %s", path)
            result = {
                "path": str(path), "selected": False,
                "qualified_frame_count": 0, "frames": [],
                "error": {
                    "error_type": type(exc).__name__, "message": str(exc),
                },
                "copied_to": None,
                "visualization": None,
                "visualization_error": None,
            }
        items.append(result)

    output = {
        "schema_version": "1.0",
        "input": str(Path(args.input).expanduser().resolve()),
        "config": {
            **asdict(thresholds),
            **detection_options,
            "frame_stride": args.frame_stride,
            "min_qualified_frames": args.min_qualified_frames,
            "device": args.device,
            "detector_config": str(detector_config),
            "detector_checkpoint": str(detector_checkpoint),
            "clip_text_model": str(clip_model),
            "visualization_dir": (
                str(Path(args.visualization_dir).expanduser().resolve())
                if args.visualization_dir else None
            ),
        },
        "sample_count": len(items),
        "selected_count": len(selected_paths),
        "selected_paths": selected_paths,
        "items": items,
    }
    write_json_atomic(Path(args.output), output)
    LOGGER.info(
        "Completed %d samples; selected=%d; output=%s",
        len(items), len(selected_paths), Path(args.output).expanduser().resolve(),
    )
    successful = any(item.get("error") is None for item in items)
    return 0 if successful and visualization_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
