"""Ultralytics YOLOv8x person detection with ByteTrack association."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import torch
import ultralytics
import yaml
from ultralytics import YOLO

from .schemas import Detection, FrameDetections, VideoInfo, VideoTrackingResult
from .serialization import write_tracking_outputs
from .statistics import compute_track_statistics

LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def resolve_yolo_weights(
    cli_weights: Optional[str] = None,
    *,
    allow_download: bool = False,
    project_root: Optional[Path] = None,
) -> str:
    """Resolve YOLOv8x weights in the documented priority order."""
    root = (project_root or Path(__file__).resolve().parents[4]).resolve()
    candidates: List[Path] = []
    if cli_weights:
        candidates.append(Path(cli_weights).expanduser())
    environment_weights = os.environ.get("YOLO_WEIGHTS")
    if environment_weights:
        candidates.append(Path(environment_weights).expanduser())
    gvhmr_root = os.environ.get("GVHMR_ROOT")
    if gvhmr_root:
        candidates.append(Path(gvhmr_root).expanduser() / "inputs/checkpoints/yolo/yolov8x.pt")
    candidates.append(root / "checkpoints/yolo/yolov8x.pt")

    checked: List[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if candidate.is_file():
            return str(candidate)
    if allow_download:
        return "yolov8x.pt"
    checked_text = "\n  - ".join(checked)
    raise FileNotFoundError(
        "YOLOv8x weights were not found. Checked:\n  - "
        f"{checked_text}\nPass --allow-download to explicitly allow Ultralytics to download yolov8x.pt."
    )


def discover_videos(input_path: Union[str, Path], recursive: bool = False) -> List[Path]:
    """Return a deterministic list of supported input videos."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension for {path}; expected one of {sorted(VIDEO_EXTENSIONS)}")
        return [path]
    iterator = path.rglob("*") if recursive else path.glob("*")
    videos = sorted(item for item in iterator if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        mode = "recursively " if recursive else ""
        raise FileNotFoundError(f"No supported videos found {mode}in directory: {path}")
    return videos


def _person_class_id(names: Union[Mapping[int, str], Sequence[str]]) -> int:
    entries = names.items() if isinstance(names, Mapping) else enumerate(names)
    for class_id, name in entries:
        if str(name).strip().lower() == "person":
            return int(class_id)
    raise ValueError(f"Loaded YOLO weights do not contain a 'person' class. Model classes: {names}")


def _stable_color(track_id: int) -> Tuple[int, int, int]:
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 137) % 192,
    )


class YOLOByteTrackPersonTracker:
    """Reusable front-end that keeps tracker state isolated per input video."""

    def __init__(
        self,
        weights: str,
        tracker_config: str,
        device: str = "0",
        conf: float = 0.10,
        iou: float = 0.70,
        imgsz: int = 640,
        half: bool = True,
        allow_download: bool = False,
    ) -> None:
        self.weights = weights
        self.tracker_config = str(Path(tracker_config).expanduser().resolve())
        self.device = str(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self.allow_download = bool(allow_download)
        if not Path(weights).expanduser().is_file() and not (
            self.allow_download and weights == "yolov8x.pt"
        ):
            raise FileNotFoundError(f"YOLO weights do not exist: {weights}")
        if not Path(self.tracker_config).is_file():
            raise FileNotFoundError(f"ByteTrack config does not exist: {self.tracker_config}")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("conf must be in [0, 1]")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou must be in [0, 1]")
        if self.imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if importlib.util.find_spec("lap") is None:
            raise ImportError("ByteTrack requires the 'lap' module. Install lapx>=0.5.2.")
        self._uses_cuda = self.device.lower() != "cpu"
        if self._uses_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device!r} was requested, but torch.cuda.is_available() is false. "
                "Use --device cpu for CPU execution."
            )
        if not self._uses_cuda and self.half:
            LOGGER.warning("Half precision is disabled automatically on CPU.")
            self.half = False
        try:
            with Path(self.tracker_config).open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid ByteTrack YAML {self.tracker_config}: {error}") from error
        if not isinstance(config, dict) or config.get("tracker_type") != "bytetrack":
            raise ValueError(f"Tracker config must define tracker_type: bytetrack: {self.tracker_config}")
        self.tracker_settings: Dict[str, Any] = config

    def _new_model(self) -> Tuple[Any, int]:
        # A fresh model for every video prevents Ultralytics predictor/tracker caches
        # from leaking track state between calls.
        model = YOLO(self.weights)
        return model, _person_class_id(model.names)

    def track_video(
        self,
        video_path: str,
        output_dir: str,
        save_visualization: bool = True,
    ) -> VideoTrackingResult:
        """Track all people in one video and write the standardized artifacts."""
        source = Path(video_path).expanduser().resolve()
        destination = Path(output_dir).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input video does not exist: {source}")

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Video cannot be opened: {source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Video has invalid dimensions {width}x{height}: {source}")
        if fps <= 0 or not math.isfinite(fps):
            raise RuntimeError(f"Video FPS is invalid ({fps}): {source}")

        destination.mkdir(parents=True, exist_ok=True)
        writer: Optional[cv2.VideoWriter] = None
        if save_visualization:
            writer = cv2.VideoWriter(
                str(destination / "tracked.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"VideoWriter could not be created: {destination / 'tracked.mp4'}")

        model, person_id = self._new_model()
        frames: List[FrameDetections] = []
        warnings: List[str] = []
        started = time.perf_counter()
        try:
            results = model.track(
                source=str(source),
                tracker=self.tracker_config,
                stream=True,
                classes=[person_id],
                device=self.device,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                half=self.half,
                verbose=False,
                save=False,
            )
            for frame_index, result in enumerate(results):
                detections: List[Detection] = []
                boxes = result.boxes
                if boxes is not None and len(boxes) and boxes.id is not None:
                    xyxy_values = boxes.xyxy.detach().cpu().tolist()
                    confidence_values = boxes.conf.detach().cpu().tolist()
                    class_values = boxes.cls.detach().cpu().tolist()
                    track_values = boxes.id.detach().cpu().tolist()
                    for raw_box, raw_conf, raw_class, raw_track in zip(
                        xyxy_values, confidence_values, class_values, track_values
                    ):
                        class_id = int(raw_class)
                        if class_id != person_id:
                            continue
                        try:
                            detections.append(
                                Detection.from_xyxy(
                                    track_id=int(raw_track),
                                    class_id=class_id,
                                    class_name="person",
                                    confidence=float(raw_conf),
                                    bbox_xyxy=raw_box,
                                    image_width=width,
                                    image_height=height,
                                )
                            )
                        except ValueError as error:
                            warnings.append(f"frame {frame_index}: dropped invalid box: {error}")
                frame = FrameDetections(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    detections=detections,
                )
                frames.append(frame)
                if writer is not None:
                    image = result.orig_img
                    if image is None:
                        raise RuntimeError(f"Ultralytics returned no source image for frame {frame_index}")
                    if image.shape[1] != width or image.shape[0] != height:
                        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
                        warnings.append(f"frame {frame_index}: visualization resized to original dimensions")
                    for detection in detections:
                        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
                        color = _stable_color(detection.track_id)
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        label = f"person #{detection.track_id} {detection.confidence:.2f}"
                        cv2.putText(
                            image,
                            label,
                            (x1, max(y1 - 7, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                    writer.write(image)
        finally:
            if writer is not None:
                writer.release()

        runtime = time.perf_counter() - started
        processed_frames = len(frames)
        if processed_frames == 0:
            raise RuntimeError(f"No frames could be decoded from video: {source}")
        if declared_frames > 0 and declared_frames != processed_frames:
            warnings.append(
                f"container declared {declared_frames} frames but {processed_frames} were decoded"
            )
        video = VideoInfo(
            path=str(source),
            width=width,
            height=height,
            fps=fps,
            num_frames=processed_frames,
            duration_sec=processed_frames / fps,
        )
        result = VideoTrackingResult(
            video=video,
            frames=frames,
            detector={
                "backend": "ultralytics",
                "model": "yolov8x",
                "weights": self.weights,
                "ultralytics_version": ultralytics.__version__,
                "person_class_id": person_id,
                "conf": self.conf,
                "iou": self.iou,
                "imgsz": self.imgsz,
            },
            tracker={
                "backend": "bytetrack",
                "config_path": self.tracker_config,
                "config": self.tracker_settings,
            },
            processing={
                "device": f"cuda:{self.device}" if self._uses_cuda and self.device.isdigit() else self.device,
                "half": self.half,
                "processed_frames": processed_frames,
                "runtime_sec": runtime,
                "fps_effective": processed_frames / runtime if runtime > 0 else 0.0,
            },
            tracks=compute_track_statistics(frames, processed_frames),
            warnings=warnings,
        )
        write_tracking_outputs(result, destination)
        return result
