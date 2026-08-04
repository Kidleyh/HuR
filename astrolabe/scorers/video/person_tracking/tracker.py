"""Single-pass Ultralytics YOLOv8x detection with ByteTrack association."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from .schemas import FrameDetections, RawDetection, TrackedDetection, VideoInfo, VideoTrackingResult
from .serialization import write_tracking_outputs
from .statistics import compute_detection_summary, compute_track_statistics

LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
GENERATED_FILENAMES = frozenset(
    {
        "detections.jsonl", "detections.csv", "raw_detections.csv",
        "tracked_detections.csv", "tracks_summary.json", "tracked.mp4", "error.json",
    }
)


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
    if os.environ.get("YOLO_WEIGHTS"):
        candidates.append(Path(os.environ["YOLO_WEIGHTS"]).expanduser())
    if os.environ.get("GVHMR_ROOT"):
        candidates.append(
            Path(os.environ["GVHMR_ROOT"]).expanduser()
            / "inputs/checkpoints/yolo/yolov8x.pt"
        )
    candidates.append(root / "checkpoints/yolo/yolov8x.pt")
    checked: List[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.is_file():
            return str(resolved)
    if allow_download:
        return "yolov8x.pt"
    raise FileNotFoundError(
        "YOLOv8x weights were not found. Checked:\n  - "
        + "\n  - ".join(checked)
        + "\nPass --allow-download to explicitly allow Ultralytics to download yolov8x.pt."
    )


def discover_videos(input_path: Union[str, Path], recursive: bool = False) -> List[Path]:
    """Return a deterministic list of supported input videos."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension for {path}; expected one of {sorted(VIDEO_EXTENSIONS)}"
            )
        return [path]
    iterator = path.rglob("*") if recursive else path.glob("*")
    videos = sorted(
        item for item in iterator if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    )
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


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1 = max(float(first[0]), float(second[0])), max(float(first[1]), float(second[1]))
    x2, y2 = min(float(first[2]), float(second[2])), min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _source_indices_for_tracks(
    tracks: np.ndarray, raw_detections: List[RawDetection], minimum_iou: float = 0.5
) -> List[Optional[int]]:
    """Use ByteTrack's source index, with one-to-one IoU fallback."""
    matches: List[Optional[int]] = [None] * len(tracks)
    used: set[int] = set()
    for track_index, track in enumerate(tracks):
        source_index = int(track[-1]) if len(track) >= 8 and math.isfinite(float(track[-1])) else -1
        track_class = int(track[6]) if len(track) >= 7 else -1
        if (
            0 <= source_index < len(raw_detections)
            and source_index not in used
            and raw_detections[source_index].class_id == track_class
        ):
            matches[track_index] = source_index
            used.add(source_index)
    candidates: List[Tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        if matches[track_index] is not None:
            continue
        track_class = int(track[6]) if len(track) >= 7 else -1
        for raw_index, raw in enumerate(raw_detections):
            if raw_index in used or raw.class_id != track_class:
                continue
            iou = _bbox_iou(track[:4], raw.bbox_xyxy)
            if iou >= minimum_iou:
                candidates.append((iou, track_index, raw_index))
    for _, track_index, raw_index in sorted(candidates, reverse=True):
        if matches[track_index] is None and raw_index not in used:
            matches[track_index] = raw_index
            used.add(raw_index)
    return matches


class YOLOByteTrackPersonTracker:
    """Load YOLO once and isolate a fresh ByteTracker for every video."""

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
        if not 0.0 <= self.conf <= 1.0 or not 0.0 <= self.iou <= 1.0:
            raise ValueError("conf and iou must be in [0, 1]")
        if self.imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if importlib.util.find_spec("lap") is None:
            raise ImportError("ByteTrack requires the 'lap' module. Install lapx>=0.5.2.")
        self._uses_cuda = self.device.lower() != "cpu"
        if self._uses_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device!r} was requested, but CUDA is unavailable. "
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
        self.model = YOLO(self.weights)
        self.person_class_id = _person_class_id(self.model.names)

    def _new_byte_tracker(self) -> BYTETracker:
        settings = dict(self.tracker_settings)
        settings["device"] = self.device
        return BYTETracker(args=SimpleNamespace(**settings))

    def _raw_detections(
        self, frame: np.ndarray, width: int, height: int, warnings: List[str], frame_index: int
    ) -> List[RawDetection]:
        predict_options: Dict[str, Any] = {
            "source": frame,
            "classes": [self.person_class_id],
            "device": self.device,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
            "save": False,
        }
        if self.half:
            predict_options["half"] = True
        prediction = self.model.predict(**predict_options)[0]
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return []
        raw_detections: List[RawDetection] = []
        values = zip(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
        )
        for raw_box, raw_confidence, raw_class in values:
            class_id = int(raw_class)
            if class_id != self.person_class_id:
                continue
            try:
                raw_detections.append(
                    RawDetection.from_xyxy(
                        class_id=class_id,
                        class_name="person",
                        confidence=float(raw_confidence),
                        bbox_xyxy=raw_box,
                        image_width=width,
                        image_height=height,
                        detection_index=len(raw_detections),
                    )
                )
            except ValueError as error:
                warnings.append(f"frame {frame_index}: dropped invalid raw box: {error}")
        return raw_detections

    def _tracked_detections(
        self,
        byte_tracker: BYTETracker,
        raw_detections: List[RawDetection],
        frame: np.ndarray,
        width: int,
        height: int,
        warnings: List[str],
        frame_index: int,
    ) -> List[TrackedDetection]:
        rows = np.asarray(
            [[*item.bbox_xyxy, item.confidence, item.class_id] for item in raw_detections],
            dtype=np.float32,
        ).reshape((-1, 6))
        tracks = byte_tracker.update(Boxes(rows, orig_shape=(height, width)), frame)
        if len(tracks) == 0:
            return []
        matches = _source_indices_for_tracks(tracks, raw_detections)
        tracked: List[TrackedDetection] = []
        for track, source_index in zip(tracks, matches):
            try:
                tracked.append(
                    TrackedDetection.from_xyxy(
                        track_id=int(track[4]),
                        class_id=int(track[6]),
                        class_name="person",
                        confidence=float(track[5]),
                        bbox_xyxy=track[:4],
                        image_width=width,
                        image_height=height,
                        source_detection_index=source_index,
                    )
                )
            except (IndexError, ValueError) as error:
                warnings.append(f"frame {frame_index}: dropped invalid tracked box: {error}")
        return tracked

    @staticmethod
    def _promote_outputs(temporary: Path, destination: Path) -> None:
        """Promote complete artifacts and clean only declared module outputs."""
        destination.mkdir(parents=True, exist_ok=True)
        completed = {
            path.name
            for path in temporary.iterdir()
            if path.is_file() and path.name in GENERATED_FILENAMES
        }
        for filename in sorted(completed):
            os.replace(temporary / filename, destination / filename)
        for filename in GENERATED_FILENAMES - completed:
            stale = destination / filename
            if stale.is_file():
                stale.unlink()

    @staticmethod
    def _draw_visualization(
        writer: cv2.VideoWriter,
        image: np.ndarray,
        raw_detections: List[RawDetection],
        tracked_detections: List[TrackedDetection],
        height: int,
    ) -> None:
        visualization = image.copy()
        associated = {
            detection.source_detection_index
            for detection in tracked_detections
            if detection.source_detection_index is not None
        }
        for detection in tracked_detections:
            x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
            color = _stable_color(detection.track_id)
            cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                visualization,
                f"person #{detection.track_id} {detection.confidence:.2f}",
                (x1, max(y1 - 7, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        for detection in raw_detections:
            if detection.detection_index in associated:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 165, 255), 1)
            cv2.putText(
                visualization,
                f"raw-untracked {detection.confidence:.2f}",
                (x1, min(y2 + 16, height - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )
        writer.write(visualization)

    @staticmethod
    def _transcode_visualization(source: Path, destination: Path) -> None:
        """Produce browser-compatible H.264/yuv420p output from OpenCV's intermediate video."""
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"H.264 visualization transcoding failed: {completed.stderr.strip()}")

    def track_video(
        self,
        video_path: str,
        output_dir: str,
        save_visualization: bool = True,
        save_raw_csv: bool = True,
    ) -> VideoTrackingResult:
        """Detect once per frame, associate, and promote only complete outputs."""
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
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError(f"Video has invalid dimensions {width}x{height}: {source}")
        if fps <= 0 or not math.isfinite(fps):
            capture.release()
            raise RuntimeError(f"Video FPS is invalid ({fps}): {source}")

        warnings: List[str] = []
        high_threshold = float(self.tracker_settings["track_high_thresh"])
        low_threshold = float(self.tracker_settings["track_low_thresh"])
        if self.conf > high_threshold:
            message = (
                "Detection confidence is above ByteTrack high threshold; "
                "low-score association will be disabled."
            )
            LOGGER.warning(message)
            warnings.append(message)
        elif self.conf > low_threshold:
            message = "Detection confidence is above ByteTrack low threshold; some candidates are filtered."
            LOGGER.warning(message)
            warnings.append(message)

        destination.parent.mkdir(parents=True, exist_ok=True)
        byte_tracker = self._new_byte_tracker()
        frames: List[FrameDetections] = []
        started = time.perf_counter()
        prefix = f".{destination.name}.tmp-"
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=prefix) as temporary_name:
            temporary = Path(temporary_name)
            writer: Optional[cv2.VideoWriter] = None
            try:
                if save_visualization:
                    intermediate_video = temporary / "tracked_intermediate.mp4"
                    writer = cv2.VideoWriter(
                        str(intermediate_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter could not be created: {intermediate_video}")
                frame_index = 0
                while True:
                    ok, frame_image = capture.read()
                    if not ok:
                        break
                    raw_detections = self._raw_detections(
                        frame_image, width, height, warnings, frame_index
                    )
                    tracked_detections = self._tracked_detections(
                        byte_tracker, raw_detections, frame_image, width, height, warnings, frame_index
                    )
                    frames.append(
                        FrameDetections(
                            frame_index=frame_index,
                            timestamp_sec=frame_index / fps,
                            raw_detections=raw_detections,
                            tracked_detections=tracked_detections,
                        )
                    )
                    if writer is not None:
                        self._draw_visualization(
                            writer, frame_image, raw_detections, tracked_detections, height
                        )
                    frame_index += 1
            finally:
                capture.release()
                if writer is not None:
                    writer.release()

            if save_visualization:
                self._transcode_visualization(intermediate_video, temporary / "tracked.mp4")

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
                    "person_class_id": self.person_class_id,
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
                    "device": (
                        f"cuda:{self.device}"
                        if self._uses_cuda and self.device.isdigit()
                        else self.device
                    ),
                    "half": self.half,
                    "processed_frames": processed_frames,
                    "runtime_sec": runtime,
                    "fps_effective": processed_frames / runtime if runtime > 0 else 0.0,
                },
                tracks=compute_track_statistics(frames, processed_frames),
                detection_summary=compute_detection_summary(frames, processed_frames),
                warnings=warnings,
            )
            write_tracking_outputs(result, temporary, save_raw_csv=save_raw_csv)
            self._promote_outputs(temporary, destination)
            return result
