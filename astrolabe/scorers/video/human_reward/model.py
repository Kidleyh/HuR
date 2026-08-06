"""Unified in-memory orchestration for HuR Human Reward."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import yaml

from astrolabe.scorers.video.human_anomaly.aggregation import aggregate_human_anomaly
from astrolabe.scorers.video.human_anomaly.engine import (
    HumanAnomalyEngine,
    _is_cuda_failure,
)
from astrolabe.scorers.video.human_anomaly.manifest import build_human_anomaly_entries
from astrolabe.scorers.video.human_anomaly.validation import validate_worker_results
from astrolabe.scorers.video.person_tracking.tracker import YOLOByteTrackPersonTracker
from astrolabe.scorers.video.tracklet_stitching.io import tracking_input_from_result
from astrolabe.scorers.video.tracklet_stitching.schemas import StitchingConfig
from astrolabe.scorers.video.tracklet_stitching.stitcher import stitch_tracking
from .visualization import write_human_reward_visualization

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VBENCH_ROOT = Path(os.environ.get(
    "VBENCH_ROOT", "/gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0"
))


@dataclass(frozen=True)
class HumanRewardConfig:
    """Paths and inference controls for one in-memory reward evaluation."""

    yolo_weights: Path = field(
        default_factory=lambda: PROJECT_ROOT / "checkpoints/yolo/yolov8x.pt"
    )
    tracker_config: Path = field(
        default_factory=lambda: PROJECT_ROOT / "configs/bytetrack_person.yaml"
    )
    stitching_config: Path = field(
        default_factory=lambda: PROJECT_ROOT / "configs/tracklet_stitching.yaml"
    )
    vbench_root: Path = field(default_factory=lambda: DEFAULT_VBENCH_ROOT)
    vbench_cache_dir: Optional[Path] = None
    vbench_clip_model: Optional[Path] = None
    device: str = "cuda:0"
    conf: float = 0.10
    iou: float = 0.70
    imgsz: int = 640
    half: bool = True
    crop_batch_size: int = 128

    def __post_init__(self) -> None:
        for name in ("yolo_weights", "tracker_config", "stitching_config", "vbench_root"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        cache = self.vbench_cache_dir or self.vbench_root / ".cache/vbench2"
        object.__setattr__(self, "vbench_cache_dir", Path(cache).expanduser().resolve())
        clip_model = self.vbench_clip_model or (
            self.vbench_root / ".cache/huggingface/openai/clip-vit-base-patch32"
        )
        object.__setattr__(
            self, "vbench_clip_model", Path(clip_model).expanduser().resolve()
        )
        if self.crop_batch_size <= 0:
            raise ValueError("crop_batch_size must be positive")

    @classmethod
    def from_value(
        cls, value: Optional[Union["HumanRewardConfig", Mapping[str, Any]]]
    ) -> "HumanRewardConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))


def _load_stitching_config(path: Path) -> StitchingConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Stitching config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Stitching config must be a YAML mapping: {path}")
    return StitchingConfig(**data)


def _tracking_device(device: str) -> str:
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def _release_cuda_models() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _invalid_result(reason: str) -> Dict[str, Any]:
    """Build the stable result shape for one invalid video."""
    return {
        "valid": False,
        "reason": reason,
        "reward": None,
        "micro_score": None,
        "macro_score": None,
        "logical_track_count": 0,
        "observed_person_frames": 0,
        "scored_person_frames": 0,
        "abnormal_person_frames": 0,
        "failed_person_frames": 0,
        "visualization": None,
    }


def _failure_reason(stage: str, error: BaseException) -> str:
    return f"{stage}_failed: {type(error).__name__}: {error}"


@dataclass(frozen=True)
class _PreparedVideo:
    video: Path
    entries: Any
    width: int
    height: int


class HumanRewardModel:
    """Run tracking, stitching, and anatomy scoring entirely in memory."""

    def __init__(
        self,
        config: Optional[Union[HumanRewardConfig, Mapping[str, Any]]] = None,
        *,
        tracker_factory: Optional[Callable[..., Any]] = None,
        anomaly_engine_factory: Optional[Callable[..., Any]] = None,
        stitcher: Optional[Callable[..., Any]] = None,
        release_callback: Optional[Callable[[], None]] = None,
        visualization_writer: Optional[Callable[..., None]] = None,
    ) -> None:
        self.config = HumanRewardConfig.from_value(config)
        self._tracker_factory = tracker_factory or YOLOByteTrackPersonTracker
        self._anomaly_engine_factory = anomaly_engine_factory or HumanAnomalyEngine
        self._stitcher = stitcher or stitch_tracking
        self._release_callback = release_callback or _release_cuda_models
        self._visualization_writer = (
            visualization_writer or write_human_reward_visualization
        )
        self._stitching_config = _load_stitching_config(self.config.stitching_config)

    def score_batch(
        self,
        video_paths: Sequence[Union[str, Path]],
        *,
        visualization_output: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Score videos in order while loading each model family only once."""
        if isinstance(video_paths, (str, Path)):
            raise TypeError(
                "score_batch expects a sequence of video paths; "
                "use score() for one video"
            )
        videos = [Path(path).expanduser().resolve() for path in video_paths]
        if not videos:
            return []
        if visualization_output is not None and len(videos) != 1:
            raise ValueError(
                "visualization_output supports single-video mode only"
            )
        results: List[Optional[Dict[str, Any]]] = [None] * len(videos)
        prepared: List[Optional[_PreparedVideo]] = [None] * len(videos)
        visualization_ready = False
        visualization_frame_results: Sequence[Dict[str, Any]] = []
        tracker = None
        try:
            tracker = self._tracker_factory(
                weights=str(self.config.yolo_weights),
                tracker_config=str(self.config.tracker_config),
                device=_tracking_device(self.config.device),
                conf=self.config.conf,
                iou=self.config.iou,
                imgsz=self.config.imgsz,
                half=self.config.half,
                allow_download=False,
            )
            for index, video in enumerate(videos):
                try:
                    if not video.is_file():
                        raise FileNotFoundError(f"Input video does not exist: {video}")
                    tracking = tracker.track_video_in_memory(str(video))
                    stitching = self._stitcher(
                        tracking_input_from_result(tracking), self._stitching_config
                    )
                    entries, _ = build_human_anomaly_entries(
                        tracking.frames,
                        stitching.track_id_to_logical_track_id,
                        tracking.video.width,
                        tracking.video.height,
                    )
                    if not entries:
                        results[index] = _invalid_result("no_person_detected")
                        if visualization_output is not None:
                            visualization_ready = True
                        continue
                    prepared[index] = _PreparedVideo(
                        video=video,
                        entries=entries,
                        width=tracking.video.width,
                        height=tracking.video.height,
                    )
                except Exception as error:
                    if _is_cuda_failure(error):
                        raise
                    results[index] = _invalid_result(
                        _failure_reason("tracking", error)
                    )
        finally:
            tracker = None
            self._release_callback()

        if any(item is not None for item in prepared):
            engine = self._anomaly_engine_factory(
                vbench_root=self.config.vbench_root,
                cache_dir=self.config.vbench_cache_dir,
                clip_model=self.config.vbench_clip_model,
                device=self.config.device,
                crop_batch_size=self.config.crop_batch_size,
            )
            try:
                for index, item in enumerate(prepared):
                    if item is None:
                        continue
                    try:
                        frame_results = engine.score_video(item.video, item.entries)
                        validate_worker_results(item.entries, frame_results)
                        _, summary = aggregate_human_anomaly(
                            item.entries,
                            frame_results,
                            item.width,
                            item.height,
                        )
                        micro = summary["video_micro_score"]
                        macro = summary["video_macro_score"]
                        if micro is None or macro is None:
                            raise RuntimeError(
                                "Human Reward aggregation produced no valid score"
                            )
                        results[index] = {
                            "valid": True,
                            "reason": None,
                            "reward": float(micro),
                            "micro_score": float(micro),
                            "macro_score": float(macro),
                            "logical_track_count": int(summary["logical_track_count"]),
                            "observed_person_frames": int(
                                summary["observed_person_frames"]
                            ),
                            "scored_person_frames": int(
                                summary["scored_person_frames"]
                            ),
                            "abnormal_person_frames": int(
                                summary["abnormal_person_frames"]
                            ),
                            "failed_person_frames": int(
                                summary["failed_person_frames"]
                            ),
                            "visualization": None,
                        }
                        if visualization_output is not None:
                            visualization_frame_results = frame_results
                            visualization_ready = True
                    except Exception as error:
                        if _is_cuda_failure(error):
                            raise
                        results[index] = _invalid_result(
                            _failure_reason("anomaly", error)
                        )
            finally:
                close = getattr(engine, "close", None)
                if callable(close):
                    close()
                engine = None
                self._release_callback()

        if visualization_output is not None and visualization_ready:
            destination = Path(visualization_output).expanduser().resolve()
            result = results[0]
            if result is None:
                raise RuntimeError("Internal error: visualization result is missing")
            self._visualization_writer(
                videos[0], visualization_frame_results, result, destination
            )
            result["visualization"] = str(destination)

        if any(result is None for result in results):
            raise RuntimeError("Internal error: batch result was not populated")
        return [result for result in results if result is not None]

    def score(
        self,
        video_path: Union[str, Path],
        visualization_output: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Score one video through the shared batch implementation."""
        if visualization_output is None:
            return self.score_batch([video_path])[0]
        return self.score_batch(
            [video_path], visualization_output=visualization_output
        )[0]
