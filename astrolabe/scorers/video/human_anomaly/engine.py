"""In-process VBench Human/Face/Hand inference engine."""

from __future__ import annotations

import gc
import importlib
import os
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union

import cv2

from .schema import (
    OFFICIAL_THRESHOLDS, HumanAnomalyInput, classifier_result, person_is_abnormal,
)


def _is_cuda_failure(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, RuntimeError) and any(
        token in message for token in ("cuda", "cudnn", "out of memory", "device-side")
    )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _environment_variable(name: str, value: Union[str, Path, None]) -> Iterator[None]:
    """Temporarily set one environment variable without leaking process state."""
    previous = os.environ.get(name)
    if value is not None:
        os.environ[name] = str(value)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


class FaceHandDetector:
    """One YOLO-World instance, parameterized only for face and hand."""

    def __init__(self, config_path: Path, weight_path: Path, device: str) -> None:
        from mmengine.dataset import Compose
        from mmdet.apis import init_detector

        self.model = init_detector(str(config_path), str(weight_path), device=device)
        self.model.cfg.test_dataloader.dataset.pipeline[0].type = (
            "mmdet.LoadImageFromNDArray"
        )
        self.pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)
        self.texts = [["face"], ["hand"], [" "]]
        self.model.reparameterize(self.texts)

    def detect(self, image: Any, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        import torch

        data_info = self.pipeline(dict(img_id=0, img=image, texts=self.texts))
        batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )
        with torch.no_grad():
            output = self.model.test_step(batch)[0].pred_instances
            output = output[output.scores.float() > score_threshold]
        boxes = output.bboxes.detach().cpu().numpy()
        labels = output.labels.detach().cpu().numpy()
        scores = output.scores.detach().cpu().numpy()
        return [
            {"bbox_xyxy": box.tolist(), "label": int(label),
             "detector_score": float(score)}
            for box, label, score in zip(boxes, labels, scores)
            if int(label) in (0, 1)
        ]


def _load_vbench(
    vbench_root: Path, cache_dir: Path, device: str, batch_size: int
) -> Tuple[Any, FaceHandDetector, Dict[str, Dict[str, str]], Path, Path]:
    vit_root = vbench_root / "vbench2/third_party/ViTDetector"
    yolo_root = vbench_root / "vbench2/third_party/YOLO-World"
    for path in (str(vit_root), str(yolo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    # Some VBench transitive imports create log_rank0.txt in the current directory.
    # Isolate that import-time side effect in an automatically removed directory.
    with tempfile.TemporaryDirectory(prefix="hur-vbench-import-") as temporary:
        with _working_directory(Path(temporary)):
            inference = importlib.import_module("inference")
    classifier_config = vit_root / "simmim_finetune__vit_base__img224__800ep.yaml"
    model_configs = {
        category: {
            "cfg_path": str(classifier_config),
            "weight_path": str(cache_dir / f"anomaly_detector/{category}.pth"),
        }
        for category in ("human", "face", "hand")
    }
    required = [classifier_config] + [
        Path(value["weight_path"]) for value in model_configs.values()
    ]
    for item in required:
        if not item.is_file():
            raise FileNotFoundError(f"Required VBench model/config is missing: {item}")
    detector_config = vbench_root / (
        "vbench2/third_party/YOLO-World/"
        "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    )
    detector_weight = cache_dir / (
        "YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
    )
    if not detector_config.is_file() or not detector_weight.is_file():
        raise FileNotFoundError("YOLO-World config or checkpoint is missing")
    with _working_directory(vbench_root):
        analyzer = inference.Analyzer(
            model_configs, device=device, batch_size=batch_size,
            class_thresholds=OFFICIAL_THRESHOLDS,
        )
        detector = FaceHandDetector(detector_config, detector_weight, device)
    return analyzer, detector, model_configs, detector_config, detector_weight


def _part_result(
    analyzer: Any, category: str, frame: Any, bbox: Sequence[float],
    detector_score: float,
) -> Dict[str, Any]:
    crop = analyzer.smart_cut(frame, bbox)
    scores = analyzer.predict_batch(category, [crop])[0]
    result = classifier_result(scores, category)
    result.update({
        "bbox_xyxy": [float(value) for value in bbox],
        "detector_score": float(detector_score),
    })
    return result


def _process_person(
    analyzer: Any, detector: FaceHandDetector, frame: Any, entry: Dict[str, Any]
) -> Dict[str, Any]:
    base = {
        "frame_index": entry["frame_index"],
        "logical_track_id": entry["logical_track_id"],
        "source_track_id": entry["source_track_id"],
        "bbox_xyxy": entry["bbox_xyxy"],
        "human": {"scored": False, "scores": [], "abnormal_probability": None,
                  "abnormal": False},
        "faces": [], "hands": [], "person_abnormal": False,
        "failure_reason": None, "failures": [],
    }

    def record_failure(stage: str, error: Exception) -> None:
        if _is_cuda_failure(error):
            raise error
        base["failures"].append({
            "stage": stage, "error_type": type(error).__name__, "message": str(error),
        })

    bbox = entry["bbox_xyxy"]
    try:
        human_crop = analyzer.smart_cut(frame, bbox)
        base["human"] = classifier_result(
            analyzer.predict_batch("human", [human_crop])[0], "human"
        )
    except Exception as error:
        record_failure("human_scoring", error)

    detected_parts: List[Dict[str, Any]] = []
    try:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        direct_crop = frame[y1:y2, x1:x2]
        if direct_crop.size == 0:
            raise ValueError("empty direct human crop for face/hand detection")
        detected_parts = detector.detect(direct_crop)
    except Exception as error:
        record_failure("face_hand_detection", error)

    for part in detected_parts:
        category = "face" if part["label"] == 0 else "hand"
        try:
            local = part["bbox_xyxy"]
            global_bbox = [
                local[0] + x1, local[1] + y1, local[2] + x1, local[3] + y1
            ]
            scored = _part_result(
                analyzer, category, frame, global_bbox, part["detector_score"]
            )
            base["faces" if category == "face" else "hands"].append(scored)
        except Exception as error:
            record_failure(f"{category}_scoring", error)

    base["person_abnormal"] = person_is_abnormal(
        base["human"], base["faces"], base["hands"]
    )
    if base["failures"]:
        base["failure_reason"] = "; ".join(
            f"{failure['stage']}: {failure['error_type']}: {failure['message']}"
            for failure in base["failures"]
        )
    return base


class HumanAnomalyEngine:
    """Load VBench models once and score HuR-provided person boxes in process."""

    def __init__(
        self, vbench_root: Union[str, Path], cache_dir: Union[str, Path],
        device: str = "cuda:0", crop_batch_size: int = 128,
        clip_model: Union[str, Path, None] = None,
    ) -> None:
        if crop_batch_size <= 0:
            raise ValueError("crop_batch_size must be positive")
        self.vbench_root = Path(vbench_root).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.clip_model = (
            Path(clip_model).expanduser().resolve() if clip_model is not None else None
        )
        if self.clip_model is not None and not self.clip_model.is_dir():
            raise FileNotFoundError(f"Local CLIP model does not exist: {self.clip_model}")
        self.device = device
        with _environment_variable("VBENCH2_CLIP_TEXT_MODEL", self.clip_model):
            (
                self.analyzer, self.detector, self.model_configs,
                self.detector_config, self.detector_weight,
            ) = _load_vbench(
                self.vbench_root, self.cache_dir, self.device, crop_batch_size
            )

    def score_video(
        self, video_path: Union[str, Path],
        entries: Sequence[Union[HumanAnomalyInput, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Read a video once and score all requested person-frame observations."""
        if not entries:
            raise ValueError("No person-frame entries were provided")
        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for item in entries:
            entry = item.to_dict() if isinstance(item, HumanAnomalyInput) else dict(item)
            grouped[int(entry["frame_index"])].append(entry)
        video = Path(video_path).expanduser().resolve()
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Video cannot be opened: {video}")
        pending = set(grouped)
        results: List[Dict[str, Any]] = []
        frame_index = 0
        try:
            while pending:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Video decode ended with pending frames: {sorted(pending)[:10]}"
                    )
                if frame_index in grouped:
                    for entry in grouped[frame_index]:
                        results.append(
                            _process_person(self.analyzer, self.detector, frame, entry)
                        )
                    pending.remove(frame_index)
                frame_index += 1
        finally:
            capture.release()
        return results

    def runtime_info(self) -> Dict[str, Any]:
        import mmcv
        import torch

        return {
            "torch_version": torch.__version__, "mmcv_version": mmcv.__version__,
            "device": self.device, "thresholds": OFFICIAL_THRESHOLDS,
            "detector_config": str(self.detector_config),
            "detector_weight": str(self.detector_weight),
            "classifier_models": self.model_configs,
        }

    def close(self) -> None:
        """Release model references and cached CUDA allocations."""
        self.analyzer = None
        self.detector = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
