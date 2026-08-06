#!/usr/bin/env python3
"""Run VBench Human/Face/Hand inference for HuR-provided person boxes."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2

HUR_ROOT = Path(__file__).resolve().parents[1]
if str(HUR_ROOT) not in sys.path:
    sys.path.insert(0, str(HUR_ROOT))

from astrolabe.scorers.video.human_anomaly.schema import (
    OFFICIAL_THRESHOLDS,
    classifier_result,
    person_is_abnormal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--runtime-info-json", required=True)
    parser.add_argument("--vbench-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-batch-size", type=int, default=128)
    return parser


def _is_cuda_failure(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, RuntimeError) and any(
        token in message for token in ("cuda", "cudnn", "out of memory", "device-side")
    )


class FaceHandDetector:
    """One YOLO-World instance, parameterized only for face and hand."""

    def __init__(self, config_path: Path, weight_path: Path, device: str) -> None:
        from mmengine.dataset import Compose
        from mmdet.apis import init_detector

        self.model = init_detector(str(config_path), str(weight_path), device=device)
        self.model.cfg.test_dataloader.dataset.pipeline[0].type = "mmdet.LoadImageFromNDArray"
        self.pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)
        self.texts = [["face"], ["hand"], [" "]]
        self.model.reparameterize(self.texts)

    def detect(self, image: Any, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        import torch

        data_info = dict(img_id=0, img=image, texts=self.texts)
        data_info = self.pipeline(data_info)
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
            {"bbox_xyxy": box.tolist(), "label": int(label), "detector_score": float(score)}
            for box, label, score in zip(boxes, labels, scores)
            if int(label) in (0, 1)
        ]


def _load_vbench(vbench_root: Path, cache_dir: Path, device: str, batch_size: int):
    vit_root = vbench_root / "vbench2/third_party/ViTDetector"
    yolo_root = vbench_root / "vbench2/third_party/YOLO-World"
    for path in (str(vit_root), str(yolo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    inference = importlib.import_module("inference")
    classifier_config = vit_root / "simmim_finetune__vit_base__img224__800ep.yaml"
    model_configs = {
        category: {
            "cfg_path": str(classifier_config),
            "weight_path": str(cache_dir / f"anomaly_detector/{category}.pth"),
        }
        for category in ("human", "face", "hand")
    }
    for item in [classifier_config] + [Path(value["weight_path"]) for value in model_configs.values()]:
        if not item.is_file():
            raise FileNotFoundError(f"Required VBench model/config is missing: {item}")
    analyzer = inference.Analyzer(
        model_configs, device=device, batch_size=batch_size,
        class_thresholds=OFFICIAL_THRESHOLDS,
    )
    detector_config = vbench_root / (
        "vbench2/third_party/YOLO-World/"
        "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    )
    detector_weight = cache_dir / (
        "YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
    )
    if not detector_config.is_file() or not detector_weight.is_file():
        raise FileNotFoundError("YOLO-World config or checkpoint is missing")
    detector = FaceHandDetector(detector_config, detector_weight, device)
    return analyzer, detector, model_configs, detector_config, detector_weight


def _part_result(
    analyzer: Any, category: str, frame: Any, bbox: Sequence[float],
    detector_score: float,
) -> Dict[str, Any]:
    crop = analyzer.smart_cut(frame, bbox)
    scores = analyzer.predict_batch(category, [crop])[0]
    result = classifier_result(scores, category)
    result.update({"bbox_xyxy": [float(value) for value in bbox],
                   "detector_score": float(detector_score)})
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
        "faces": [], "hands": [], "person_abnormal": False, "failure_reason": None,
        "failures": [],
    }

    def record_failure(stage: str, error: Exception) -> None:
        if _is_cuda_failure(error):
            raise error
        base["failures"].append({
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
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


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.crop_batch_size <= 0:
        raise ValueError("--crop-batch-size must be positive")
    video, manifest = Path(args.video).resolve(), Path(args.input_manifest).resolve()
    output, runtime_path = Path(args.output_jsonl).resolve(), Path(args.runtime_info_json).resolve()
    vbench_root = Path(args.vbench_root).resolve()
    cache_dir = Path(__import__("os").environ["VBENCH2_CACHE_DIR"]).resolve()
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["frame_index"])].append(entry)
    analyzer, detector, model_configs, detector_config, detector_weight = _load_vbench(
        vbench_root, cache_dir, args.device, args.crop_batch_size
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Video cannot be opened: {video}")
    pending = set(grouped)
    results = []
    frame_index = 0
    try:
        while pending:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video decode ended with pending frames: {sorted(pending)[:10]}")
            if frame_index in grouped:
                for entry in grouped[frame_index]:
                    results.append(_process_person(analyzer, detector, frame, entry))
                pending.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    import mmcv
    import torch
    runtime = {
        "torch_version": torch.__version__, "mmcv_version": mmcv.__version__,
        "device": args.device, "thresholds": OFFICIAL_THRESHOLDS,
        "detector_config": str(detector_config), "detector_weight": str(detector_weight),
        "classifier_models": model_configs,
    }
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
