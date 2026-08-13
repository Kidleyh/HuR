from pathlib import Path

from astrolabe.scorers.video.human_temporal.part_engines import (
    HandTemporalEngine, HeadTemporalEngine,
)
from astrolabe.scorers.video.human_temporal.schema import HandTemporalConfig, PartTemporalConfig
import cv2
import numpy as np


def _resources(tmp_path):
    cfg, ckpt = tmp_path / "pose.py", tmp_path / "pose.pth"
    cfg.write_text("config")
    ckpt.write_bytes(b"weights")
    return cfg, ckpt


def test_face_and_hand_models_are_lazy_loaded_once(tmp_path):
    cfg, ckpt = _resources(tmp_path)
    loads = []

    class Model:
        dataset_meta = {}

    def loader(config, checkpoint, device):
        loads.append((config, checkpoint, device))
        return Model()

    head = HeadTemporalEngine(
        PartTemporalConfig(cfg, ckpt), model_loader=loader, inference_fn=lambda *a, **k: []
    )
    hand = HandTemporalEngine(
        HandTemporalConfig(cfg, ckpt), model_loader=loader, inference_fn=lambda *a, **k: []
    )
    assert len(loads) == 2
    head.close()
    hand.close()


def test_head_engine_batches_existing_face_boxes_and_maps_people(tmp_path):
    cfg, ckpt = _resources(tmp_path)
    video = tmp_path / "people.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    persons = [
        {"logical_track_id": logical_id, "frames": [{
            "frame_index": 0, "faces": [{"bbox_xyxy": box, "detector_score": 0.9}],
            "hands": [], "bbox_xyxy": [0, 0, 32, 24],
        }], "temporal": {}}
        for logical_id, box in ((0, [1, 2, 8, 10]), (1, [20, 2, 30, 12]))
    ]
    calls = []

    def inference(model, image, *, bboxes, bbox_format):
        calls.append((list(bboxes), bbox_format))
        return [{
            "keypoints_xy": [[float(box[0]), float(box[1])] for _ in range(6)],
            "keypoint_scores": [0.99] * 6,
        } for box in bboxes]

    engine = HeadTemporalEngine(
        PartTemporalConfig(cfg, ckpt, min_valid_keypoints=5),
        model_loader=lambda *a, **k: object(), inference_fn=inference,
    )
    engine.score_video(video, persons)
    assert calls == [([[1, 2, 8, 10], [20, 2, 30, 12]], "xyxy")]
    assert persons[0]["frames"][0]["faces"][0]["face_pose"]["keypoints_xy"][0] == [1.0, 2.0]
    assert persons[1]["frames"][0]["faces"][0]["face_pose"]["keypoints_xy"][0] == [20.0, 2.0]
