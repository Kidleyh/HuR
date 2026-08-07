from types import SimpleNamespace

import cv2
import numpy as np

from astrolabe.scorers.video.human_temporal.engine import HumanTemporalEngine
from astrolabe.scorers.video.human_temporal.metrics import BODY_JOINT_NAMES
from astrolabe.scorers.video.human_temporal.schema import HumanTemporalConfig
from astrolabe.scorers.video.human_reward.visualization import _draw_person


def _video(path):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()


def test_frame_batches_existing_person_boxes_and_maps_results(tmp_path):
    video = tmp_path / "people.mp4"
    _video(video)
    pose_config, checkpoint = tmp_path / "pose.py", tmp_path / "pose.pth"
    pose_config.write_text("config")
    checkpoint.write_bytes(b"checkpoint")
    config = HumanTemporalConfig(pose_config, checkpoint)
    calls = []

    def load_model(config_path, checkpoint_path, device):
        calls.append(("load", config_path, checkpoint_path, device))
        return SimpleNamespace(dataset_meta={
            "keypoint_name2id": {
                name: index for index, name in enumerate(BODY_JOINT_NAMES)
            }
        })

    def infer(model, frame, *, bboxes, bbox_format):
        calls.append(("infer", [list(box) for box in bboxes], bbox_format))
        results = []
        for box in bboxes:
            points = [[box[0] + index, box[1] + index] for index in range(12)]
            results.append({"keypoints_xy": points, "keypoint_scores": [0.9] * 12})
        return results

    persons = [
        {"logical_track_id": 1, "frames": [{
            "frame_index": 0, "bbox_xyxy": [15, 2, 30, 22],
        }], "temporal": {}},
        {"logical_track_id": 0, "frames": [{
            "frame_index": 0, "bbox_xyxy": [1, 2, 14, 22],
        }], "temporal": {}},
    ]
    engine = HumanTemporalEngine(
        config, device="cpu", model_loader=load_model, inference_fn=infer
    )
    engine.score_video(video, persons)

    assert [call[0] for call in calls] == ["load", "infer"]
    assert calls[1] == (
        "infer", [[1, 2, 14, 22], [15, 2, 30, 22]], "xyxy"
    )
    assert persons[1]["frames"][0]["human_pose"]["keypoints_xy"][0] == [1.0, 2.0]
    assert persons[0]["frames"][0]["human_pose"]["keypoints_xy"][0] == [15.0, 2.0]
    assert all(person["temporal"]["human"]["score"] is None for person in persons)
    assert not any(call[0] == "detector" for call in calls)


def test_visualization_draws_pose_and_metrics_without_changing_anomaly_state():
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    keypoints = [[40 + index * 2, 60 + index * 2] for index in range(12)]
    item = {
        "bbox_xyxy": [20, 30, 180, 170],
        "human": {"scored": True, "abnormal": False},
        "faces": [], "hands": [], "person_abnormal": False,
        "human_pose": {
            "keypoints_xy": keypoints, "keypoint_scores": [0.9] * 12,
        },
    }
    temporal = {
        "keypoint_threshold": 0.3,
        "keypoint_name_to_index": {
            name: index for index, name in enumerate(BODY_JOINT_NAMES)
        },
    }
    _draw_person(
        frame, 0, item, temporal,
        {"bone_length_jump": 0.1, "joint_acceleration": 0.2},
    )
    assert np.count_nonzero(frame) > 0
