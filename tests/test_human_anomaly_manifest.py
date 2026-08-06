import json
from pathlib import Path

import cv2
import numpy as np

from astrolabe.scorers.video.human_anomaly.manifest import build_human_anomaly_manifest


def _video(path: Path, width=100, height=80):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    assert writer.isOpened()
    writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def test_manifest_clips_deduplicates_and_stably_sorts(tmp_path):
    video = tmp_path / "input.mp4"
    _video(video)
    stitching = tmp_path / "stitching"
    stitching.mkdir()
    frames = [
        {"frame_index": 0, "tracked_detections": [
            {"logical_track_id": 2, "track_id": 8, "bbox_xyxy": [-5, 2, 110, 70], "confidence": 0.8},
            {"logical_track_id": 1, "track_id": 4, "bbox_xyxy": [10, 10, 40, 60], "confidence": 0.7},
            {"logical_track_id": 1, "track_id": 5, "bbox_xyxy": [12, 12, 42, 62], "confidence": 0.9},
            {"logical_track_id": 3, "track_id": 9, "bbox_xyxy": [20, 20, 20, 50], "confidence": 0.9},
        ]}
    ]
    (stitching / "stitched_detections.jsonl").write_text(
        "\n".join(json.dumps(item) for item in frames) + "\n"
    )
    entries, failures, width, height = build_human_anomaly_manifest(video, stitching)
    assert (width, height) == (100, 80)
    assert [(item.frame_index, item.logical_track_id) for item in entries] == [(0, 1), (0, 2)]
    assert entries[0].source_track_id == 5
    assert entries[1].bbox_xyxy == [0.0, 2.0, 100.0, 70.0]
    assert len(failures) == 2
    assert any("duplicate" in item.failure_reason for item in failures)
    assert any("empty after clipping" in item.failure_reason for item in failures)
