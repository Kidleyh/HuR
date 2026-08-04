"""Optional real-model integration test for person tracking."""

import json
import os
from pathlib import Path

import cv2
import pytest
import torch

from astrolabe.scorers.video.person_tracking.tracker import YOLOByteTrackPersonTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path.resolve() for path in paths if path.is_file()), None)


@pytest.mark.integration
def test_real_video_tracking(tmp_path: Path) -> None:
    gvhmr_root = Path(os.environ["GVHMR_ROOT"]).expanduser() if os.environ.get("GVHMR_ROOT") else None
    weight_candidates = []
    video_candidates = []
    if gvhmr_root:
        weight_candidates.append(gvhmr_root / "inputs/checkpoints/yolo/yolov8x.pt")
        video_candidates.append(gvhmr_root / "docs/example_video/tennis.mp4")
    weight_candidates.append(PROJECT_ROOT / "checkpoints/yolo/yolov8x.pt")
    video_candidates.extend(
        [PROJECT_ROOT / "docs/example_video/tennis.mp4", PROJECT_ROOT / "tests/assets/person_short.mp4"]
    )
    weights = _first_existing(weight_candidates)
    video = _first_existing(video_candidates)
    if weights is None:
        pytest.skip("YOLOv8x weights not found")
    if video is None:
        pytest.skip("integration test video not found")
    require_cuda = os.environ.get("PERSON_TRACKING_TEST_CUDA") == "1"
    if require_cuda and not torch.cuda.is_available():
        pytest.skip("integration test explicitly requires CUDA, but CUDA is unavailable")
    device = "0" if torch.cuda.is_available() else "cpu"
    tracker = YOLOByteTrackPersonTracker(
        str(weights),
        str(PROJECT_ROOT / "configs/bytetrack_person.yaml"),
        device=device,
        half=torch.cuda.is_available(),
    )
    output_dir = tmp_path / video.stem
    result = tracker.track_video(str(video), str(output_dir), save_visualization=True)

    for filename in ["detections.jsonl", "detections.csv", "tracks_summary.json", "tracked.mp4"]:
        assert (output_dir / filename).is_file()
    assert result.processing["processed_frames"] > 0
    lines = (output_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == result.processing["processed_frames"]
    for line in lines:
        frame = json.loads(line)
        for detection in frame["detections"]:
            assert detection["class_name"] == "person"
            assert isinstance(detection["track_id"], int)
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            assert 0 <= x1 < x2 <= result.video.width
            assert 0 <= y1 < y2 <= result.video.height

    capture = cv2.VideoCapture(str(output_dir / "tracked.mp4"))
    assert capture.isOpened()
    visualized_frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        visualized_frames += 1
    capture.release()
    assert abs(visualized_frames - result.processing["processed_frames"]) <= 1
