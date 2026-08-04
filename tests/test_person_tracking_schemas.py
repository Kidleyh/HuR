"""Unit tests for the tracking output schema."""

import json

import pytest

from astrolabe.scorers.video.person_tracking.schemas import (
    FrameDetections,
    RawDetection,
    TrackedDetection,
    VideoTrackingResult,
)


def make_detection(**overrides: object) -> TrackedDetection:
    values = {
        "track_id": 3,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "bbox_xyxy": [10.0, 20.0, 50.0, 80.0],
        "bbox_xywh": [10.0, 20.0, 40.0, 60.0],
        "bbox_xyxy_normalized": [0.1, 0.1, 0.5, 0.4],
        "bbox_area_ratio": 0.12,
    }
    values.update(overrides)
    return TrackedDetection(**values)  # type: ignore[arg-type]


def test_from_xyxy_clips_and_normalizes_bbox() -> None:
    detection = TrackedDetection.from_xyxy(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.8,
        bbox_xyxy=[-5.0, 10.0, 110.0, 60.0],
        image_width=100,
        image_height=50,
    )
    assert detection.bbox_xyxy == [0.0, 10.0, 100.0, 50.0]
    assert detection.bbox_xywh == [0.0, 10.0, 100.0, 40.0]
    assert detection.bbox_xyxy_normalized == [0.0, 0.2, 1.0, 1.0]
    assert detection.bbox_area_ratio == pytest.approx(0.8)


def test_json_serialization_and_empty_frame() -> None:
    frame = FrameDetections(frame_index=0, timestamp_sec=0.0)
    assert json.loads(json.dumps(frame.to_dict())) == {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "raw_detections": [],
        "tracked_detections": [],
    }
    populated = FrameDetections(
        frame_index=1, timestamp_sec=0.04, tracked_detections=[make_detection()]
    )
    assert json.loads(json.dumps(populated.to_dict()))["tracked_detections"][0]["track_id"] == 3


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        make_detection(confidence=confidence)


@pytest.mark.parametrize("track_id", [-1, 1.5, True])
def test_invalid_track_id(track_id: object) -> None:
    with pytest.raises(ValueError, match="track_id"):
        make_detection(track_id=track_id)


@pytest.mark.parametrize(
    "bbox",
    ([10.0, 20.0, 10.0, 80.0], [-1.0, 20.0, 30.0, 80.0], [10.0, 20.0, 30.0]),
)
def test_invalid_bbox(bbox: list[float]) -> None:
    with pytest.raises(ValueError, match="bbox_xyxy"):
        make_detection(bbox_xyxy=bbox)


def test_invalid_normalized_bbox() -> None:
    with pytest.raises(ValueError, match="normalized bbox"):
        make_detection(bbox_xyxy_normalized=[0.1, 0.2, 1.1, 0.9])


def test_raw_detection_has_index_but_no_track_id() -> None:
    raw = RawDetection.from_xyxy(
        class_id=0,
        class_name="person",
        confidence=0.18,
        bbox_xyxy=[1.0, 2.0, 10.0, 20.0],
        image_width=20,
        image_height=40,
        detection_index=0,
    )
    payload = raw.to_dict()
    assert payload["detection_index"] == 0
    assert "track_id" not in payload


@pytest.mark.parametrize("source_index", [None, 0, 4])
def test_tracked_source_detection_index(source_index: int | None) -> None:
    assert make_detection(source_detection_index=source_index).source_detection_index == source_index


def test_schema_version_is_1_1() -> None:
    assert VideoTrackingResult.__dataclass_fields__["schema_version"].default == "1.1"
