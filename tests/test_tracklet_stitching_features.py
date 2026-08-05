"""Tests for normalized motion extrapolation and box geometry."""

from astrolabe.scorers.video.person_tracking.schemas import TrackedDetection
from astrolabe.scorers.video.tracklet_stitching.features import predict_tracklet_bbox
from astrolabe.scorers.video.tracklet_stitching.schemas import Tracklet


def detection(track_id: int, cx: float, cy: float = 0.5, width: float = 0.1, height: float = 0.2):
    return TrackedDetection.from_xyxy(
        track_id=track_id, class_id=0, class_name="person", confidence=0.9,
        bbox_xyxy=[(cx-width/2)*1000, (cy-height/2)*1000,
                   (cx+width/2)*1000, (cy+height/2)*1000],
        image_width=1000, image_height=1000,
    )


def tracklet(track_id: int, frames, centers):
    items = [detection(track_id, center) for center in centers]
    return Tracklet(track_id, list(frames), items, frames[0], frames[-1])


def center(box):
    return (box[0] + box[2]) / 2


def test_multi_point_linear_prediction():
    box = predict_tracklet_bbox(tracklet(1, [0, 1, 2], [0.2, 0.3, 0.4]), 4, 5)
    assert box is not None
    assert center(box) == pytest.approx(0.6)


def test_two_point_regression():
    box = predict_tracklet_bbox(tracklet(1, [3, 5], [0.2, 0.4]), 7, 5)
    assert box is not None
    assert center(box) == pytest.approx(0.6)


def test_single_frame_uses_zero_velocity():
    box = predict_tracklet_bbox(tracklet(1, [2], [0.4]), 9, 5)
    assert box is not None
    assert center(box) == pytest.approx(0.4)


def test_prediction_is_clipped():
    box = predict_tracklet_bbox(tracklet(1, [0, 1], [0.8, 0.9]), 2, 5)
    assert box is not None
    assert 0 <= box[0] < box[2] <= 1


def test_completely_outside_prediction_is_invalid():
    assert predict_tracklet_bbox(tracklet(1, [0, 1], [0.8, 0.9]), 10, 5) is None


import pytest
