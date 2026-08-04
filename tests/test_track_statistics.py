"""Unit tests for track aggregation."""

import pytest

from astrolabe.scorers.video.person_tracking.schemas import Detection, FrameDetections
from astrolabe.scorers.video.person_tracking.statistics import compute_track_statistics


def observation(track_id: int, confidence: float, area_ratio: float) -> Detection:
    return Detection(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=confidence,
        bbox_xyxy=[10.0, 10.0, 20.0, 20.0],
        bbox_xywh=[10.0, 10.0, 10.0, 10.0],
        bbox_xyxy_normalized=[0.1, 0.1, 0.2, 0.2],
        bbox_area_ratio=area_ratio,
    )


def test_track_statistics_with_missing_gaps() -> None:
    indices = [0, 1, 2, 5, 6, 9]
    confidences = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    areas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    frames = [
        FrameDetections(
            frame_index=index,
            timestamp_sec=index / 10.0,
            detections=[observation(1, confidence, area)],
        )
        for index, confidence, area in zip(indices, confidences, areas)
    ]
    track = compute_track_statistics(frames, total_video_frames=10)[0]
    assert track.track_id == 1
    assert track.start_frame == 0
    assert track.end_frame == 9
    assert track.num_observed_frames == 6
    assert track.global_coverage == pytest.approx(0.6)
    assert track.span_coverage == pytest.approx(0.6)
    assert track.max_missing_gap == 2
    assert track.mean_confidence == pytest.approx(0.75)
    assert track.median_confidence == pytest.approx(0.75)
    assert track.mean_bbox_area_ratio == pytest.approx(0.35)
    assert track.median_bbox_area_ratio == pytest.approx(0.35)


def test_empty_tracks() -> None:
    frames = [FrameDetections(frame_index=0, timestamp_sec=0.0)]
    assert compute_track_statistics(frames, total_video_frames=1) == []


def test_duplicate_identity_in_frame_uses_highest_confidence() -> None:
    frame = FrameDetections(
        frame_index=0,
        timestamp_sec=0.0,
        detections=[observation(2, 0.4, 0.2), observation(2, 0.8, 0.4)],
    )
    track = compute_track_statistics([frame], total_video_frames=1)[0]
    assert track.num_observed_frames == 1
    assert track.mean_confidence == pytest.approx(0.8)
