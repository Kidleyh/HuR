"""Tests that raw detections survive when ByteTrack does not associate them."""

import pytest

from astrolabe.scorers.video.person_tracking.schemas import (
    FrameDetections,
    RawDetection,
    TrackedDetection,
)
from astrolabe.scorers.video.person_tracking.statistics import compute_detection_summary


def raw(index: int, confidence: float) -> RawDetection:
    return RawDetection.from_xyxy(
        class_id=0,
        class_name="person",
        confidence=confidence,
        bbox_xyxy=[10.0, 10.0, 30.0, 40.0],
        image_width=100,
        image_height=100,
        detection_index=index,
    )


def tracked(source_index: int) -> TrackedDetection:
    return TrackedDetection.from_xyxy(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox_xyxy=[10.0, 10.0, 30.0, 40.0],
        image_width=100,
        image_height=100,
        source_detection_index=source_index,
    )


def test_raw_and_tracked_detection_summary_are_separate() -> None:
    frames = [
        FrameDetections(0, 0.0, [raw(0, 0.92)], [tracked(0)]),
        FrameDetections(1, 0.1, [raw(0, 0.18)], []),
        FrameDetections(2, 0.2, [], []),
    ]
    assert frames[1].raw_detections[0].confidence == pytest.approx(0.18)
    assert frames[1].tracked_detections == []
    summary = compute_detection_summary(frames, total_video_frames=3)
    assert summary.frames_with_raw_person == 2
    assert summary.frames_with_tracked_person == 1
    assert summary.raw_person_frame_coverage == pytest.approx(2 / 3)
    assert summary.tracked_person_frame_coverage == pytest.approx(1 / 3)
    assert summary.total_raw_detections == 2
    assert summary.total_tracked_detections == 1
    assert summary.untracked_raw_detections == 1
    assert summary.mean_raw_confidence == pytest.approx(0.55)
    assert summary.median_raw_confidence == pytest.approx(0.55)
    assert summary.min_raw_confidence == pytest.approx(0.18)
