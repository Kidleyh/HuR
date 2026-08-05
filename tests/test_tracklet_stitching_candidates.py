"""Tests for candidate hard gates and raw detection bridge evidence."""

from astrolabe.scorers.video.person_tracking.schemas import FrameDetections, RawDetection, TrackedDetection
from astrolabe.scorers.video.tracklet_stitching.candidates import raw_bridge_score, score_candidate
from astrolabe.scorers.video.tracklet_stitching.schemas import StitchingConfig, Tracklet


def tracked(track_id, cx, width=0.1, height=0.2):
    return TrackedDetection.from_xyxy(
        track_id=track_id, class_id=0, class_name="person", confidence=0.9,
        bbox_xyxy=[(cx-width/2)*1000, 400, (cx+width/2)*1000, 400+height*1000],
        image_width=1000, image_height=1000,
    )


def raw(index, cx, width=0.1, height=0.2):
    return RawDetection.from_xyxy(
        class_id=0, class_name="person", confidence=0.4,
        bbox_xyxy=[(cx-width/2)*1000, 400, (cx+width/2)*1000, 400+height*1000],
        image_width=1000, image_height=1000, detection_index=index,
    )


def tl(track_id, frames, centers, width=0.1, height=0.2):
    detections = [tracked(track_id, c, width, height) for c in centers]
    return Tracklet(track_id, frames, detections, frames[0], frames[-1])


def test_hard_rejections_and_normal_candidate():
    config = StitchingConfig()
    first = tl(1, [0, 1, 2], [0.2, 0.25, 0.3])
    overlap = tl(2, [2, 3], [0.35, 0.4])
    assert "time_overlap" in score_candidate(first, overlap, {}, config).rejection_reasons
    far_time = tl(3, [10], [0.7])
    assert "gap_too_large" in score_candidate(first, far_time, {}, config).rejection_reasons
    far_center = tl(4, [4], [0.9])
    assert "center_distance_too_large" in score_candidate(first, far_center, {}, config).rejection_reasons
    huge = tl(5, [4], [0.4], width=0.5, height=0.8)
    assert "area_ratio_change_too_large" in score_candidate(first, huge, {}, config).rejection_reasons
    wide = tl(6, [4], [0.4], width=0.3, height=0.1)
    assert "aspect_ratio_change_too_large" in score_candidate(first, wide, {}, config).rejection_reasons
    normal = tl(7, [4], [0.4])
    assert not score_candidate(first, normal, {}, config).rejection_reasons


def test_raw_bridge_uses_best_compatible_detection():
    config = StitchingConfig()
    first = tl(1, [1, 2], [0.2, 0.3])
    second = tl(3, [5, 6], [0.6, 0.7])
    frames = {
        3: FrameDetections(3, 0.3, [raw(0, 0.4), raw(1, 0.9)], []),
        4: FrameDetections(4, 0.4, [raw(0, 0.5)], []),
    }
    score, coverage, compatibility, matches = raw_bridge_score(first, second, frames, config)
    assert coverage == 1.0
    assert score > 0.8 and compatibility > 0.6
    assert [item.raw_detection_index for item in matches] == [0, 0]
    missing_score = raw_bridge_score(first, second, {}, config)[0]
    assert missing_score == 0.0
    far_score = raw_bridge_score(
        first, second, {3: FrameDetections(3, 0.3, [raw(0, 0.95)], [])}, config
    )[0]
    assert far_score == 0.0


def test_zero_gap_raw_bridge_is_one():
    first, second = tl(1, [0], [0.2]), tl(2, [1], [0.2])
    score, coverage, compatibility, matches = raw_bridge_score(
        first, second, {}, StitchingConfig()
    )
    assert (score, coverage, compatibility, matches) == (1.0, 1.0, 1.0, [])
