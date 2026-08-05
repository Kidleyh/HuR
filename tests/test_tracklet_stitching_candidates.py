"""Tests for candidate hard gates and raw detection bridge evidence."""

from astrolabe.scorers.video.person_tracking.schemas import FrameDetections, RawDetection, TrackedDetection
from astrolabe.scorers.video.tracklet_stitching.candidates import raw_bridge_score, score_candidate
from astrolabe.scorers.video.tracklet_stitching.schemas import StitchingConfig, Tracklet


def tracked(
    track_id, cx, width=0.1, height=0.2, source_detection_index=None
):
    return TrackedDetection.from_xyxy(
        track_id=track_id, class_id=0, class_name="person", confidence=0.9,
        bbox_xyxy=[(cx-width/2)*1000, 400, (cx+width/2)*1000, 400+height*1000],
        image_width=1000, image_height=1000,
        source_detection_index=source_detection_index,
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
    result = raw_bridge_score(first, second, frames, config)
    assert result.coverage == 1.0
    assert result.score > 0.8 and result.compatibility > 0.6
    assert [item.raw_detection_index for item in result.matches] == [0, 0]
    assert raw_bridge_score(first, second, {}, config).score == 0.0
    far_score = raw_bridge_score(
        first, second, {3: FrameDetections(3, 0.3, [raw(0, 0.95)], [])}, config
    ).score
    assert far_score == 0.0


def test_raw_bridge_excludes_associated_raw_by_default():
    first = tl(1, [1, 2], [0.2, 0.3])
    second = tl(3, [4], [0.5])
    frame = FrameDetections(
        3,
        0.3,
        [raw(0, 0.4)],
        [tracked(9, 0.4, source_detection_index=0)],
    )
    result = raw_bridge_score(first, second, {3: frame}, StitchingConfig())
    assert result.score == 0.0
    assert result.matches == []
    assert result.excluded_associated_count == 1


def test_raw_bridge_can_allow_associated_raw_for_ablation():
    first = tl(1, [1, 2], [0.2, 0.3])
    second = tl(3, [4], [0.5])
    frame = FrameDetections(
        3,
        0.3,
        [raw(0, 0.4)],
        [tracked(9, 0.4, source_detection_index=0)],
    )
    result = raw_bridge_score(
        first,
        second,
        {3: frame},
        StitchingConfig(raw_bridge_allow_associated_raw=True),
    )
    assert result.score > 0.0
    assert [match.raw_detection_index for match in result.matches] == [0]
    assert result.excluded_associated_count == 0


def test_raw_bridge_prefers_best_unassociated_raw():
    first = tl(1, [1, 2], [0.2, 0.3])
    second = tl(3, [4], [0.5])
    frame = FrameDetections(
        3,
        0.3,
        [raw(0, 0.4), raw(1, 0.42)],
        [tracked(9, 0.4, source_detection_index=0)],
    )
    result = raw_bridge_score(first, second, {3: frame}, StitchingConfig())
    assert [match.raw_detection_index for match in result.matches] == [1]
    assert result.excluded_associated_count == 1


def test_raw_bridge_counts_exclusions_across_gap_frames():
    first = tl(1, [1, 2], [0.2, 0.3])
    second = tl(3, [5], [0.6])
    frames = {
        3: FrameDetections(
            3,
            0.3,
            [raw(0, 0.4), raw(1, 0.41)],
            [
                tracked(8, 0.4, source_detection_index=0),
                tracked(9, 0.41, source_detection_index=1),
            ],
        ),
        4: FrameDetections(
            4,
            0.4,
            [raw(0, 0.5), raw(1, 0.52)],
            [
                tracked(10, 0.5, source_detection_index=0),
                tracked(11, 0.8, source_detection_index=None),
                tracked(12, 0.8, source_detection_index=99),
            ],
        ),
    }
    result = raw_bridge_score(first, second, frames, StitchingConfig())
    assert result.excluded_associated_count == 3
    assert [match.raw_detection_index for match in result.matches] == [1]


def test_zero_gap_raw_bridge_is_one():
    first, second = tl(1, [0], [0.2]), tl(2, [1], [0.2])
    result = raw_bridge_score(
        first, second, {}, StitchingConfig()
    )
    assert result.score == 1.0
    assert result.coverage == 1.0
    assert result.compatibility == 1.0
    assert result.matches == []
    assert result.excluded_associated_count == 0


def test_raw_bridge_allow_associated_raw_requires_bool():
    with pytest.raises(ValueError, match="must be a bool"):
        StitchingConfig(raw_bridge_allow_associated_raw="false")


import pytest
