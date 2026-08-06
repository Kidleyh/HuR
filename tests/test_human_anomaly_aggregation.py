import pytest

from astrolabe.scorers.video.human_anomaly.aggregation import aggregate_human_anomaly
from astrolabe.scorers.video.human_anomaly.schema import (
    OFFICIAL_THRESHOLDS, HumanAnomalyInput, classifier_result, person_is_abnormal,
)


def entry(frame, track, bbox=(0, 0, 20, 50)):
    return HumanAnomalyInput(frame, track, track + 10, list(bbox), 0.9)


def component(category, probability):
    return classifier_result([probability, 1 - probability], category)


def test_official_thresholds_and_missing_parts_semantics():
    for category, threshold in OFFICIAL_THRESHOLDS.items():
        assert not component(category, threshold)["abnormal"]
        assert component(category, threshold + 1e-4)["abnormal"]
    human = component("human", 0.1)
    assert not person_is_abnormal(human, [], [])
    hands = [component("hand", 0.1), component("hand", 0.9)]
    assert person_is_abnormal(human, [], hands)


def test_track_and_video_micro_macro_aggregation():
    entries = [entry(0, 0), entry(1, 0, (10, 10, 30, 60)), entry(0, 1)]
    results = [
        {"frame_index": 0, "logical_track_id": 0, "human": component("human", 0.1),
         "faces": [], "hands": [], "person_abnormal": False},
        {"frame_index": 1, "logical_track_id": 0, "human": component("human", 0.9),
         "faces": [component("face", 0.1)], "hands": [], "person_abnormal": True},
        {"frame_index": 0, "logical_track_id": 1, "human": component("human", 0.1),
         "faces": [], "hands": [component("hand", 0.9)], "person_abnormal": True},
    ]
    tracks, summary = aggregate_human_anomaly(entries, results, 100, 100)
    assert tracks[0]["anatomy_quality_score"] == pytest.approx(0.5)
    assert tracks[0]["face_detection_coverage"] == pytest.approx(0.5)
    assert tracks[0]["hand_detection_coverage"] == 0.0
    assert tracks[0]["boundary_truncation_rate"] == pytest.approx(0.5)
    assert summary["video_micro_score"] == pytest.approx(1 / 3)
    assert summary["video_macro_score"] == pytest.approx(0.25)


def test_zero_scored_frames_has_no_division_error():
    tracks, summary = aggregate_human_anomaly([entry(0, 0)], [], 100, 100)
    assert tracks[0]["scored_frames"] == 0
    assert tracks[0]["anatomy_quality_score"] is None
    assert summary["video_micro_score"] is None
    assert summary["failed_person_frames"] == 1
