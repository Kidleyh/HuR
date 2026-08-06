import pytest

from astrolabe.scorers.video.human_anomaly.schema import HumanAnomalyInput
from astrolabe.scorers.video.human_anomaly.validation import validate_worker_results


def entry(frame_index, logical_track_id):
    return HumanAnomalyInput(
        frame_index, logical_track_id, logical_track_id + 10,
        [0.0, 0.0, 10.0, 20.0], 0.9,
    )


def result(frame_index, logical_track_id, scored=True):
    return {
        "frame_index": frame_index,
        "logical_track_id": logical_track_id,
        "human": {"scored": scored},
    }


def test_worker_results_accept_exact_key_set():
    validate_worker_results([entry(0, 1), entry(2, 3)], [result(2, 3), result(0, 1)])


def test_worker_results_reject_missing_key():
    with pytest.raises(ValueError, match="missing"):
        validate_worker_results([entry(0, 1), entry(2, 3)], [result(0, 1)])


def test_worker_results_reject_duplicate_key():
    with pytest.raises(ValueError, match="duplicate"):
        validate_worker_results(
            [entry(0, 1), entry(2, 3)], [result(0, 1), result(0, 1)]
        )


def test_worker_results_reject_extra_key():
    with pytest.raises(ValueError, match="extra"):
        validate_worker_results([entry(0, 1)], [result(0, 1), result(2, 3)])


@pytest.mark.parametrize(
    "bad_result",
    [[], {"frame_index": "0", "logical_track_id": 1, "human": {"scored": True}},
     {"frame_index": 0, "logical_track_id": None, "human": {"scored": True}}],
)
def test_worker_results_reject_invalid_records(bad_result):
    with pytest.raises(ValueError):
        validate_worker_results([entry(0, 1)], [bad_result])


def test_worker_results_reject_all_unscored():
    with pytest.raises(ValueError, match="Worker produced no scored person frames"):
        validate_worker_results([entry(0, 1)], [result(0, 1, scored=False)])


def test_worker_results_reject_empty_manifest():
    with pytest.raises(ValueError, match="No valid person-frame entries"):
        validate_worker_results([], [])
