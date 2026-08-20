from astrolabe.scorers.video.human_reward.pair_evaluation import (
    build_pair_frame_evaluation,
    build_pair_score_summary,
    extract_video_frame_metrics,
)


def _frame(
    frame_index,
    *,
    human_probability,
    face_probabilities=(),
    hand_probabilities=(),
    person_abnormal=False,
):
    return {
        "frame_index": frame_index,
        "source_track_id": 7,
        "human": {
            "scored": human_probability is not None,
            "abnormal_probability": human_probability,
        },
        "faces": [
            {"abnormal_probability": probability}
            for probability in face_probabilities
        ],
        "hands": [
            {"abnormal_probability": probability}
            for probability in hand_probabilities
        ],
        "person_abnormal": person_abnormal,
        "failure_reason": None,
    }


def _result(frames):
    return {
        "valid": True,
        "reason": None,
        "persons": [{"logical_track_id": 2, "frames": frames}],
    }


def _paired(gt, render):
    return {
        "pairs": [{
            "name": "pair-a",
            "positive": {"result": gt},
            "negative": {"result": render},
        }]
    }


def test_extracts_each_frame_metric_without_treating_missing_parts_as_anomaly():
    metrics = extract_video_frame_metrics(_result([
        _frame(2, human_probability=0.2),
        _frame(
            1,
            human_probability=0.1,
            face_probabilities=(0.05, 0.3),
            hand_probabilities=(0.4,),
            person_abnormal=True,
        ),
        _frame(3, human_probability=None),
    ]))

    assert [item["frame_index"] for item in metrics["frame_observations"]] == [1, 2, 3]
    assert metrics["observed_person_frames"] == 3
    assert metrics["scored_person_frames"] == 2
    assert metrics["failed_person_frames"] == 1
    assert metrics["face_scored_person_frames"] == 1
    assert metrics["hand_scored_person_frames"] == 1
    assert metrics["metrics"]["human_probability_quality"]["quality_score"] == 0.85
    assert metrics["metrics"]["face_probability_quality"]["quality_score"] == 0.7
    assert metrics["metrics"]["hand_probability_quality"]["quality_score"] == 0.6
    assert metrics["metrics"]["person_binary_quality"]["quality_score"] == 0.5
    assert metrics["metrics"]["combined_probability_quality"]["quality_score"] == 0.7


def test_pair_evaluation_reports_gt_wins_render_wins_ties_and_missing():
    gt = _result([
        _frame(0, human_probability=0.1, person_abnormal=False),
        _frame(1, human_probability=0.2, person_abnormal=False),
    ])
    render = _result([
        _frame(0, human_probability=0.7, person_abnormal=True),
        _frame(1, human_probability=0.8, person_abnormal=True),
    ])
    evaluation = build_pair_frame_evaluation(_paired(gt, render))

    comparison = evaluation["pairs"][0]["comparisons"]
    assert comparison["human_probability_quality"]["winner"] == "gt"
    assert comparison["human_probability_quality"]["correct"] is True
    assert comparison["person_binary_quality"]["winner"] == "gt"
    assert comparison["face_probability_quality"]["winner"] == "not_comparable"
    dataset = evaluation["dataset_metrics"]["human_probability_quality"]
    assert dataset["gt_win_count"] == 1
    assert dataset["strict_accuracy"] == 1.0
    assert dataset["gt_score_mean"] == 0.85
    assert dataset["render_score_mean"] == 0.25
    assert evaluation["temporal_enabled"] is False


def test_ties_are_not_counted_as_strictly_correct():
    result = _result([_frame(0, human_probability=0.2)])
    evaluation = build_pair_frame_evaluation(_paired(result, result))
    comparison = evaluation["pairs"][0]["comparisons"]["human_probability_quality"]
    assert comparison["winner"] == "tie"
    assert comparison["correct"] is False
    dataset = evaluation["dataset_metrics"]["human_probability_quality"]
    assert dataset["strict_accuracy"] == 0.0
    assert dataset["tie_aware_accuracy"] == 0.5


def test_score_summary_removes_only_frame_evidence():
    gt = _result([_frame(0, human_probability=0.1)])
    render = _result([_frame(0, human_probability=0.2)])
    evaluation = build_pair_frame_evaluation(_paired(gt, render))
    summary = build_pair_score_summary(evaluation)

    assert "frame_observations" not in summary["pairs"][0]["gt"]
    assert "frame_observations" not in summary["pairs"][0]["render"]
    assert summary["pairs"][0]["comparisons"] == evaluation["pairs"][0]["comparisons"]
    assert summary["dataset_metrics"] == evaluation["dataset_metrics"]
