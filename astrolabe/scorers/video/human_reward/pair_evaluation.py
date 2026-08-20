"""Paired GT/render evaluation for frame-level Human Anomaly outputs."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence


METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "person_binary_quality": {
        "source": "person_abnormal",
        "description": "1 - abnormal person-frame rate using official thresholds",
    },
    "human_probability_quality": {
        "source": "human.abnormal_probability",
        "description": "1 - mean Human classifier abnormal probability",
    },
    "face_probability_quality": {
        "source": "max(faces[].abnormal_probability) per person-frame",
        "description": "1 - mean worst-face abnormal probability on detected faces",
    },
    "hand_probability_quality": {
        "source": "max(hands[].abnormal_probability) per person-frame",
        "description": "1 - mean worst-hand abnormal probability on detected hands",
    },
    "combined_probability_quality": {
        "source": "max(human, detected faces, detected hands) per person-frame",
        "description": "1 - mean worst available component abnormal probability",
    },
}


def _finite_probability(value: Any) -> Optional[float]:
    """Return a finite probability or ``None`` for missing/invalid values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        return None
    return result


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    """Summarize raw abnormal values and expose a higher-is-better quality."""
    if not values:
        return {
            "observation_count": 0,
            "abnormal_mean": None,
            "abnormal_median": None,
            "abnormal_p90": None,
            "quality_score": None,
        }
    abnormal_mean = mean(values)
    return {
        "observation_count": len(values),
        "abnormal_mean": abnormal_mean,
        "abnormal_median": median(values),
        "abnormal_p90": _percentile(values, 90.0),
        "quality_score": 1.0 - abnormal_mean,
    }


def _component_probabilities(components: Any) -> List[float]:
    if not isinstance(components, list):
        return []
    values = []
    for component in components:
        if not isinstance(component, Mapping):
            continue
        value = _finite_probability(component.get("abnormal_probability"))
        if value is not None:
            values.append(value)
    return values


def extract_video_frame_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract auditable person-frame observations and video metric summaries.

    Face/hand metrics include only frames where that component was detected and
    scored. Missing face/hand detections are neutral and are never converted to
    either zero abnormality or an anomaly.
    """
    observations: List[Dict[str, Any]] = []
    human_values: List[float] = []
    face_values: List[float] = []
    hand_values: List[float] = []
    combined_values: List[float] = []
    person_binary_values: List[float] = []

    persons = result.get("persons", [])
    if not isinstance(persons, list):
        persons = []
    for person in persons:
        if not isinstance(person, Mapping):
            continue
        logical_track_id = person.get("logical_track_id")
        frames = person.get("frames", [])
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            human = frame.get("human", {})
            human = human if isinstance(human, Mapping) else {}
            human_scored = human.get("scored") is True
            human_probability = (
                _finite_probability(human.get("abnormal_probability"))
                if human_scored else None
            )
            faces = _component_probabilities(frame.get("faces"))
            hands = _component_probabilities(frame.get("hands"))
            face_probability = max(faces) if faces else None
            hand_probability = max(hands) if hands else None
            available = [
                value for value in (
                    human_probability, face_probability, hand_probability
                ) if value is not None
            ]
            combined_probability = max(available) if human_scored and available else None
            person_abnormal = (
                bool(frame.get("person_abnormal")) if human_scored else None
            )

            if human_probability is not None:
                human_values.append(human_probability)
                person_binary_values.append(float(person_abnormal))
            if face_probability is not None:
                face_values.append(face_probability)
            if hand_probability is not None:
                hand_values.append(hand_probability)
            if combined_probability is not None:
                combined_values.append(combined_probability)

            observations.append({
                "frame_index": frame.get("frame_index"),
                "logical_track_id": logical_track_id,
                "source_track_id": frame.get("source_track_id"),
                "human_scored": human_scored,
                "human_abnormal_probability": human_probability,
                "face_count": len(faces),
                "face_max_abnormal_probability": face_probability,
                "hand_count": len(hands),
                "hand_max_abnormal_probability": hand_probability,
                "combined_max_abnormal_probability": combined_probability,
                "person_abnormal": person_abnormal,
                "failure_reason": frame.get("failure_reason"),
            })

    observations.sort(key=lambda item: (
        int(item["frame_index"]), int(item["logical_track_id"])
    ))
    scored_count = len(human_values)
    return {
        "valid": result.get("valid") is True,
        "reason": result.get("reason"),
        "observed_person_frames": len(observations),
        "scored_person_frames": scored_count,
        "failed_person_frames": len(observations) - scored_count,
        "face_scored_person_frames": len(face_values),
        "hand_scored_person_frames": len(hand_values),
        "metrics": {
            "person_binary_quality": _distribution(person_binary_values),
            "human_probability_quality": _distribution(human_values),
            "face_probability_quality": _distribution(face_values),
            "hand_probability_quality": _distribution(hand_values),
            "combined_probability_quality": _distribution(combined_values),
        },
        "frame_observations": observations,
    }


def compare_pair_metrics(
    gt: Mapping[str, Any], render: Mapping[str, Any], *, tie_epsilon: float = 1e-12
) -> Dict[str, Any]:
    """Compare higher-is-better video quality values for one labeled pair."""
    comparisons = {}
    gt_metrics = gt.get("metrics", {})
    render_metrics = render.get("metrics", {})
    for metric_name in METRIC_DEFINITIONS:
        gt_score = gt_metrics.get(metric_name, {}).get("quality_score")
        render_score = render_metrics.get(metric_name, {}).get("quality_score")
        if gt_score is None or render_score is None:
            winner = "not_comparable"
            difference = None
            correct = None
        else:
            difference = float(gt_score) - float(render_score)
            if abs(difference) <= tie_epsilon:
                winner = "tie"
                correct = False
            elif difference > 0:
                winner = "gt"
                correct = True
            else:
                winner = "render"
                correct = False
        comparisons[metric_name] = {
            "gt_score": gt_score,
            "render_score": render_score,
            "gt_minus_render": difference,
            "winner": winner,
            "correct": correct,
        }
    return comparisons


def build_pair_frame_evaluation(
    paired_result: Mapping[str, Any], *, tie_epsilon: float = 1e-12
) -> Dict[str, Any]:
    """Build per-frame evidence, pair decisions, and dataset-level accuracy."""
    pair_details = []
    for pair in paired_result.get("pairs", []):
        gt = extract_video_frame_metrics(pair["positive"]["result"])
        render = extract_video_frame_metrics(pair["negative"]["result"])
        pair_details.append({
            "name": pair["name"],
            "gt": gt,
            "render": render,
            "comparisons": compare_pair_metrics(
                gt, render, tie_epsilon=tie_epsilon
            ),
        })

    dataset_metrics = {}
    for metric_name, definition in METRIC_DEFINITIONS.items():
        decisions = [pair["comparisons"][metric_name] for pair in pair_details]
        comparable = [item for item in decisions if item["winner"] != "not_comparable"]
        gt_wins = sum(item["winner"] == "gt" for item in comparable)
        render_wins = sum(item["winner"] == "render" for item in comparable)
        ties = sum(item["winner"] == "tie" for item in comparable)
        differences = [
            item["gt_minus_render"] for item in comparable
            if item["gt_minus_render"] is not None
        ]
        gt_scores = [float(item["gt_score"]) for item in comparable]
        render_scores = [float(item["render_score"]) for item in comparable]
        dataset_metrics[metric_name] = {
            **definition,
            "direction": "higher_quality_score_is_better",
            "pair_count": len(decisions),
            "comparable_pair_count": len(comparable),
            "not_comparable_pair_count": len(decisions) - len(comparable),
            "gt_win_count": gt_wins,
            "render_win_count": render_wins,
            "tie_count": ties,
            "gt_win_rate": gt_wins / len(comparable) if comparable else None,
            "strict_accuracy": gt_wins / len(comparable) if comparable else None,
            "tie_aware_accuracy": (
                (gt_wins + 0.5 * ties) / len(comparable) if comparable else None
            ),
            "gt_score_mean": mean(gt_scores) if gt_scores else None,
            "gt_score_median": median(gt_scores) if gt_scores else None,
            "render_score_mean": mean(render_scores) if render_scores else None,
            "render_score_median": median(render_scores) if render_scores else None,
            "mean_gt_minus_render": mean(differences) if differences else None,
            "median_gt_minus_render": median(differences) if differences else None,
        }
    return {
        "schema_version": "1.0",
        "evaluation_target": "GT quality score should exceed render quality score",
        "temporal_enabled": False,
        "tie_epsilon": tie_epsilon,
        "pair_count": len(pair_details),
        "metric_definitions": METRIC_DEFINITIONS,
        "dataset_metrics": dataset_metrics,
        "pairs": pair_details,
    }


def build_pair_score_summary(evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove person-frame evidence while retaining every pair decision."""
    pairs = []
    for pair in evaluation.get("pairs", []):
        pairs.append({
            "name": pair["name"],
            "gt": {
                key: value for key, value in pair["gt"].items()
                if key != "frame_observations"
            },
            "render": {
                key: value for key, value in pair["render"].items()
                if key != "frame_observations"
            },
            "comparisons": pair["comparisons"],
        })
    return {
        key: value for key, value in evaluation.items() if key != "pairs"
    } | {"pairs": pairs}
