#!/usr/bin/env python3
"""Summarize person-level Human Temporal metrics from paired reward JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

METRIC_NAMES = (
    "bone_length_jump_mean",
    "bone_length_jump_p90",
    "bone_length_jump_max",
    "joint_acceleration_mean",
    "joint_acceleration_p90",
    "joint_acceleration_max",
    "mean_keypoint_coverage",
    "valid_structure_pairs",
    "valid_motion_triplets",
)


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def extract_video_temporal(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Average each metric over valid persons in one video."""
    persons: List[Dict[str, Any]] = []
    for person in result.get("persons", []):
        temporal = person.get("temporal", {}).get("human", {})
        if temporal.get("valid") is not True:
            continue
        metrics = temporal.get("metrics", {})
        values = {
            "bone_length_jump_mean": _finite_number(
                metrics.get("bone_length_jump_mean")
            ),
            "bone_length_jump_p90": _finite_number(
                metrics.get("bone_length_jump_p90")
            ),
            "bone_length_jump_max": _finite_number(
                metrics.get("bone_length_jump_max")
            ),
            "joint_acceleration_mean": _finite_number(
                metrics.get("joint_acceleration_mean")
            ),
            "joint_acceleration_p90": _finite_number(
                metrics.get("joint_acceleration_p90")
            ),
            "joint_acceleration_max": _finite_number(
                metrics.get("joint_acceleration_max")
            ),
            "mean_keypoint_coverage": _finite_number(
                temporal.get("mean_keypoint_coverage")
            ),
            "valid_structure_pairs": _finite_number(
                temporal.get("valid_structure_pairs")
            ),
            "valid_motion_triplets": _finite_number(
                temporal.get("valid_motion_triplets")
            ),
        }
        persons.append({
            "logical_track_id": person.get("logical_track_id"), **values,
        })
    video = {
        name: (
            statistics.fmean(valid)
            if (valid := [p[name] for p in persons if p[name] is not None])
            else None
        )
        for name in METRIC_NAMES
    }
    return {"valid_person_count": len(persons), "metrics": video, "persons": persons}


def _distribution(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    valid = [float(value) for value in values if value is not None]
    return {
        "count": len(valid),
        "mean": statistics.fmean(valid) if valid else None,
        "median": statistics.median(valid) if valid else None,
    }


def summarize_pairs(data: Mapping[str, Any]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    for pair in data.get("pairs", []):
        gt = extract_video_temporal(pair.get("positive", {}).get("result", {}))
        render = extract_video_temporal(pair.get("negative", {}).get("result", {}))
        bone_gt = gt["metrics"]["bone_length_jump_p90"]
        bone_render = render["metrics"]["bone_length_jump_p90"]
        motion_gt = gt["metrics"]["joint_acceleration_p90"]
        motion_render = render["metrics"]["joint_acceleration_p90"]
        details.append({
            "name": pair.get("name"),
            "gt": gt,
            "render": render,
            "bone_p90_difference": (
                bone_render - bone_gt
                if bone_render is not None and bone_gt is not None else None
            ),
            "motion_p90_difference": (
                motion_render - motion_gt
                if motion_render is not None and motion_gt is not None else None
            ),
        })

    def comparison(metric: str) -> Dict[str, Any]:
        gt_values = [item["gt"]["metrics"][metric] for item in details]
        render_values = [item["render"]["metrics"][metric] for item in details]
        comparable = [
            (gt, render) for gt, render in zip(gt_values, render_values)
            if gt is not None and render is not None
        ]
        return {
            "gt": _distribution(gt_values),
            "render": _distribution(render_values),
            "comparable_pair_count": len(comparable),
            "render_greater_than_gt_count": sum(
                render > gt for gt, render in comparable
            ),
            "render_greater_than_gt_ratio": (
                sum(render > gt for gt, render in comparable) / len(comparable)
                if comparable else None
            ),
        }

    def largest(metric: str) -> List[Dict[str, Any]]:
        ranked = [
            {"name": item["name"], "difference": item[metric]}
            for item in details if item[metric] is not None
        ]
        ranked.sort(key=lambda item: (-item["difference"], str(item["name"])))
        return ranked[:10]

    return {
        "schema_version": "1.0",
        "pair_count": len(details),
        "bone_p90": comparison("bone_length_jump_p90"),
        "motion_p90": comparison("joint_acceleration_p90"),
        "largest_render_minus_gt_bone_p90": largest("bone_p90_difference"),
        "largest_render_minus_gt_motion_p90": largest("motion_p90_difference"),
        "pairs": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Paired Human Reward JSON must contain an object")
    summary = summarize_pairs(data)
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
