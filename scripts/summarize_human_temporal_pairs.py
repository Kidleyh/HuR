#!/usr/bin/env python3
"""Summarize person-level Body, Head and Hand Temporal paired metrics."""

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

TEMPORAL_METRICS = {
    "human": ("bone_length_jump_p90", "joint_acceleration_p90"),
    "head": ("face_shape_jump_p90", "head_motion_acceleration_p90"),
    "hand": ("bone_length_jump_p90", "joint_acceleration_p90"),
    "human_3d": (
        "joint_acceleration_p90",
        "joint_jerk_p90",
        "root_acceleration_p90",
    ),
}


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def extract_video_temporal(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Average each temporal family over its valid persons in one video."""
    persons: List[Dict[str, Any]] = []
    for person in result.get("persons", []):
        all_temporal = person.get("temporal", {})
        temporal = all_temporal.get("human", {})
        metrics = temporal.get("metrics", {}) if temporal.get("valid") is True else {}
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
        families: Dict[str, Any] = {}
        for family, names in TEMPORAL_METRICS.items():
            family_result = all_temporal.get(family, {})
            family_metrics = family_result.get("metrics", {})
            families[family] = {
                "valid": family_result.get("valid") is True,
                "metrics": {
                    name: _finite_number(family_metrics.get(name))
                    for name in names
                },
            }
        persons.append({
            "logical_track_id": person.get("logical_track_id"), **values,
            "temporal": families,
        })
    video = {
        name: (
            statistics.fmean(valid)
            if (valid := [p[name] for p in persons if p[name] is not None])
            else None
        )
        for name in METRIC_NAMES
    }
    family_video = {}
    for family, names in TEMPORAL_METRICS.items():
        family_video[family] = {
            name: (
                statistics.fmean(valid)
                if (valid := [
                    item["temporal"][family]["metrics"][name]
                    for item in persons
                    if item["temporal"][family]["valid"]
                    and item["temporal"][family]["metrics"][name] is not None
                ]) else None
            )
            for name in names
        }
    return {
        "valid_person_count": sum(
            item["temporal"]["human"]["valid"] for item in persons
        ),
        "metrics": video,
        "temporal": family_video,
        "persons": persons,
    }


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

    temporal_comparisons: Dict[str, Any] = {}
    for family, names in TEMPORAL_METRICS.items():
        temporal_comparisons[family] = {}
        for metric in names:
            gt_values = [item["gt"]["temporal"][family][metric] for item in details]
            render_values = [
                item["render"]["temporal"][family][metric] for item in details
            ]
            comparable = [
                (item["name"], gt, render)
                for item, gt, render in zip(details, gt_values, render_values)
                if gt is not None and render is not None
            ]
            ranked = sorted(
                ({"name": name, "difference": render - gt}
                 for name, gt, render in comparable),
                key=lambda item: (-item["difference"], str(item["name"])),
            )
            greater = sum(render > gt for _, gt, render in comparable)
            temporal_comparisons[family][metric] = {
                "gt": _distribution(gt_values),
                "render": _distribution(render_values),
                "comparable_pair_count": len(comparable),
                "render_greater_than_gt_count": greater,
                "render_greater_than_gt_ratio": (
                    greater / len(comparable) if comparable else None
                ),
                "largest_render_minus_gt_pairs": ranked[:10],
            }

    return {
        "schema_version": "1.1",
        "pair_count": len(details),
        "temporal": temporal_comparisons,
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
