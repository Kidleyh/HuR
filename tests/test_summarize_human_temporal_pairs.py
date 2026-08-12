from scripts.summarize_human_temporal_pairs import summarize_pairs


def _result(bone, motion, logical_id=0):
    return {"persons": [{
        "logical_track_id": logical_id,
        "temporal": {"human": {
            "valid": True,
            "mean_keypoint_coverage": 0.8,
            "valid_structure_pairs": 5,
            "valid_motion_triplets": 4,
            "metrics": {
                "bone_length_jump_mean": bone / 2,
                "bone_length_jump_p90": bone,
                "bone_length_jump_max": bone * 2,
                "joint_acceleration_mean": motion / 2,
                "joint_acceleration_p90": motion,
                "joint_acceleration_max": motion * 2,
            },
        }},
    }]}


def test_summary_averages_people_then_compares_pairs():
    gt = _result(0.1, 0.2)
    gt["persons"].extend(_result(0.3, 0.4, 1)["persons"])
    data = {"pairs": [
        {"name": "render_worse", "positive": {"result": gt},
         "negative": {"result": _result(0.4, 0.5)}},
        {"name": "render_better", "positive": {"result": _result(0.3, 0.6)},
         "negative": {"result": _result(0.2, 0.1)}},
    ]}
    summary = summarize_pairs(data)
    assert summary["pair_count"] == 2
    assert summary["pairs"][0]["gt"]["metrics"]["bone_length_jump_p90"] == 0.2
    assert summary["bone_p90"]["render_greater_than_gt_ratio"] == 0.5
    assert summary["motion_p90"]["render_greater_than_gt_ratio"] == 0.5
    assert summary["largest_render_minus_gt_bone_p90"][0]["name"] == "render_worse"
    assert summary["largest_render_minus_gt_motion_p90"][0]["name"] == "render_worse"
