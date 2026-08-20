from pathlib import Path

import numpy as np
import pytest

from astrolabe.scorers.video.human_temporal_3d.engine import GVHMRTemporalEngine
from astrolabe.scorers.video.human_temporal_3d.metrics import analyze_3d_temporal
from astrolabe.scorers.video.human_temporal_3d.schema import (
    GVHMRSequence,
    GVHMRTemporalConfig,
)


def _sequence(frames, joints, roots=None):
    joints = np.asarray(joints, dtype=float)
    if roots is None:
        roots = np.zeros((len(frames), 3), dtype=float)
    return GVHMRSequence(
        np.asarray(frames), joints, np.asarray(roots),
        {"body_pose": np.zeros((len(frames), 63)), "transl": roots},
    )


def test_static_3d_sequence_has_zero_derivatives():
    base = np.arange(18, dtype=float).reshape(6, 3)
    result = analyze_3d_temporal(
        _sequence([0, 1, 2, 3], [base] * 4),
        fps=10.0, total_observed_frames=4,
    )
    assert result["metrics"]["joint_velocity_max"] == 0.0
    assert result["metrics"]["joint_acceleration_max"] == 0.0
    assert result["metrics"]["joint_jerk_max"] == 0.0
    assert result["score"] is None


def test_root_motion_is_separate_from_root_relative_joint_motion():
    base = np.arange(18, dtype=float).reshape(6, 3) / 10
    roots = np.asarray([[0, 0, 0], [1, 0, 0], [4, 0, 0], [9, 0, 0]], float)
    joints = [base + root for root in roots]
    result = analyze_3d_temporal(
        _sequence([0, 1, 2, 3], joints, roots),
        fps=1.0, total_observed_frames=4,
    )
    assert result["metrics"]["joint_acceleration_max"] == pytest.approx(0.0, abs=1e-12)
    assert result["metrics"]["root_acceleration_max"] == pytest.approx(2.0)


def test_real_frame_gap_controls_derivative_time():
    base = np.zeros((6, 3), dtype=float)
    changed = base.copy()
    changed[:, 0] = 10.0
    result = analyze_3d_temporal(
        _sequence([0, 10], [base, changed]),
        fps=10.0, total_observed_frames=2,
    )
    assert result["metrics"]["joint_velocity_p90"] == 10.0


def test_acceleration_and_jerk_are_attributed_to_internal_frames():
    values = [0.0, 0.0, 1.0, 1.0, 1.0]
    joints = [np.full((6, 3), value) for value in values]
    result = analyze_3d_temporal(
        _sequence(range(5), joints), fps=1.0, total_observed_frames=5,
    )
    acceleration_frames = {
        item["frame_index"] for item in result["worst_acceleration_frames"]
    }
    jerk_frames = {item["frame_index"] for item in result["worst_jerk_frames"]}
    assert acceleration_frames <= {1, 2, 3}
    assert jerk_frames <= {2, 3}
    assert result["worst_acceleration_frames"][0]["frame_index"] in {1, 2}


def test_engine_maps_each_logical_person_and_loads_backend_once(tmp_path, monkeypatch):
    root = tmp_path / "GVHMR"
    root.mkdir()
    checkpoint = tmp_path / "gvhmr.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    class Backend:
        def __init__(self, config, device):
            calls.append(("init", device))

        def infer(self, images, boxes, frame_indices, width, height):
            calls.append(("infer", tuple(frame_indices), tuple(boxes[0])))
            joints = np.zeros((len(frame_indices), 6, 3))
            roots = np.zeros((len(frame_indices), 3))
            return _sequence(frame_indices, joints, roots)

        def close(self):
            calls.append(("close",))

    engine = GVHMRTemporalEngine(
        GVHMRTemporalConfig(root, checkpoint), "cuda:0", Backend
    )
    monkeypatch.setattr(
        engine, "_read_needed_frames",
        lambda video, needed: {
            index: np.zeros((24, 32, 3), dtype=np.uint8) for index in needed
        },
    )
    persons = [
        {"logical_track_id": 0, "frames": [
            {"frame_index": 0, "bbox_xyxy": [1, 2, 10, 20]},
            {"frame_index": 2, "bbox_xyxy": [2, 2, 11, 20]},
        ]},
        {"logical_track_id": 1, "frames": [
            {"frame_index": 1, "bbox_xyxy": [12, 2, 22, 20]},
        ]},
    ]
    engine.score_video(
        Path("video.mp4"), persons, fps=10.0, width=32, height=24
    )
    engine.close()
    assert calls[0] == ("init", "cuda:0")
    assert [call[1] for call in calls if call[0] == "infer"] == [(0, 2), (1,)]
    assert calls[-1] == ("close",)
    assert all(person["temporal"]["human_3d"]["valid"] for person in persons)
