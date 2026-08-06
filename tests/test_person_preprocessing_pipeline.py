"""Tests for the single-video end-to-end preprocessing entry point."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import run_person_preprocessing_pipeline as pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_output_dirs_use_explicit_name_and_stage_names(tmp_path):
    tracking, stitching = pipeline.pipeline_output_dirs(
        "experiment 27k", tmp_path / "outputs"
    )
    assert tracking == (tmp_path / "outputs/experiment 27k_person_tracking").resolve()
    assert stitching == (
        tmp_path / "outputs/experiment 27k_tracklet_stitching"
    ).resolve()


def test_pipeline_output_dirs_reject_unsafe_names(tmp_path):
    for name in ("", " ", ".", "..", "../escape", "a/b", "a\\b", " padded "):
        with pytest.raises(ValueError, match="single directory name"):
            pipeline.pipeline_output_dirs(name, tmp_path / "outputs")


def test_human_anomaly_output_uses_same_explicit_name(tmp_path):
    assert pipeline.human_anomaly_output_dir("run42", tmp_path / "outputs") == (
        tmp_path / "outputs/run42_human_anomaly"
    ).resolve()


def test_pipeline_runs_both_stages_into_exact_directories(tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    calls = {}

    class FakeTracker:
        def __init__(self, **kwargs):
            calls["tracker_init"] = kwargs

        def track_video(self, video_path, output_dir, **kwargs):
            calls["tracking"] = (Path(video_path), Path(output_dir), kwargs)

    monkeypatch.setattr(pipeline, "resolve_yolo_weights", lambda *args, **kwargs: "weights.pt")
    monkeypatch.setattr(pipeline, "YOLOByteTrackPersonTracker", FakeTracker)
    monkeypatch.setattr(pipeline, "_tracking_complete", lambda *args: False)
    monkeypatch.setattr(pipeline, "is_complete_result", lambda *args: False)
    monkeypatch.setattr(
        pipeline,
        "load_tracking_input",
        lambda path: SimpleNamespace(frames=[], source_dir=Path(path)),
    )
    monkeypatch.setattr(pipeline, "stitch_tracking", lambda tracking, config: "result")

    def fake_write(result, frames, source_video, output_dir, save_visualization):
        calls["stitching"] = (
            result,
            Path(source_video),
            Path(output_dir),
            save_visualization,
        )

    monkeypatch.setattr(pipeline, "write_stitching_outputs", fake_write)
    output_root = tmp_path / "outputs"
    code = pipeline.main(
        [
            "--input", str(video),
            "--name", "custom_run",
            "--output-root", str(output_root),
            "--device", "cpu",
            "--no-half",
            "--no-save-visualization",
        ]
    )
    assert code == 0
    assert calls["tracking"][1] == output_root / "custom_run_person_tracking"
    assert calls["stitching"][2] == output_root / "custom_run_tracklet_stitching"
    assert calls["stitching"][1] == video.resolve()


def test_pipeline_skips_both_complete_stages(tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(pipeline, "_tracking_complete", lambda *args: True)
    monkeypatch.setattr(pipeline, "is_complete_result", lambda *args: True)

    class MustNotLoad:
        def __init__(self, **kwargs):
            raise AssertionError("YOLO must not load when both stages are complete")

    monkeypatch.setattr(pipeline, "YOLOByteTrackPersonTracker", MustNotLoad)
    assert pipeline.main([
        "--input", str(video),
        "--name", "skip_test",
        "--output-root", str(tmp_path / "outputs"),
        "--no-save-visualization",
    ]) == 0


def test_pipeline_help_works_outside_repository(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_person_preprocessing_pipeline.py"),
            "--help",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--name" in completed.stdout
    assert "--output-root" in completed.stdout


def test_pipeline_invokes_optional_human_anomaly_stage(tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(pipeline, "_tracking_complete", lambda *args: True)
    monkeypatch.setattr(pipeline, "is_complete_result", lambda *args: True)
    captured = {}

    def successful(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(pipeline, "run_human_anomaly_main", successful)
    code = pipeline.main([
        "--input", str(video), "--name", "stage13",
        "--output-root", str(tmp_path / "outputs"), "--no-save-visualization",
        "--human-anomaly", "--vbench-root", str(tmp_path / "VBench"),
        "--vbench-cache-dir", str(tmp_path / "cache"),
        "--vbench-clip-model", str(tmp_path / "clip"),
    ])
    assert code == 0
    args = captured["args"]
    assert str(tmp_path / "outputs/stage13_tracklet_stitching") in args
    assert str(tmp_path / "outputs/stage13_human_anomaly") in args


import pytest
