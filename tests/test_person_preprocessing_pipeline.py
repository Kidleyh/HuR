"""Tests for the single-video end-to-end preprocessing entry point."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import run_person_preprocessing_pipeline as pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_output_dirs_use_video_stem_and_stage_names(tmp_path):
    video = tmp_path / "clips" / "person sample.final.mp4"
    tracking, stitching = pipeline.pipeline_output_dirs(video, tmp_path / "outputs")
    assert tracking == (tmp_path / "outputs/person sample.final_person_tracking").resolve()
    assert stitching == (
        tmp_path / "outputs/person sample.final_tracklet_stitching"
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
            "--output-root", str(output_root),
            "--device", "cpu",
            "--no-half",
            "--no-save-visualization",
        ]
    )
    assert code == 0
    assert calls["tracking"][1] == output_root / "sample_person_tracking"
    assert calls["stitching"][2] == output_root / "sample_tracklet_stitching"
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
    assert "--output-root" in completed.stdout
