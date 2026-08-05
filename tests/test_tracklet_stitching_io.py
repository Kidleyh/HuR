"""Tests for schema-1.1 input, tracklet construction, paths, and atomic output."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from astrolabe.scorers.video.person_tracking.schemas import FrameDetections, RawDetection, TrackedDetection
from astrolabe.scorers.video.tracklet_stitching import serialization
from astrolabe.scorers.video.tracklet_stitching.io import (
    build_tracklets, load_tracking_input, output_dir_for_result,
)
from astrolabe.scorers.video.tracklet_stitching.schemas import StitchingConfig
from astrolabe.scorers.video.tracklet_stitching.stitcher import stitch_tracking
from scripts.run_tracklet_stitching import build_parser, is_complete_result, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tracked(track_id, confidence=0.9):
    return TrackedDetection.from_xyxy(
        track_id=track_id, class_id=0, class_name="person", confidence=confidence,
        bbox_xyxy=[10, 10, 30, 60], image_width=100, image_height=100,
    )


def raw(index=0):
    return RawDetection.from_xyxy(
        class_id=0, class_name="person", confidence=0.5,
        bbox_xyxy=[10, 10, 30, 60], image_width=100, image_height=100,
        detection_index=index,
    )


def write_source(path: Path):
    path.mkdir(parents=True)
    frames = [
        FrameDetections(0, 0.0, [raw()], [tracked(1, 0.8), tracked(1, 0.9)]),
        FrameDetections(1, 0.1, [], []),
    ]
    with (path / "detections.jsonl").open("w", encoding="utf-8") as handle:
        for frame in frames:
            payload = frame.to_dict()
            payload["unknown_frame_field"] = "ignored"
            handle.write(json.dumps(payload) + "\n")
    (path / "tracks_summary.json").write_text(json.dumps({
        "schema_version": "1.1",
        "video": {"path": str(path / "missing.mp4"), "width": 100, "height": 100,
                  "fps": 10.0, "num_frames": 2, "duration_sec": 0.2},
    }), encoding="utf-8")
    return frames


def test_load_schema_unknown_fields_and_deduplicate(tmp_path):
    source = tmp_path / "source"
    write_source(source)
    loaded = load_tracking_input(source)
    assert len(loaded.frames) == 2
    tracklets = build_tracklets(loaded.frames)
    assert len(tracklets) == 1
    assert tracklets[0].num_observed_frames == 1
    assert tracklets[0].detections[0].confidence == 0.9


def test_missing_jsonl_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="detections.jsonl"):
        load_tracking_input(tmp_path)


def test_frame_count_mismatch_raises(tmp_path):
    source = tmp_path / "source"
    write_source(source)
    summary = json.loads((source / "tracks_summary.json").read_text())
    summary["video"]["num_frames"] = 3
    (source / "tracks_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="declares 3"):
        load_tracking_input(source)


def test_relative_output_path_prevents_name_collision(tmp_path):
    root, output = tmp_path / "root", tmp_path / "output"
    a, b = root / "a" / "sample", root / "b" / "sample"
    assert output_dir_for_result(a, root, output) == output.resolve() / "a/sample"
    assert output_dir_for_result(b, root, output) == output.resolve() / "b/sample"


def test_serialization_preserves_source_and_adds_logical_id(tmp_path):
    source = tmp_path / "source"
    write_source(source)
    before = (source / "detections.jsonl").read_bytes()
    loaded = load_tracking_input(source)
    result = stitch_tracking(loaded, StitchingConfig())
    output = tmp_path / "output"
    serialization.write_stitching_outputs(
        result, loaded.frames, Path("missing.mp4"), output, False
    )
    assert (source / "detections.jsonl").read_bytes() == before
    payload = json.loads((output / "tracklet_stitching.json").read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["visualization"]["skip_reason"] == "not_requested"
    line = json.loads((output / "stitched_detections.jsonl").read_text().splitlines()[0])
    assert line["tracked_detections"][0]["track_id"] == 1
    assert line["tracked_detections"][0]["logical_track_id"] == 0


def test_missing_source_video_is_complete_success(tmp_path):
    source = tmp_path / "source"
    write_source(source)
    loaded = load_tracking_input(source)
    result = stitch_tracking(loaded, StitchingConfig())
    output = tmp_path / "output"
    missing_video = tmp_path / "missing.mp4"
    serialization.write_stitching_outputs(
        result, loaded.frames, missing_video, output, True
    )
    for name in (
        "tracklet_stitching.json",
        "stitched_detections.jsonl",
        "stitched_tracks_summary.json",
    ):
        assert (output / name).is_file()
    assert not (output / "stitched.mp4").exists()
    assert not (output / "stitching_error.json").exists()
    summary = json.loads((output / "stitched_tracks_summary.json").read_text())
    assert summary["visualization"] == {
        "requested": True,
        "generated": False,
        "source_video_path": str(missing_video.resolve()),
        "skip_reason": "source_video_missing",
    }
    main = json.loads((output / "tracklet_stitching.json").read_text())
    assert main["visualization"] == summary["visualization"]
    assert is_complete_result(output, visualization_requested=True)

    missing_video.touch()
    assert not is_complete_result(output, visualization_requested=True)


def test_generated_visualization_requires_nonempty_mp4(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    for name in ("tracklet_stitching.json", "stitched_detections.jsonl"):
        (output / name).write_text("{}\n")
    summary = {
        "visualization": {
            "requested": True,
            "generated": True,
            "source_video_path": str(tmp_path / "video.mp4"),
            "skip_reason": None,
        }
    }
    (output / "stitched_tracks_summary.json").write_text(json.dumps(summary))
    assert not is_complete_result(output, visualization_requested=True)


def test_visualization_failure_does_not_write_success_json(tmp_path, monkeypatch):
    source = tmp_path / "source"
    write_source(source)
    loaded = load_tracking_input(source)
    result = stitch_tracking(loaded, StitchingConfig())
    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "output"

    def fail_visualization(*args, **kwargs):
        raise RuntimeError("simulated encoder failure")

    monkeypatch.setattr(serialization, "write_stitched_video", fail_visualization)
    with pytest.raises(RuntimeError, match="encoder failure"):
        serialization.write_stitching_outputs(
            result, loaded.frames, video, output, True
        )
    assert not (output / "tracklet_stitching.json").exists()


def test_no_visualization_request_only_requires_core_outputs(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    for name in ("tracklet_stitching.json", "stitched_detections.jsonl"):
        (output / name).write_text("{}\n")
    (output / "stitched_tracks_summary.json").write_text(json.dumps({
        "visualization": {
            "requested": True,
            "generated": False,
            "source_video_path": str(tmp_path / "missing.mp4"),
            "skip_reason": "source_video_missing",
        }
    }))
    assert is_complete_result(output, visualization_requested=False)


def test_legacy_or_invalid_visualization_summary_is_incomplete_when_requested(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "tracklet_stitching.json").write_text("{}\n")
    (output / "stitched_detections.jsonl").write_text("{}\n")
    summary_path = output / "stitched_tracks_summary.json"
    summary_path.write_text(json.dumps({"schema_version": "1.0"}))
    assert not is_complete_result(output, visualization_requested=True)
    assert is_complete_result(output, visualization_requested=False)

    for invalid in (None, [], "bad", {"requested": "true"}):
        summary_path.write_text(json.dumps({"visualization": invalid}))
        assert not is_complete_result(output, visualization_requested=True)

    summary_path.write_text("not-json")
    assert not is_complete_result(output, visualization_requested=True)


def test_success_removes_old_stitching_error(tmp_path):
    source = tmp_path / "source"
    write_source(source)
    loaded = load_tracking_input(source)
    result = stitch_tracking(loaded, StitchingConfig())
    output = tmp_path / "output"
    output.mkdir()
    (output / "stitching_error.json").write_text("old error")
    serialization.write_stitching_outputs(
        result, loaded.frames, Path("missing.mp4"), output, False
    )
    assert not (output / "stitching_error.json").exists()


def test_cli_help_from_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/run_tracklet_stitching.py"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"data=runpy.run_path({str(PROJECT_ROOT / 'scripts/run_tracklet_stitching.py')!r}); "
                "assert data['DEFAULT_CONFIG'].is_file()"
            ),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_cli_can_override_associated_raw_setting():
    common = [
        "--input", "input", "--output-dir", "output",
        "--config", str(PROJECT_ROOT / "configs/tracklet_stitching.yaml"),
    ]
    default_args = build_parser().parse_args(common)
    assert default_args.raw_bridge_allow_associated_raw is None
    assert load_config(default_args).raw_bridge_allow_associated_raw is False
    enabled = build_parser().parse_args(
        common + ["--raw-bridge-allow-associated-raw"]
    )
    assert load_config(enabled).raw_bridge_allow_associated_raw is True
    disabled = build_parser().parse_args(
        common + ["--no-raw-bridge-allow-associated-raw"]
    )
    assert load_config(disabled).raw_bridge_allow_associated_raw is False


def test_failed_write_preserves_existing_output_and_cleans_temp(tmp_path, monkeypatch):
    source = tmp_path / "source"
    write_source(source)
    loaded = load_tracking_input(source)
    result = stitch_tracking(loaded, StitchingConfig())
    output = tmp_path / "output"
    output.mkdir()
    old = output / "tracklet_stitching.json"
    old.write_text("old-complete")
    original = serialization._write_json
    calls = 0

    def fail_second(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated failure")
        return original(path, payload)

    monkeypatch.setattr(serialization, "_write_json", fail_second)
    with pytest.raises(OSError, match="simulated"):
        serialization.write_stitching_outputs(
            result, loaded.frames, Path("missing.mp4"), output, False
        )
    assert old.read_text() == "old-complete"
    assert not list(tmp_path.glob(".output.tmp-*"))
