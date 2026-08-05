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
    line = json.loads((output / "stitched_detections.jsonl").read_text().splitlines()[0])
    assert line["tracked_detections"][0]["track_id"] == 1
    assert line["tracked_detections"][0]["logical_track_id"] == 0


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
