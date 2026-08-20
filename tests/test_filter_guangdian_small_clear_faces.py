import json
from pathlib import Path

import cv2
import numpy as np

from scripts import filter_guangdian_small_clear_faces as dataset_filter
from scripts import filter_small_clear_faces as face_filter


def _create_triple(version: Path, stem: str) -> None:
    (version / "caption_v2").mkdir(parents=True, exist_ok=True)
    (version / "video" / "data").mkdir(parents=True, exist_ok=True)
    (version / "label").mkdir(parents=True, exist_ok=True)
    (version / "caption_v2" / f"{stem}.json").write_text("{}")
    (version / "video" / "data" / f"{stem}.mp4").write_bytes(b"video")
    (version / "label" / f"{stem}.json").write_text("{}")


def test_discovery_matches_stems_and_records_missing_directories(tmp_path):
    complete = tmp_path / "1" / "v2.0.0"
    _create_triple(complete, "sample name")
    (tmp_path / "2" / "v2.0.0").mkdir(parents=True)
    (tmp_path / "3").mkdir()
    triples, skipped = dataset_filter.discover_dataset(tmp_path)
    assert len(triples) == 1
    assert triples[0][0].name == "sample name.json"
    assert triples[0][1].name == "sample name.mp4"
    assert {item["reason"] for item in skipped} == {
        "caption_directory_missing", "version_directory_missing"
    }


def test_discovery_skips_missing_video_or_label(tmp_path):
    version = tmp_path / "1" / "v2.0.0"
    (version / "caption_v2").mkdir(parents=True)
    (version / "caption_v2" / "missing.json").write_text("{}")
    triples, skipped = dataset_filter.discover_dataset(tmp_path)
    assert triples == []
    assert skipped[0]["reason"] == "paired_file_missing"
    assert skipped[0]["missing"] == ["video", "label"]


def test_selected_manifest_entry_has_exact_requested_fields(tmp_path):
    result = dataset_filter.selected_manifest_entry(
        tmp_path / "caption.json", tmp_path / "video.mp4", tmp_path / "label.json"
    )
    assert list(result) == [
        "video_caption_path", "file_path", "label_path", "lipsync"
    ]
    assert result["lipsync"] == {}


def test_first_frame_analysis_decodes_only_one_frame(tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for value in (0, 100, 200):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    calls = []

    def fake_analyze(frame, frame_index, detector, thresholds, **options):
        calls.append((frame_index, float(frame.mean())))
        return {"frame_index": 0, "faces": [{"qualified": True}]}

    monkeypatch.setattr(dataset_filter, "analyze_frame", fake_analyze)
    result = dataset_filter.analyze_video_first_frame(
        video, object(), face_filter.FilterThresholds(), {}
    )
    assert calls == [(0, 0.0)]
    assert result["selected"] is True
    assert result["width"] == 64
    assert result["height"] == 48


def test_outputs_are_separate_and_resume_tracks_processed(tmp_path):
    output = tmp_path / "selected.json"
    skipped_output = dataset_filter.default_skipped_output(output)
    selected = [{"video_caption_path": "/a.json", "file_path": "/a.mp4",
                 "label_path": "/a-label.json", "lipsync": {}}]
    skipped = [{"video_caption_path": "/b.json", "reason": "not_qualified"}]
    dataset_filter._write_outputs(
        output, skipped_output, selected, skipped, tmp_path, 2
    )
    assert json.loads(output.read_text()) == selected
    assert json.loads(skipped_output.read_text())["entries"] == skipped
    loaded_selected, loaded_skipped, processed = dataset_filter._load_existing(
        output, skipped_output, resume=True
    )
    assert loaded_selected == selected
    assert loaded_skipped == skipped
    assert processed == {"/a.json", "/b.json"}
