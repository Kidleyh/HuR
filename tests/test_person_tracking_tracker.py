"""Mocked lifecycle and atomic-output tests for the tracking runtime."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes

import astrolabe.scorers.video.person_tracking.tracker as tracker_module
from astrolabe.scorers.video.person_tracking.tracker import YOLOByteTrackPersonTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeCapture:
    def __init__(self, _: str) -> None:
        self.frames = [np.zeros((48, 64, 3), dtype=np.uint8)]
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: 1.0,
        }.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.opened = False


class FakeYOLO:
    constructions = 0
    predictions = 0

    def __init__(self, _: str) -> None:
        type(self).constructions += 1
        self.names = {0: "person"}

    def predict(self, **_: object) -> list[SimpleNamespace]:
        type(self).predictions += 1
        data = torch.tensor([[10.0, 8.0, 30.0, 40.0, 0.92, 0.0]])
        return [SimpleNamespace(boxes=Boxes(data, orig_shape=(48, 64)))]


class FailingYOLO(FakeYOLO):
    def predict(self, **_: object) -> list[SimpleNamespace]:
        raise RuntimeError("synthetic inference failure")


class FakeByteTracker:
    constructions = 0

    def __init__(self, args: object) -> None:
        type(self).constructions += 1
        self.args = args

    def update(self, results: Boxes, _: np.ndarray) -> np.ndarray:
        if len(results) == 0:
            return np.empty((0, 8), dtype=np.float32)
        row = np.asarray(results.data[0], dtype=np.float32)
        return np.asarray([[*row[:4], 1.0, row[4], row[5], 0.0]], dtype=np.float32)


@pytest.fixture
def runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    weights = tmp_path / "weights.pt"
    weights.touch()
    video = tmp_path / "video.mp4"
    video.touch()
    return weights, video


def build_tracker(monkeypatch: pytest.MonkeyPatch, weights: Path, model: type = FakeYOLO):
    monkeypatch.setattr(tracker_module, "YOLO", model)
    monkeypatch.setattr(tracker_module, "BYTETracker", FakeByteTracker)
    monkeypatch.setattr(tracker_module.cv2, "VideoCapture", FakeCapture)
    return YOLOByteTrackPersonTracker(
        str(weights),
        str(PROJECT_ROOT / "configs/bytetrack_person.yaml"),
        device="cpu",
        half=False,
    )


def test_model_loaded_once_and_tracker_reset_per_video(
    monkeypatch: pytest.MonkeyPatch, runtime_files: tuple[Path, Path], tmp_path: Path
) -> None:
    weights, first_video = runtime_files
    second_video = tmp_path / "second.mp4"
    second_video.touch()
    FakeYOLO.constructions = FakeYOLO.predictions = FakeByteTracker.constructions = 0
    tracker = build_tracker(monkeypatch, weights)
    first = tracker.track_video(str(first_video), str(tmp_path / "first"), False)
    second = tracker.track_video(str(second_video), str(tmp_path / "second"), False)
    assert FakeYOLO.constructions == 1
    assert FakeYOLO.predictions == 2
    assert FakeByteTracker.constructions == 2
    assert first.frames[0].tracked_detections[0].track_id == 1
    assert second.frames[0].tracked_detections[0].track_id == 1


def test_success_removes_old_error_but_preserves_user_file(
    monkeypatch: pytest.MonkeyPatch, runtime_files: tuple[Path, Path], tmp_path: Path
) -> None:
    weights, video = runtime_files
    destination = tmp_path / "result"
    destination.mkdir()
    (destination / "error.json").write_text("old error", encoding="utf-8")
    (destination / "user-note.txt").write_text("keep", encoding="utf-8")
    tracker = build_tracker(monkeypatch, weights)
    tracker.track_video(str(video), str(destination), False)
    assert not (destination / "error.json").exists()
    assert (destination / "user-note.txt").read_text(encoding="utf-8") == "keep"
    assert (destination / "raw_detections.csv").is_file()
    assert (destination / "tracked_detections.csv").is_file()
    assert (destination / "detections.csv").read_bytes() == (
        destination / "tracked_detections.csv"
    ).read_bytes()


def test_failure_preserves_complete_output_and_cleans_temporary_directory(
    monkeypatch: pytest.MonkeyPatch, runtime_files: tuple[Path, Path], tmp_path: Path
) -> None:
    weights, video = runtime_files
    destination = tmp_path / "result"
    destination.mkdir()
    previous = destination / "detections.jsonl"
    previous.write_text("previous complete output\n", encoding="utf-8")
    tracker = build_tracker(monkeypatch, weights, FailingYOLO)
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        tracker.track_video(str(video), str(destination), False)
    assert previous.read_text(encoding="utf-8") == "previous complete output\n"
    assert list(tmp_path.glob(".result.tmp-*")) == []
