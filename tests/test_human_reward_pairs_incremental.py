import json
from pathlib import Path

import pytest

from scripts import run_human_reward_pairs_incremental as incremental


def _make_pair(root: Path, name: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "gt.mp4").write_bytes(b"gt")
    (directory / "render.mp4").write_bytes(b"render")


def _result(path: Path):
    is_gt = path.name == "gt.mp4"
    probability = 0.1 if is_gt else 0.8
    abnormal = not is_gt
    return {
        "valid": True,
        "reason": None,
        "persons": [{
            "logical_track_id": 0,
            "track": {"source_track_ids": [1]},
            "frames": [{
                "frame_index": 0,
                "source_track_id": 1,
                "human": {
                    "scored": True,
                    "abnormal_probability": probability,
                    "abnormal": abnormal,
                },
                "faces": [],
                "hands": [],
                "person_abnormal": abnormal,
                "failure_reason": None,
            }],
            "score": {"binary_score": float(is_gt)},
            "temporal": {},
        }],
        "reward": float(is_gt),
        "micro_score": float(is_gt),
        "macro_score": float(is_gt),
    }


def _manifest(tmp_path: Path):
    root = tmp_path / "pairs"
    _make_pair(root, "first")
    _make_pair(root, "second")
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({
        "data_root": str(root),
        "selected_pairs": [{"folder": "second"}, {"folder": "first"}],
    }))
    return root, manifest


class _Model:
    calls = []

    def __init__(self, config):
        pass

    def score_batch(self, paths):
        paths = list(paths)
        self.calls.append(paths)
        return [_result(path) for path in paths]


def test_checkpoints_after_every_pair_and_reports_current_accuracy(
    tmp_path, monkeypatch
):
    _, manifest = _manifest(tmp_path)
    output = tmp_path / "incremental"
    _Model.calls = []
    checkpoints = []
    original = incremental.write_incremental_checkpoint

    def recording_checkpoint(**kwargs):
        checkpoints.append(len(kwargs["completed_pairs"]))
        return original(**kwargs)

    monkeypatch.setattr(incremental, "HumanRewardModel", _Model)
    monkeypatch.setattr(
        incremental, "write_incremental_checkpoint", recording_checkpoint
    )
    assert incremental.main([
        "--selection-manifest", str(manifest),
        "--output", str(output), "--device", "cpu",
    ]) == 0

    assert checkpoints == [1, 2]
    assert len(_Model.calls) == 2
    assert all(len(call) == 2 for call in _Model.calls)
    progress = json.loads(
        (output / incremental.PROGRESS_FILENAME).read_text()
    )
    assert progress["status"] == "complete"
    assert progress["completed_pair_count"] == 2
    metric = progress["dataset_metrics"]["human_probability_quality"]
    assert metric["gt_win_count"] == 2
    assert metric["strict_accuracy"] == 1.0
    full = json.loads((output / incremental.FULL_RESULT_FILENAME).read_text())
    assert [pair["name"] for pair in full["pairs"]] == ["second", "first"]


def test_existing_output_is_not_overwritten_without_resume(
    tmp_path, monkeypatch
):
    _, manifest = _manifest(tmp_path)
    output = tmp_path / "incremental"
    output.mkdir()
    (output / incremental.PROGRESS_FILENAME).write_text("existing")
    monkeypatch.setattr(incremental, "HumanRewardModel", _Model)

    with pytest.raises(SystemExit):
        incremental.main([
            "--selection-manifest", str(manifest),
            "--output", str(output), "--device", "cpu",
        ])
    assert (output / incremental.PROGRESS_FILENAME).read_text() == "existing"


def test_resume_processes_only_the_remaining_ordered_pairs(
    tmp_path, monkeypatch
):
    _, manifest = _manifest(tmp_path)
    output = tmp_path / "incremental"
    _Model.calls = []
    monkeypatch.setattr(incremental, "HumanRewardModel", _Model)
    assert incremental.main([
        "--selection-manifest", str(manifest),
        "--output", str(output), "--device", "cpu", "--max-pairs", "1",
    ]) == 0
    assert len(_Model.calls) == 1

    _Model.calls = []
    assert incremental.main([
        "--selection-manifest", str(manifest),
        "--output", str(output), "--device", "cpu", "--resume",
    ]) == 0
    assert len(_Model.calls) == 1
    assert _Model.calls[0][0].parent.name == "first"
    progress = json.loads(
        (output / incremental.PROGRESS_FILENAME).read_text()
    )
    assert progress["completed_pair_count"] == 2
    assert progress["status"] == "complete"
