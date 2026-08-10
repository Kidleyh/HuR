import json
from pathlib import Path

import pytest

from scripts import run_human_reward_pairs


def _make_pair(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "gt.mp4").write_bytes(b"gt")
    (directory / "render.mp4").write_bytes(b"render")
    (directory / "sample.json").write_text('{"ignored": true}')
    return directory


def _full_result(path: Path):
    return {
        "valid": True,
        "video": {"path": str(path), "width": 32, "height": 24},
        "persons": [{
            "logical_track_id": 0,
            "track": {"source_track_ids": [1]},
            "frames": [{
                "frame_index": 0,
                "human": {"scored": True},
                "faces": [],
                "hands": [],
                "human_pose": {"keypoints_xy": [], "keypoint_scores": []},
            }],
            "score": {"binary_score": 1.0},
            "temporal": {"human": {"score": None}},
        }],
        "video_score": {"reward": 1.0, "micro_score": 1.0, "macro_score": 1.0},
        "reward": 1.0,
        "micro_score": 1.0,
        "macro_score": 1.0,
    }


def test_pair_cli_scores_once_and_preserves_complete_results(
    tmp_path, monkeypatch
):
    root = tmp_path / "pairs"
    first = _make_pair(root, "a_pair")
    second = _make_pair(root, "中文_pair")
    output = tmp_path / "result/all_scores.json"
    calls = []

    class Model:
        def __init__(self, config):
            calls.append("init")

        def score_batch(self, video_paths):
            paths = list(video_paths)
            calls.append(paths)
            return [_full_result(path) for path in paths]

    monkeypatch.setattr(run_human_reward_pairs, "HumanRewardModel", Model)
    assert run_human_reward_pairs.main([
        "--input-dir", str(root), "--output", str(output), "--device", "cpu",
    ]) == 0

    expected_paths = [
        first / "gt.mp4", first / "render.mp4",
        second / "gt.mp4", second / "render.mp4",
    ]
    assert calls == ["init", [path.resolve() for path in expected_paths]]
    data = json.loads(output.read_text())
    assert data["schema_version"] == "1.0"
    assert data["pair_count"] == 2
    assert data["video_count"] == 4
    assert [pair["name"] for pair in data["pairs"]] == ["a_pair", "中文_pair"]
    assert data["pairs"][0]["positive"]["kind"] == "gt"
    assert data["pairs"][0]["negative"]["kind"] == "render"
    assert data["pairs"][0]["positive"]["result"]["persons"][0][
        "temporal"
    ] == {"human": {"score": None}}
    assert not list(output.parent.glob(".*.tmp"))


def test_discovery_rejects_incomplete_pair(tmp_path):
    directory = tmp_path / "pairs/incomplete"
    directory.mkdir(parents=True)
    (directory / "gt.mp4").write_bytes(b"gt")
    with pytest.raises(ValueError, match="incomplete: missing render.mp4"):
        run_human_reward_pairs.discover_video_pairs(tmp_path / "pairs")


def test_build_paired_result_rejects_result_count_mismatch(tmp_path):
    pair_dir = _make_pair(tmp_path / "pairs", "sample")
    pair = run_human_reward_pairs.VideoPair(
        "sample", (pair_dir / "gt.mp4").resolve(),
        (pair_dir / "render.mp4").resolve(),
    )
    with pytest.raises(RuntimeError, match="1 results for 2 videos"):
        run_human_reward_pairs.build_paired_result(
            tmp_path / "pairs", [pair], [{}]
        )


def test_pair_cli_writes_visualizations_in_matching_directory_layout(
    tmp_path, monkeypatch
):
    root = tmp_path / "pairs"
    first = _make_pair(root, "a_pair")
    second = _make_pair(root, "中文_pair")
    output = tmp_path / "scores.json"
    visualization_dir = tmp_path / "visualizations"
    model_calls = []
    visualization_calls = []

    class Model:
        def __init__(self, config):
            pass

        def score_batch(self, video_paths):
            paths = list(video_paths)
            model_calls.append(paths)
            return [_full_result(path) for path in paths]

    def visualize(video, result, destination):
        visualization_calls.append((video, destination, result["reward"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"visualization")

    monkeypatch.setattr(run_human_reward_pairs, "HumanRewardModel", Model)
    monkeypatch.setattr(
        run_human_reward_pairs, "write_human_reward_visualization", visualize
    )
    assert run_human_reward_pairs.main([
        "--input-dir", str(root),
        "--output", str(output),
        "--visualization-dir", str(visualization_dir),
        "--device", "cpu",
    ]) == 0

    expected_videos = [
        first / "gt.mp4", first / "render.mp4",
        second / "gt.mp4", second / "render.mp4",
    ]
    expected_outputs = [
        visualization_dir / "a_pair/gt.mp4",
        visualization_dir / "a_pair/render.mp4",
        visualization_dir / "中文_pair/gt.mp4",
        visualization_dir / "中文_pair/render.mp4",
    ]
    assert model_calls == [[path.resolve() for path in expected_videos]]
    assert [call[0] for call in visualization_calls] == [
        path.resolve() for path in expected_videos
    ]
    assert [call[1] for call in visualization_calls] == [
        path.resolve() for path in expected_outputs
    ]
    assert all(path.read_bytes() == b"visualization" for path in expected_outputs)
    data = json.loads(output.read_text())
    assert data["pairs"][0]["positive"]["result"]["visualization"] == str(
        expected_outputs[0].resolve()
    )
    assert data["pairs"][1]["negative"]["result"]["visualization"] == str(
        expected_outputs[3].resolve()
    )
