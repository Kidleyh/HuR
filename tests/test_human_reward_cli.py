import json

from pathlib import Path

import pytest

from scripts import run_human_reward


RESULT = {
    "valid": True, "reason": None,
    "reward": 0.9, "micro_score": 0.9, "macro_score": 0.8,
    "logical_track_count": 1, "observed_person_frames": 2,
    "scored_person_frames": 2, "abnormal_person_frames": 0,
    "failed_person_frames": 0,
    "visualization": None,
}


class Model:
    def __init__(self, config):
        self.config = config

    def score(self, video, visualization_output=None):
        result = dict(RESULT)
        if visualization_output is not None:
            destination = Path(visualization_output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"mp4")
            result["visualization"] = str(destination)
        return result

    def score_batch(self, videos):
        return [dict(RESULT, reward=index / 10) for index, _ in enumerate(videos)]


def test_cli_without_output_prints_one_json_and_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(run_human_reward, "HumanRewardModel", Model)
    monkeypatch.chdir(tmp_path)
    assert run_human_reward.main(["--video", str(video), "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out) == RESULT
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.mp4"]


def test_cli_writes_only_atomic_final_json(tmp_path, monkeypatch, capsys):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "result/reward.json"
    monkeypatch.setattr(run_human_reward, "HumanRewardModel", Model)
    assert run_human_reward.main([
        "--video", str(video), "--output", str(output), "--device", "cpu"
    ]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text()) == RESULT
    assert list(output.parent.glob(".*.tmp")) == []


def test_cli_accepts_repeated_videos_and_writes_one_json_array(
    tmp_path, monkeypatch, capsys
):
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    first.write_bytes(b"video")
    second.write_bytes(b"video")
    output = tmp_path / "rewards.json"
    monkeypatch.setattr(run_human_reward, "HumanRewardModel", Model)

    assert run_human_reward.main([
        "--video", str(first), "--video", str(second),
        "--output", str(output), "--device", "cpu",
    ]) == 0

    assert capsys.readouterr().out == ""
    data = json.loads(output.read_text())
    assert [item["reward"] for item in data] == [0.0, 0.1]
    assert list(tmp_path.glob(".*.tmp")) == []


def test_cli_single_video_adds_one_visualization(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    reward = tmp_path / "reward.json"
    visualization = tmp_path / "reward_visualization.mp4"
    monkeypatch.setattr(run_human_reward, "HumanRewardModel", Model)

    assert run_human_reward.main([
        "--video", str(video), "--output", str(reward),
        "--visualization-output", str(visualization), "--device", "cpu",
    ]) == 0

    assert visualization.read_bytes() == b"mp4"
    assert json.loads(reward.read_text())["visualization"] == str(
        visualization.resolve()
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "input.mp4", "reward.json", "reward_visualization.mp4"
    ]


def test_cli_rejects_batch_visualization_before_model_load(
    tmp_path, monkeypatch, capsys
):
    loaded = []

    class UnexpectedModel:
        def __init__(self, config):
            loaded.append(True)

    monkeypatch.setattr(run_human_reward, "HumanRewardModel", UnexpectedModel)
    with pytest.raises(SystemExit) as error:
        run_human_reward.main([
            "--video", str(tmp_path / "a.mp4"),
            "--video", str(tmp_path / "b.mp4"),
            "--visualization-output", str(tmp_path / "result.mp4"),
        ])
    assert error.value.code == 2
    assert "supports single-video mode only" in capsys.readouterr().err
    assert loaded == []
