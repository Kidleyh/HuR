import json

from scripts import run_human_reward


RESULT = {
    "reward": 0.9, "micro_score": 0.9, "macro_score": 0.8,
    "logical_track_count": 1, "observed_person_frames": 2,
    "scored_person_frames": 2, "abnormal_person_frames": 0,
    "failed_person_frames": 0,
}


class Model:
    def __init__(self, config):
        self.config = config

    def score(self, video):
        return RESULT


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
