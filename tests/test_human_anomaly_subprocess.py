import json
import subprocess
from pathlib import Path

import pytest

from astrolabe.scorers.video.human_anomaly.subprocess_backend import (
    BackendResult, VBenchWorkerError, build_worker_command, run_vbench_worker,
)
from astrolabe.scorers.video.human_anomaly.schema import HumanAnomalyInput
from scripts import run_person_human_anomaly as main_script


def kwargs(tmp_path):
    return dict(
        conda_env="vbench2-human-anomaly", worker_script=tmp_path / "worker.py",
        video=tmp_path / "video.mp4", input_manifest=tmp_path / "input.jsonl",
        output_jsonl=tmp_path / "frames.jsonl", runtime_info_json=tmp_path / "runtime.json",
        vbench_root=tmp_path / "VBench-2.0", cache_dir=tmp_path / ".cache/vbench2",
        clip_model=tmp_path / "clip", hur_root=tmp_path / "HuR", device="cuda:2",
        crop_batch_size=64,
    )


def test_worker_command_contains_isolated_environment_and_no_shell(tmp_path):
    command = build_worker_command(**kwargs(tmp_path))
    assert command[:5] == ["conda", "run", "-n", "vbench2-human-anomaly", "env"]
    assert "CUDA_VISIBLE_DEVICES=2" in command
    assert any(item.startswith("VBENCH2_CACHE_DIR=") for item in command)
    assert any(item.startswith("VBENCH2_CLIP_TEXT_MODEL=") for item in command)
    assert "--crop-batch-size" in command and "64" in command


def test_worker_failure_preserves_logs_and_raises(tmp_path):
    def runner(command, **options):
        assert options["shell"] is False
        assert options["cwd"] == str(tmp_path.resolve())
        return subprocess.CompletedProcess(command, 7, stdout="worker out", stderr="cuda failed")

    with pytest.raises(VBenchWorkerError) as caught:
        run_vbench_worker(
            ["conda", "run"], tmp_path / "stdout.log", tmp_path / "stderr.log",
            working_directory=tmp_path, runner=runner,
        )
    assert caught.value.result.returncode == 7
    assert (tmp_path / "stdout.log").read_text() == "worker out"
    assert (tmp_path / "stderr.log").read_text() == "cuda failed"


def test_main_returns_nonzero_and_writes_no_fake_scores_on_worker_failure(
    tmp_path, monkeypatch
):
    video = tmp_path / "video.mp4"
    video.touch()
    stitching = tmp_path / "stitching"
    stitching.mkdir()
    vbench = tmp_path / "VBench-2.0"
    cache = vbench / ".cache/vbench2"
    clip = vbench / "clip"
    for path in (cache, clip):
        path.mkdir(parents=True)
    output = tmp_path / "output"
    monkeypatch.setattr(
        main_script,
        "build_human_anomaly_manifest",
        lambda *args: ([HumanAnomalyInput(0, 0, 1, [0, 0, 10, 20], 0.9)], [], 100, 100),
    )
    monkeypatch.setattr(main_script, "_git_commit", lambda path: "commit")

    def fail(command, stdout_path, stderr_path, working_directory=None):
        assert working_directory == vbench
        stdout_path.write_text("out")
        stderr_path.write_text("failed")
        raise VBenchWorkerError(BackendResult(list(command), 9, "out", "failed", 0.1))

    monkeypatch.setattr(main_script, "run_vbench_worker", fail)
    code = main_script.main([
        "--video", str(video), "--stitching-dir", str(stitching),
        "--output-dir", str(output), "--vbench-root", str(vbench),
        "--vbench-cache-dir", str(cache), "--vbench-clip-model", str(clip),
    ])
    assert code == 1
    assert not (output / "human_anomaly_frames.jsonl").exists()
    assert not (output / "human_anomaly_summary.json").exists()
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "failed"
