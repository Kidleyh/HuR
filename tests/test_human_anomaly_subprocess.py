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
    assert command[command.index("--device") + 1] == "cuda:0"
    assert any(item.startswith("VBENCH2_CACHE_DIR=") for item in command)
    assert any(item.startswith("VBENCH2_CLIP_TEXT_MODEL=") for item in command)
    assert "--crop-batch-size" in command and "64" in command


@pytest.mark.parametrize(
    ("device", "visible", "worker_device"),
    [
        ("cuda:0", "0", "cuda:0"),
        ("cuda:1", "1", "cuda:0"),
        ("cuda:3", "3", "cuda:0"),
        ("cuda", "0", "cuda:0"),
        ("cpu", "", "cpu"),
    ],
)
def test_worker_device_mapping(tmp_path, device, visible, worker_device):
    options = kwargs(tmp_path)
    options["device"] = device
    command = build_worker_command(**options)
    assert f"CUDA_VISIBLE_DEVICES={visible}" in command
    assert command[command.index("--device") + 1] == worker_device


def write_complete_result(output, status="success"):
    output.mkdir()
    payloads = {
        "human_anomaly_input.jsonl": '{"frame_index": 0}\n',
        "human_anomaly_frames.jsonl": '{"frame_index": 0, "logical_track_id": 0}\n',
        "human_anomaly_tracks.json": "[]\n",
        "human_anomaly_summary.json": "{}\n",
        "run_manifest.json": json.dumps({"status": status}) + "\n",
        "worker_stdout.log": "worker output\n",
        "worker_stderr.log": "worker warnings\n",
    }
    for name, content in payloads.items():
        (output / name).write_text(content)


def test_complete_result_rejects_zero_byte_file(tmp_path):
    output = tmp_path / "output"
    write_complete_result(output)
    (output / "worker_stdout.log").write_text("")
    assert main_script._is_complete(output) is False


def test_complete_result_rejects_failed_manifest(tmp_path):
    output = tmp_path / "output"
    write_complete_result(output, status="failed")
    assert main_script._is_complete(output) is False


@pytest.mark.parametrize(
    "name",
    ["run_manifest.json", "human_anomaly_frames.jsonl",
     "human_anomaly_tracks.json", "human_anomaly_summary.json"],
)
def test_complete_result_rejects_corrupt_json(tmp_path, name):
    output = tmp_path / "output"
    write_complete_result(output)
    (output / name).write_text("not json\n")
    assert main_script._is_complete(output) is False


def test_complete_result_accepts_valid_outputs(tmp_path):
    output = tmp_path / "output"
    write_complete_result(output)
    assert main_script._is_complete(output) is True


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


def test_main_rejects_empty_manifest_before_worker(tmp_path, monkeypatch):
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
        main_script, "build_human_anomaly_manifest", lambda *args: ([], [], 100, 100)
    )
    monkeypatch.setattr(main_script, "_git_commit", lambda path: "commit")
    monkeypatch.setattr(
        main_script, "run_vbench_worker",
        lambda *args, **kwargs: pytest.fail("worker must not run for an empty manifest"),
    )
    code = main_script.main([
        "--video", str(video), "--stitching-dir", str(stitching),
        "--output-dir", str(output), "--vbench-root", str(vbench),
        "--vbench-cache-dir", str(cache), "--vbench-clip-model", str(clip),
    ])
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert code == 1
    assert run_manifest["status"] == "failed"
    assert "No valid person-frame entries" in run_manifest["message"]
    assert not (output / "human_anomaly_summary.json").exists()


def test_main_rejects_all_unscored_worker_output(tmp_path, monkeypatch):
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
    entry = HumanAnomalyInput(0, 0, 1, [0, 0, 10, 20], 0.9)
    monkeypatch.setattr(
        main_script, "build_human_anomaly_manifest",
        lambda *args: ([entry], [], 100, 100),
    )
    monkeypatch.setattr(main_script, "_git_commit", lambda path: "commit")

    def unscored(command, stdout_path, stderr_path, working_directory=None):
        output_index = command.index("--output-jsonl") + 1
        Path(command[output_index]).write_text(json.dumps({
            "frame_index": 0, "logical_track_id": 0,
            "human": {"scored": False},
        }) + "\n")
        stdout_path.write_text("worker output\n")
        stderr_path.write_text("worker warning\n")
        return BackendResult(list(command), 0, "worker output", "worker warning", 0.1)

    monkeypatch.setattr(main_script, "run_vbench_worker", unscored)
    code = main_script.main([
        "--video", str(video), "--stitching-dir", str(stitching),
        "--output-dir", str(output), "--vbench-root", str(vbench),
        "--vbench-cache-dir", str(cache), "--vbench-clip-model", str(clip),
    ])
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert code == 1
    assert run_manifest["status"] == "failed"
    assert run_manifest["message"] == "Worker produced no scored person frames"
    assert (output / "worker_stdout.log").is_file()
    assert (output / "worker_stderr.log").is_file()
    assert not (output / "human_anomaly_tracks.json").exists()
    assert not (output / "human_anomaly_summary.json").exists()
