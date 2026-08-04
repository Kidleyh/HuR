"""CLI path handling and cwd-independence tests."""

import subprocess
import sys
from pathlib import Path

from scripts.run_person_tracking import DEFAULT_TRACKER_CONFIG, output_dir_for_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_recursive_same_stem_outputs_do_not_collide(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    first = input_root / "a" / "sample.mp4"
    second = input_root / "b" / "sample.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    output_root = tmp_path / "outputs"
    assert output_dir_for_video(first, input_root, output_root) == output_root / "a/sample"
    assert output_dir_for_video(second, input_root, output_root) == output_root / "b/sample"


def test_single_file_spaces_and_multiple_extensions(tmp_path: Path) -> None:
    video = tmp_path / "a video.final.mp4"
    video.touch()
    assert output_dir_for_video(video, video, tmp_path / "out") == tmp_path / "out/a video.final"


def test_default_tracker_config_is_absolute_and_help_works_outside_repo(tmp_path: Path) -> None:
    assert DEFAULT_TRACKER_CONFIG.is_absolute()
    assert DEFAULT_TRACKER_CONFIG.is_file()
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/run_person_tracking.py"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--tracker-config" in completed.stdout
