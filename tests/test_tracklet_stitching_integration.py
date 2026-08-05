"""Integration test using an existing schema-1.1 person-tracking result."""

import json
import os
from pathlib import Path

import cv2
import pytest

from astrolabe.scorers.video.tracklet_stitching.io import load_tracking_input
from astrolabe.scorers.video.tracklet_stitching.schemas import StitchingConfig
from astrolabe.scorers.video.tracklet_stitching.serialization import write_stitching_outputs
from astrolabe.scorers.video.tracklet_stitching.stitcher import stitch_tracking

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_real_tracking_result(tmp_path: Path) -> None:
    candidates = []
    if os.environ.get("TRACKLET_STITCHING_TEST_INPUT"):
        candidates.append(Path(os.environ["TRACKLET_STITCHING_TEST_INPUT"]))
    candidates.extend((PROJECT_ROOT / "outputs/person_tracking_smoke").glob("*"))
    source = next(
        (
            path.resolve()
            for path in candidates
            if (path / "detections.jsonl").is_file()
            and (path / "tracks_summary.json").is_file()
        ),
        None,
    )
    if source is None:
        pytest.skip("existing schema-1.1 person tracking result not found")
    tracking = load_tracking_input(source)
    video = Path(str(tracking.summary["video"]["path"]))
    if not video.is_file():
        pytest.skip("source video referenced by tracking result no longer exists")
    result = stitch_tracking(tracking, StitchingConfig())
    output = tmp_path / source.name
    write_stitching_outputs(result, tracking.frames, video, output, True)

    for name in (
        "tracklet_stitching.json",
        "stitched_detections.jsonl",
        "stitched_tracks_summary.json",
        "stitched.mp4",
    ):
        assert (output / name).is_file()
    payload = json.loads((output / "tracklet_stitching.json").read_text())
    source_ids = {
        track["track_id"] for track in tracking.summary.get("tracks", [])
    }
    assert set(map(int, payload["track_id_to_logical_track_id"])) == source_ids
    lines = (output / "stitched_detections.jsonl").read_text().splitlines()
    assert len(lines) == tracking.summary["video"]["num_frames"]
    for line in lines:
        frame = json.loads(line)
        for detection in frame["tracked_detections"]:
            assert isinstance(detection["track_id"], int)
            assert isinstance(detection["logical_track_id"], int)
    capture = cv2.VideoCapture(str(output / "stitched.mp4"))
    assert capture.isOpened()
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    assert abs(count - len(lines)) <= 1
