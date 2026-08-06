from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.scorers.video.human_reward import HumanRewardConfig, HumanRewardModel
from astrolabe.scorers.video.human_reward.model import _PreparedVideo
from astrolabe.scorers.video.person_tracking.schemas import (
    DetectionSummary, FrameDetections, TrackedDetection, VideoInfo, VideoTrackingResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tracking_result(video: Path, with_person: bool = True) -> VideoTrackingResult:
    detections = [
        TrackedDetection.from_xyxy(
            track_id=1, class_id=0, class_name="person", confidence=0.9,
            bbox_xyxy=[1, 1, 11, 21], image_width=32, image_height=24,
            source_detection_index=0,
        ),
        TrackedDetection.from_xyxy(
            track_id=1, class_id=0, class_name="person", confidence=0.8,
            bbox_xyxy=[2, 1, 12, 21], image_width=32, image_height=24,
            source_detection_index=0,
        ),
    ]
    frames = (
        [
            FrameDetections(0, 0.0, tracked_detections=[detections[0]]),
            FrameDetections(1, 0.1, tracked_detections=[detections[1]]),
        ]
        if with_person
        else [FrameDetections(0, 0.0), FrameDetections(1, 0.1)]
    )
    return VideoTrackingResult(
        video=VideoInfo(str(video), 32, 24, 10.0, 2, 0.2), frames=frames,
        detector={}, tracker={}, processing={}, tracks=[],
        detection_summary=DetectionSummary(0, 2, 0.0, 1.0, 0, 2, 0, 0.0, 0.0, 0.0),
    )


def config(tmp_path: Path) -> HumanRewardConfig:
    return HumanRewardConfig(
        yolo_weights=tmp_path / "weights.pt",
        tracker_config=PROJECT_ROOT / "configs/bytetrack_person.yaml",
        stitching_config=PROJECT_ROOT / "configs/tracklet_stitching.yaml",
        vbench_root=tmp_path / "VBench-2.0",
        vbench_cache_dir=tmp_path / "cache",
        device="cpu", half=False,
    )


def test_score_passes_all_stages_in_memory_without_intermediate_files(
    tmp_path, monkeypatch
):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    (tmp_path / "weights.pt").write_bytes(b"weights")
    events = []
    tracked = tracking_result(video)

    class Tracker:
        def __init__(self, **kwargs):
            events.append("load_tracking")
            assert kwargs["allow_download"] is False

        def track_video_in_memory(self, source):
            events.append("tracking")
            assert source == str(video.resolve())
            return tracked

    def stitcher(data, stitching_config):
        events.append("stitching")
        assert data.frames is tracked.frames
        return SimpleNamespace(track_id_to_logical_track_id={1: 0})

    class Engine:
        def __init__(self, **kwargs):
            events.append("load_anomaly")
            assert kwargs["clip_model"] == (
                tmp_path / "VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32"
            ).resolve()

        def score_video(self, source, entries):
            events.append("anomaly")
            assert source == video.resolve()
            assert [(item.frame_index, item.logical_track_id) for item in entries] == [
                (0, 0), (1, 0)
            ]
            return [
                {
                    "frame_index": item.frame_index,
                    "logical_track_id": item.logical_track_id,
                    "human": {"scored": True, "abnormal": index == 1},
                    "faces": [], "hands": [], "person_abnormal": index == 1,
                }
                for index, item in enumerate(entries)
            ]

        def close(self):
            events.append("close_anomaly")

    monkeypatch.chdir(tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    model = HumanRewardModel(
        config(tmp_path), tracker_factory=Tracker, anomaly_engine_factory=Engine,
        stitcher=stitcher, release_callback=lambda: events.append("release"),
    )
    result = model.score(video)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    assert before == after
    assert events == [
        "load_tracking", "tracking", "stitching", "release",
        "load_anomaly", "anomaly", "close_anomaly", "release",
    ]
    assert result == {
        "valid": True, "reason": None,
        "reward": 0.5, "micro_score": 0.5, "macro_score": 0.5,
        "logical_track_count": 1, "observed_person_frames": 2,
        "scored_person_frames": 2, "abnormal_person_frames": 1,
        "failed_person_frames": 0,
        "visualization": None,
    }


def test_score_releases_tracking_before_loading_anomaly(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    (tmp_path / "weights.pt").write_bytes(b"weights")
    events = []

    class Tracker:
        def __init__(self, **kwargs):
            pass

        def track_video_in_memory(self, source):
            events.append("tracking")
            return tracking_result(video)

    class Engine:
        def __init__(self, **kwargs):
            assert events == ["tracking", "stitching", "released"]
            events.append("engine")

        def score_video(self, source, entries):
            return [
                {"frame_index": item.frame_index,
                 "logical_track_id": item.logical_track_id,
                 "human": {"scored": True, "abnormal": False},
                 "faces": [], "hands": [], "person_abnormal": False}
                for item in entries
            ]

        def close(self):
            pass

    def stitcher(data, stitching_config):
        events.append("stitching")
        return SimpleNamespace(track_id_to_logical_track_id={1: 0})

    HumanRewardModel(
        config(tmp_path), tracker_factory=Tracker, anomaly_engine_factory=Engine,
        stitcher=stitcher, release_callback=lambda: events.append("released"),
    ).score(video)
    assert events[:4] == ["tracking", "stitching", "released", "engine"]


def test_score_batch_reuses_models_preserves_order_and_writes_nothing(
    tmp_path, monkeypatch
):
    videos = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    for video in videos:
        video.write_bytes(b"video")
    events = []

    class Tracker:
        def __init__(self, **kwargs):
            events.append("tracker_init")

        def track_video_in_memory(self, source):
            video = Path(source)
            events.append(f"track:{video.stem}")
            return tracking_result(video)

    class Engine:
        def __init__(self, **kwargs):
            events.append("engine_init")

        def score_video(self, source, entries):
            events.append(f"anomaly:{Path(source).stem}")
            abnormal = Path(source).stem == "second"
            return [
                {
                    "frame_index": item.frame_index,
                    "logical_track_id": item.logical_track_id,
                    "human": {"scored": True, "abnormal": abnormal},
                    "faces": [], "hands": [], "person_abnormal": abnormal,
                }
                for item in entries
            ]

        def close(self):
            events.append("engine_close")

    def stitcher(data, stitching_config):
        events.append("stitch")
        return SimpleNamespace(track_id_to_logical_track_id={1: 0})

    monkeypatch.chdir(tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    results = HumanRewardModel(
        config(tmp_path), tracker_factory=Tracker, anomaly_engine_factory=Engine,
        stitcher=stitcher, release_callback=lambda: events.append("release"),
    ).score_batch(videos)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    assert before == after
    assert events.count("tracker_init") == 1
    assert events.count("engine_init") == 1
    assert events.index("release") < events.index("engine_init")
    assert [result["reward"] for result in results] == [1.0, 0.0]
    assert all(result["valid"] for result in results)
    assert events == [
        "tracker_init", "track:first", "stitch", "track:second", "stitch",
        "release", "engine_init", "anomaly:first", "anomaly:second",
        "engine_close", "release",
    ]


def test_no_person_and_one_failure_do_not_block_valid_video(tmp_path):
    videos = [
        tmp_path / "empty.mp4", tmp_path / "broken.mp4", tmp_path / "valid.mp4"
    ]
    for video in videos:
        video.write_bytes(b"video")
    counts = {"tracker": 0, "engine": 0, "anomaly": 0}

    class Tracker:
        def __init__(self, **kwargs):
            counts["tracker"] += 1

        def track_video_in_memory(self, source):
            video = Path(source)
            if video.stem == "broken":
                raise ValueError("mock decode failure")
            return tracking_result(video, with_person=video.stem != "empty")

    class Engine:
        def __init__(self, **kwargs):
            counts["engine"] += 1

        def score_video(self, source, entries):
            counts["anomaly"] += 1
            return [
                {
                    "frame_index": item.frame_index,
                    "logical_track_id": item.logical_track_id,
                    "human": {"scored": True, "abnormal": False},
                    "faces": [], "hands": [], "person_abnormal": False,
                }
                for item in entries
            ]

        def close(self):
            pass

    model = HumanRewardModel(
        config(tmp_path), tracker_factory=Tracker, anomaly_engine_factory=Engine,
        stitcher=lambda data, cfg: SimpleNamespace(
            track_id_to_logical_track_id={1: 0}
        ),
        release_callback=lambda: None,
    )
    results = model.score_batch(videos)

    assert results[0] == {
        "valid": False, "reason": "no_person_detected", "reward": None,
        "micro_score": None, "macro_score": None, "logical_track_count": 0,
        "observed_person_frames": 0, "scored_person_frames": 0,
        "abnormal_person_frames": 0, "failed_person_frames": 0,
        "visualization": None,
    }
    assert results[1]["valid"] is False
    assert results[1]["reason"].startswith("tracking_failed: ValueError:")
    assert results[2]["valid"] is True
    assert counts == {"tracker": 1, "engine": 1, "anomaly": 1}


def test_score_delegates_to_single_element_batch(tmp_path, monkeypatch):
    model = HumanRewardModel(config(tmp_path), release_callback=lambda: None)
    expected = {"valid": True, "reward": 0.75}
    calls = []

    def score_batch(paths):
        calls.append(paths)
        return [expected]

    monkeypatch.setattr(model, "score_batch", score_batch)
    video = tmp_path / "sample.mp4"
    assert model.score(video) is expected
    assert calls == [[video]]


def test_visualization_runs_after_model_release_without_reinference(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    destination = tmp_path / "result.mp4"
    events = []

    class Tracker:
        def __init__(self, **kwargs):
            events.append("tracker_init")

        def track_video_in_memory(self, source):
            events.append("tracking")
            return tracking_result(video)

    class Engine:
        def __init__(self, **kwargs):
            events.append("engine_init")

        def score_video(self, source, entries):
            events.append("anomaly")
            return [
                {
                    "frame_index": item.frame_index,
                    "logical_track_id": item.logical_track_id,
                    "bbox_xyxy": item.bbox_xyxy,
                    "human": {"scored": True, "abnormal": index == 1},
                    "faces": [], "hands": [], "person_abnormal": index == 1,
                }
                for index, item in enumerate(entries)
            ]

        def close(self):
            events.append("engine_close")

    def visualize(source, frame_results, summary, output):
        events.append("visualize")
        assert source == video.resolve()
        assert len(frame_results) == 2
        assert summary["reward"] == 0.5
        output.write_bytes(b"mp4")

    model = HumanRewardModel(
        config(tmp_path), tracker_factory=Tracker, anomaly_engine_factory=Engine,
        stitcher=lambda data, cfg: SimpleNamespace(
            track_id_to_logical_track_id={1: 0}
        ),
        release_callback=lambda: events.append("release"),
        visualization_writer=visualize,
    )
    result = model.score(video, visualization_output=destination)

    assert result["reward"] == 0.5
    assert result["visualization"] == str(destination.resolve())
    assert destination.read_bytes() == b"mp4"
    assert events == [
        "tracker_init", "tracking", "release", "engine_init", "anomaly",
        "engine_close", "release", "visualize",
    ]


def test_prepared_video_retains_only_anomaly_inputs():
    assert [field.name for field in fields(_PreparedVideo)] == [
        "video", "entries", "width", "height"
    ]


@pytest.mark.parametrize("video_paths", ["video.mp4", Path("video.mp4")])
def test_score_batch_rejects_single_path_value(tmp_path, video_paths):
    model = HumanRewardModel(config(tmp_path), release_callback=lambda: None)
    with pytest.raises(
        TypeError,
        match=r"score_batch expects a sequence of video paths; use score\(\) for one video",
    ):
        model.score_batch(video_paths)


def test_score_batch_empty_sequence_returns_empty_list(tmp_path):
    model = HumanRewardModel(config(tmp_path), release_callback=lambda: None)
    assert model.score_batch([]) == []


def test_one_anomaly_failure_does_not_block_later_video(tmp_path):
    videos = [tmp_path / "broken.mp4", tmp_path / "valid.mp4"]
    for video in videos:
        video.write_bytes(b"video")

    class Tracker:
        def __init__(self, **kwargs):
            pass

        def track_video_in_memory(self, source):
            return tracking_result(Path(source))

    class Engine:
        def __init__(self, **kwargs):
            pass

        def score_video(self, source, entries):
            if Path(source).stem == "broken":
                raise ValueError("mock crop failure")
            return [
                {
                    "frame_index": item.frame_index,
                    "logical_track_id": item.logical_track_id,
                    "human": {"scored": True, "abnormal": False},
                    "faces": [], "hands": [], "person_abnormal": False,
                }
                for item in entries
            ]

        def close(self):
            pass

    results = HumanRewardModel(
        config(tmp_path), tracker_factory=Tracker, anomaly_engine_factory=Engine,
        stitcher=lambda data, cfg: SimpleNamespace(
            track_id_to_logical_track_id={1: 0}
        ),
        release_callback=lambda: None,
    ).score_batch(videos)

    assert results[0]["valid"] is False
    assert results[0]["reason"].startswith("anomaly_failed: ValueError:")
    assert results[1]["valid"] is True


def test_cuda_failure_terminates_batch(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")

    class Tracker:
        def __init__(self, **kwargs):
            pass

        def track_video_in_memory(self, source):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        HumanRewardModel(
            config(tmp_path), tracker_factory=Tracker, release_callback=lambda: None
        ).score_batch([video])
