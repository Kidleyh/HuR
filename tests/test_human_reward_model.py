from pathlib import Path
from types import SimpleNamespace

from astrolabe.scorers.video.human_reward import HumanRewardConfig, HumanRewardModel
from astrolabe.scorers.video.person_tracking.schemas import (
    DetectionSummary, FrameDetections, TrackedDetection, VideoInfo, VideoTrackingResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tracking_result(video: Path) -> VideoTrackingResult:
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
    frames = [
        FrameDetections(0, 0.0, tracked_detections=[detections[0]]),
        FrameDetections(1, 0.1, tracked_detections=[detections[1]]),
    ]
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
        "reward": 0.5, "micro_score": 0.5, "macro_score": 0.5,
        "logical_track_count": 1, "observed_person_frames": 2,
        "scored_person_frames": 2, "abnormal_person_frames": 1,
        "failed_person_frames": 0,
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
