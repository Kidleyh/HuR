from astrolabe.scorers.video.human_anomaly.schema import HumanAnomalyInput
from astrolabe.scorers.video.human_reward.person_centric import (
    build_frame_to_person_refs,
    build_person_centric_result,
)


def _entry(frame, logical, source, confidence=0.9):
    return HumanAnomalyInput(
        frame_index=frame,
        logical_track_id=logical,
        source_track_id=source,
        bbox_xyxy=[float(frame + 1), 2.0, float(frame + 11), 22.0],
        detection_confidence=confidence,
    )


def _result(entry, abnormal=False, with_parts=False):
    return {
        "frame_index": entry.frame_index,
        "logical_track_id": entry.logical_track_id,
        "source_track_id": entry.source_track_id,
        "bbox_xyxy": list(entry.bbox_xyxy),
        "human": {
            "scored": True, "scores": [0.8 if abnormal else 0.1, 0.2 if abnormal else 0.9],
            "abnormal_probability": 0.8 if abnormal else 0.1,
            "abnormal": abnormal,
        },
        "faces": [{"bbox_xyxy": [2, 3, 4, 5], "abnormal": False}] if with_parts else [],
        "hands": [{"bbox_xyxy": [6, 7, 8, 9], "abnormal": False}] if with_parts else [],
        "person_abnormal": abnormal,
        "failure_reason": None,
        "failures": [],
    }


def test_person_centric_join_preserves_tracks_gaps_results_and_video_scores():
    entries = [
        _entry(9, 0, 7, 0.79), _entry(0, 0, 1, 0.95),
        _entry(5, 0, 7, 0.85), _entry(3, 1, 2, 0.88),
    ]
    frame_results = [
        _result(entries[2], with_parts=True), _result(entries[3]),
        _result(entries[0], abnormal=True), _result(entries[1]),
    ]
    logical_tracks = [
        {
            "logical_track_id": 0, "source_track_ids": [1, 7],
            "start_frame": 0, "end_frame": 9, "num_observed_frames": 3,
            "num_fragments": 2, "max_internal_gap": 3,
            "global_coverage": 0.3, "span_coverage": 0.3,
            "mean_confidence": 0.863, "median_confidence": 0.85,
            "mean_bbox_area_ratio": 0.1, "median_bbox_area_ratio": 0.1,
        },
        {
            "logical_track_id": 1, "source_track_ids": [2],
            "start_frame": 3, "end_frame": 3, "num_observed_frames": 1,
            "num_fragments": 1, "max_internal_gap": 0,
        },
    ]
    track_scores = [
        {
            "logical_track_id": 0, "observed_frames": 3, "scored_frames": 3,
            "abnormal_frames": 1, "anatomy_quality_score": 2 / 3,
            "scored_frame_coverage": 1.0,
            "human_anomaly_rate": 1 / 3,
            "face_anomaly_rate": 0.0,
            "hand_anomaly_rate": 0.0,
            "face_detected_frames": 1,
            "face_detection_coverage": 1 / 3,
            "hand_detected_frames": 1,
            "hand_detection_coverage": 1 / 3,
            "median_bbox_area_ratio": 0.123,
            "boundary_truncation_rate": 0.25,
        },
        {
            "logical_track_id": 1, "observed_frames": 1, "scored_frames": 1,
            "abnormal_frames": 0, "anatomy_quality_score": 1.0,
            "scored_frame_coverage": 1.0,
            "human_anomaly_rate": 0.0,
            "face_anomaly_rate": 0.0,
            "hand_anomaly_rate": 0.0,
            "face_detected_frames": 0,
            "face_detection_coverage": 0.0,
            "hand_detected_frames": 0,
            "hand_detection_coverage": 0.0,
            "median_bbox_area_ratio": 0.08,
            "boundary_truncation_rate": 0.0,
        },
    ]
    summary = {
        "video_micro_score": 0.75, "video_macro_score": 5 / 6,
        "logical_track_count": 2, "observed_person_frames": 4,
        "scored_person_frames": 4, "abnormal_person_frames": 1,
        "failed_person_frames": 0,
    }

    result = build_person_centric_result(
        video={"path": "/video.mp4", "width": 1280, "height": 720,
               "fps": 24.0, "num_frames": 10},
        entries=entries, frame_results=frame_results,
        logical_tracks=logical_tracks, track_scores=track_scores, summary=summary,
    )

    assert [person["logical_track_id"] for person in result["persons"]] == [0, 1]
    first = result["persons"][0]
    assert first["track"] == {key: value for key, value in logical_tracks[0].items()
                              if key != "logical_track_id"}
    assert first["track"]["source_track_ids"] == [1, 7]
    assert [frame["frame_index"] for frame in first["frames"]] == [0, 5, 9]
    assert 1 not in [frame["frame_index"] for frame in first["frames"]]
    middle = first["frames"][1]
    assert middle["bbox_xyxy"] == entries[2].bbox_xyxy
    assert middle["detection_confidence"] == 0.85
    assert middle["human"] == frame_results[0]["human"]
    assert middle["faces"] == frame_results[0]["faces"]
    assert middle["hands"] == frame_results[0]["hands"]
    assert first["score"] == {
        "binary_score": 2 / 3, "observed_frames": 3, "scored_frames": 3,
        "failed_frames": 0, "scored_frame_coverage": 1.0,
        "abnormal_frames": 1,
        "human_anomaly_rate": 1 / 3,
        "face_anomaly_rate": 0.0,
        "hand_anomaly_rate": 0.0,
        "face_detected_frames": 1,
        "face_detection_coverage": 1 / 3,
        "hand_detected_frames": 1,
        "hand_detection_coverage": 1 / 3,
        "median_bbox_area_ratio": 0.123,
        "boundary_truncation_rate": 0.25,
    }
    assert first["temporal"] == {}
    assert result["video_score"] == {
        "reward": 0.75, "micro_score": 0.75, "macro_score": 5 / 6,
    }

    refs = build_frame_to_person_refs(result["persons"])
    assert refs == {0: [(0, 0)], 3: [(1, 0)], 5: [(0, 1)], 9: [(0, 2)]}
    logical_id, person_frame_index = refs[5][0]
    assert result["persons"][logical_id]["frames"][person_frame_index] is middle


def test_person_score_supports_unscored_person_without_division():
    entry = _entry(4, 3, 9)
    result = _result(entry)
    result["human"] = {"scored": False}
    result["person_abnormal"] = False
    built = build_person_centric_result(
        video={"path": "/video.mp4", "width": 32, "height": 24,
               "fps": 10.0, "num_frames": 5},
        entries=[entry], frame_results=[result], logical_tracks=[],
        track_scores=[{
            "logical_track_id": 3, "observed_frames": 1, "scored_frames": 0,
            "abnormal_frames": 0, "anatomy_quality_score": None,
            "scored_frame_coverage": 0.0,
            "human_anomaly_rate": 0.0,
            "face_anomaly_rate": 0.0,
            "hand_anomaly_rate": 0.0,
            "face_detected_frames": 0,
            "face_detection_coverage": 0.0,
            "hand_detected_frames": 0,
            "hand_detection_coverage": 0.0,
            "median_bbox_area_ratio": 0.12,
            "boundary_truncation_rate": 0.0,
        }],
        summary={"video_micro_score": 1.0, "video_macro_score": 1.0},
    )
    assert built["persons"][0]["score"] == {
        "binary_score": None, "observed_frames": 1, "scored_frames": 0,
        "failed_frames": 1, "scored_frame_coverage": 0.0,
        "abnormal_frames": 0,
        "human_anomaly_rate": 0.0,
        "face_anomaly_rate": 0.0,
        "hand_anomaly_rate": 0.0,
        "face_detected_frames": 0,
        "face_detection_coverage": 0.0,
        "hand_detected_frames": 0,
        "hand_detection_coverage": 0.0,
        "median_bbox_area_ratio": 0.12,
        "boundary_truncation_rate": 0.0,
    }
