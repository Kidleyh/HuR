"""Model-free composite visualization for the in-memory Human Reward pipeline."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import cv2
import imageio_ffmpeg

from astrolabe.scorers.video.human_temporal.metrics import BODY_BONES
from astrolabe.scorers.video.human_temporal.hand_metrics import HAND_BONES
from .person_centric import build_frame_to_person_refs

NORMAL_COLOR = (0, 200, 0)
ABNORMAL_COLOR = (0, 0, 255)
UNSCORED_COLOR = (0, 255, 255)


def _state(scored: bool, abnormal: bool) -> Tuple[str, Tuple[int, int, int]]:
    if not scored:
        return "UNSCORED", UNSCORED_COLOR
    if abnormal:
        return "ABNORMAL", ABNORMAL_COLOR
    return "NORMAL", NORMAL_COLOR


def _probability_text(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def _clipped_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width - 1)),
        max(0, min(y2, height - 1)),
    )


def _draw_summary(frame: Any, summary: Mapping[str, Any]) -> None:
    video_score = summary.get("video_score", {})
    micro = video_score.get("micro_score")
    macro = video_score.get("macro_score")
    lines = [
        f"Micro score: {_probability_text(micro)}",
        f"Macro score: {_probability_text(macro)}",
        "Observed / Scored / Failed: "
        f"{summary.get('observed_person_frames', 0)} / "
        f"{summary.get('scored_person_frames', 0)} / "
        f"{summary.get('failed_person_frames', 0)}",
        f"Abnormal person frames: {summary.get('abnormal_person_frames', 0)}",
    ]
    panel_width = max(1, min(frame.shape[1] - 16, 570))
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 112), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame, line, (18, 32 + index * 23), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 1, cv2.LINE_AA,
        )


def _draw_person(
    frame: Any,
    logical_track_id: int,
    item: Mapping[str, Any],
    temporal_human: Mapping[str, Any],
    temporal_metric: Mapping[str, Any],
) -> None:
    height, width = frame.shape[:2]
    human = item.get("human", {})
    person_state, person_color = _state(
        bool(human.get("scored")), bool(item.get("person_abnormal"))
    )
    human_state, _ = _state(
        bool(human.get("scored")), bool(human.get("abnormal"))
    )
    x1, y1, x2, y2 = _clipped_box(item["bbox_xyxy"], width, height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
    label = (
        f"L{logical_track_id} person={person_state} "
        f"human={_probability_text(human.get('abnormal_probability'))}({human_state})"
    )
    cv2.putText(
        frame, label, (x1, y1 - 7 if y1 >= 130 else 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48, person_color, 2, cv2.LINE_AA,
    )
    pose = item.get("human_pose", {})
    keypoints = pose.get("keypoints_xy", [])
    keypoint_scores = pose.get("keypoint_scores", [])
    name_to_index = temporal_human.get("keypoint_name_to_index", {})
    threshold = float(temporal_human.get("keypoint_threshold", 0.3))
    skeleton_color = (255, 200, 0)
    valid_point_indices = set()
    for first_name, second_name in BODY_BONES:
        first_index = name_to_index.get(first_name)
        second_index = name_to_index.get(second_name)
        if (
            first_index is None or second_index is None
            or first_index >= len(keypoints) or second_index >= len(keypoints)
            or first_index >= len(keypoint_scores) or second_index >= len(keypoint_scores)
            or float(keypoint_scores[first_index]) < threshold
            or float(keypoint_scores[second_index]) < threshold
        ):
            continue
        first_point = tuple(int(round(value)) for value in keypoints[first_index])
        second_point = tuple(int(round(value)) for value in keypoints[second_index])
        cv2.line(frame, first_point, second_point, skeleton_color, 2, cv2.LINE_AA)
        valid_point_indices.update((first_index, second_index))
    for index in valid_point_indices:
        point = tuple(int(round(value)) for value in keypoints[index])
        cv2.circle(frame, point, 3, skeleton_color, -1, cv2.LINE_AA)
    if temporal_human:
        cv2.putText(
            frame,
            "Struct=" + _probability_text(temporal_metric.get("bone_length_jump"))
            + " Motion="
            + _probability_text(temporal_metric.get("joint_acceleration")),
            (x1, y1 - 28 if y1 >= 150 else 153),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            skeleton_color,
            1,
            cv2.LINE_AA,
        )
    for category in ("faces", "hands"):
        part_name = category[:-1].capitalize()
        for part in item.get(category, []):
            state, color = _state(
                bool(part.get("scored", True)), bool(part.get("abnormal"))
            )
            px1, py1, px2, py2 = _clipped_box(part["bbox_xyxy"], width, height)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)
            cv2.putText(
                frame,
                f"{part_name} {state} "
                f"{_probability_text(part.get('abnormal_probability'))}",
                (px1, py1 - 4 if py1 >= 130 else min(height - 4, py2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1, cv2.LINE_AA,
            )
            pose_key = "face_pose" if category == "faces" else "hand_pose"
            pose = part.get(pose_key, {})
            points = pose.get("keypoints_xy", [])
            scores = pose.get("keypoint_scores", [])
            point_color = (255, 0, 255) if category == "faces" else (0, 165, 255)
            for index, point in enumerate(points):
                if index < len(scores) and float(scores[index]) >= 0.3:
                    cv2.circle(
                        frame, tuple(int(round(value)) for value in point),
                        1 if category == "faces" else 2, point_color, -1, cv2.LINE_AA,
                    )
            if category == "hands":
                for start, end in HAND_BONES:
                    if (
                        end < len(points) and end < len(scores)
                        and float(scores[start]) >= 0.3 and float(scores[end]) >= 0.3
                    ):
                        cv2.line(
                            frame,
                            tuple(int(round(value)) for value in points[start]),
                            tuple(int(round(value)) for value in points[end]),
                            point_color, 1, cv2.LINE_AA,
                        )
                if part.get("side"):
                    cv2.putText(
                        frame, part["side"][0].upper(), (px1, py2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1, cv2.LINE_AA,
                    )
    temporal = item.get("_temporal_metrics", {})
    head_metric = temporal.get("head", {})
    hand_metric = temporal.get("hand", {})
    if head_metric:
        cv2.putText(
            frame,
            "Shape=" + _probability_text(head_metric.get("face_shape_jump"))
            + " Motion=" + _probability_text(head_metric.get("head_motion_acceleration")),
            (x1, min(height - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (255, 0, 255), 1, cv2.LINE_AA,
        )
    if hand_metric:
        cv2.putText(
            frame,
            "HandStruct=" + _probability_text(hand_metric.get("hand_bone_length_jump"))
            + " Motion=" + _probability_text(hand_metric.get("hand_joint_acceleration")),
            (x1, min(height - 8, y2 + 36)), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 165, 255), 1, cv2.LINE_AA,
        )


def write_human_reward_visualization(
    video_path: Path,
    result: Mapping[str, Any],
    destination: Path,
) -> None:
    """Render one browser-compatible MP4 and atomically promote it."""
    source = Path(video_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    persons = {
        int(person["logical_track_id"]): person for person in result["persons"]
    }
    frame_to_person_refs = build_frame_to_person_refs(result["persons"])
    temporal_metrics = {}
    head_metrics = {}
    hand_metrics = {}
    for person in result["persons"]:
        logical_id = int(person["logical_track_id"])
        human_temporal = person.get("temporal", {}).get("human", {})
        for metric in human_temporal.get("frame_metrics", []):
            temporal_metrics[(logical_id, int(metric["frame_index"]))] = metric
        for metric in person.get("temporal", {}).get("head", {}).get("frame_metrics", []):
            head_metrics[(logical_id, int(metric["frame_index"]))] = metric
        for side in ("left", "right"):
            for metric in person.get("temporal", {}).get("hand", {}).get(side, {}).get("frame_metrics", []):
                key = (logical_id, int(metric["frame_index"]))
                current = hand_metrics.setdefault(key, {})
                for name in ("hand_bone_length_jump", "hand_joint_acceleration"):
                    if metric.get(name) is not None:
                        current[name] = max(float(metric[name]), current.get(name, 0.0))

    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.tmp-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        intermediate = temporary / "intermediate.mp4"
        encoded = temporary / "encoded.mp4"
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Video cannot be opened for visualization: {source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0 or not math.isfinite(fps):
            capture.release()
            raise RuntimeError(
                f"Video has invalid visualization metadata: {width}x{height} at {fps} FPS"
            )
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            writer.release()
            raise RuntimeError("Could not create Human Reward visualization writer")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                _draw_summary(frame, result)
                for logical_track_id, person_frame_index in frame_to_person_refs.get(
                    frame_index, []
                ):
                    item = persons[logical_track_id]["frames"][person_frame_index]
                    visual_item = dict(item)
                    visual_item["_temporal_metrics"] = {
                        "head": head_metrics.get((logical_track_id, frame_index), {}),
                        "hand": hand_metrics.get((logical_track_id, frame_index), {}),
                    }
                    temporal_human = persons[logical_track_id].get(
                        "temporal", {}
                    ).get("human", {})
                    _draw_person(
                        frame,
                        logical_track_id,
                        visual_item,
                        temporal_human,
                        temporal_metrics.get((logical_track_id, frame_index), {}),
                    )
                writer.write(frame)
                frame_index += 1
        finally:
            capture.release()
            writer.release()
        if declared_frames > 0 and frame_index != declared_frames:
            raise RuntimeError(
                f"Visualization decoded {frame_index} frames; expected {declared_frames}"
            )
        completed = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                "-i", str(intermediate), "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded),
            ],
            capture_output=True, text=True, check=False,
        )
        if (
            completed.returncode != 0
            or not encoded.is_file()
            or encoded.stat().st_size == 0
        ):
            raise RuntimeError(
                f"Human Reward visualization encoding failed: {completed.stderr.strip()}"
            )
        os.replace(encoded, output)
