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
    frame: Any, logical_track_id: int, item: Mapping[str, Any]
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
                    _draw_person(frame, logical_track_id, item)
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
