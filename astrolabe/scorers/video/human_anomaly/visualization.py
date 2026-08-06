"""Optional anomaly visualization; never used for numeric aggregation."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import imageio_ffmpeg


def write_anomaly_visualization(
    video_path: Path, results: Sequence[Dict[str, Any]], destination: Path
) -> None:
    grouped = defaultdict(list)
    for result in results:
        grouped[int(result["frame_index"])].append(result)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Video cannot be opened for visualization: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    intermediate = destination.with_name("human_anomaly_intermediate.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create anomaly visualization writer")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for item in grouped.get(frame_index, []):
                human = item.get("human", {})
                color = (0, 255, 255) if not human.get("scored") else (
                    (0, 0, 255) if item.get("person_abnormal") else (0, 255, 0)
                )
                x1, y1, x2, y2 = [int(round(value)) for value in item["bbox_xyxy"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                probability = human.get("abnormal_probability")
                probability_text = "NA" if probability is None else f"{probability:.3f}"
                cv2.putText(
                    frame, f"L{item['logical_track_id']} human={probability_text}",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2, cv2.LINE_AA,
                )
                for category, part_color in (("faces", (255, 128, 0)), ("hands", (255, 0, 255))):
                    for part in item.get(category, []):
                        px1, py1, px2, py2 = [int(round(value)) for value in part["bbox_xyxy"]]
                        cv2.rectangle(frame, (px1, py1), (px2, py2), part_color, 1)
                        cv2.putText(
                            frame, f"{category[:-1]}={part['abnormal_probability']:.3f}",
                            (px1, max(18, py1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, part_color, 1, cv2.LINE_AA,
                        )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    completed = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i",
         str(intermediate), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(destination)],
        capture_output=True, text=True, check=False,
    )
    intermediate.unlink(missing_ok=True)
    if completed.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"Anomaly visualization encoding failed: {completed.stderr}")
