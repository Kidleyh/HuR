"""OpenCV visualization for source and logical track identities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import imageio_ffmpeg

from astrolabe.scorers.video.person_tracking.schemas import FrameDetections


def _color(logical_id: int) -> Tuple[int, int, int]:
    return (
        64 + logical_id * 47 % 192,
        64 + logical_id * 89 % 192,
        64 + logical_id * 137 % 192,
    )


def write_stitched_video(
    video_path: Path,
    frames: Sequence[FrameDetections],
    mapping: Dict[int, int],
    destination: Path,
) -> None:
    """Draw T(source) to L(logical) labels and produce portable H.264 output."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Original video cannot be opened: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    intermediate = destination.with_name("stitched_intermediate.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"VideoWriter could not be created: {intermediate}")
    try:
        for frame_record in frames:
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Original video ended before frame {frame_record.frame_index}"
                )
            for detection in frame_record.tracked_detections:
                logical_id = mapping[detection.track_id]
                x1, y1, x2, y2 = (
                    int(round(value)) for value in detection.bbox_xyxy
                )
                color = _color(logical_id)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = (
                    f"T{detection.track_id} -> L{logical_id} "
                    f"{detection.confidence:.2f}"
                )
                cv2.putText(
                    image,
                    label,
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(image)
    finally:
        capture.release()
        writer.release()
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(intermediate),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    intermediate.unlink(missing_ok=True)
    if (
        completed.returncode != 0
        or not destination.is_file()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            f"H.264 stitched visualization failed: {completed.stderr.strip()}"
        )
