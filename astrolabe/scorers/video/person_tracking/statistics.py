"""Track-level descriptive statistics."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import DefaultDict, Dict, List

from .schemas import DetectionSummary, FrameDetections, TrackStatistics, TrackedDetection


def compute_track_statistics(
    frames: List[FrameDetections], total_video_frames: int
) -> List[TrackStatistics]:
    """Compute coverage, confidence, area, and gap statistics without filtering tracks."""
    if total_video_frames < 0:
        raise ValueError("total_video_frames must be non-negative")
    observations: DefaultDict[int, Dict[int, TrackedDetection]] = defaultdict(dict)
    for frame in frames:
        for detection in frame.tracked_detections:
            previous = observations[detection.track_id].get(frame.frame_index)
            if previous is None or detection.confidence > previous.confidence:
                observations[detection.track_id][frame.frame_index] = detection

    output: List[TrackStatistics] = []
    for track_id in sorted(observations):
        by_frame = observations[track_id]
        indices = sorted(by_frame)
        detections = [by_frame[index] for index in indices]
        start_frame, end_frame = indices[0], indices[-1]
        observed = len(indices)
        span = end_frame - start_frame + 1
        missing_gaps = [right - left - 1 for left, right in zip(indices, indices[1:])]
        output.append(
            TrackStatistics(
                track_id=track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                num_observed_frames=observed,
                global_coverage=observed / total_video_frames if total_video_frames else 0.0,
                span_coverage=observed / span,
                mean_confidence=mean(item.confidence for item in detections),
                median_confidence=median(item.confidence for item in detections),
                mean_bbox_area_ratio=mean(item.bbox_area_ratio for item in detections),
                median_bbox_area_ratio=median(item.bbox_area_ratio for item in detections),
                max_missing_gap=max(missing_gaps, default=0),
            )
        )
    return output


def compute_detection_summary(
    frames: List[FrameDetections], total_video_frames: int
) -> DetectionSummary:
    """Summarize raw-person presence and successful ByteTrack associations."""
    if total_video_frames < 0:
        raise ValueError("total_video_frames must be non-negative")
    raw_confidences = [
        detection.confidence for frame in frames for detection in frame.raw_detections
    ]
    total_tracked = sum(len(frame.tracked_detections) for frame in frames)
    associated_raw = 0
    for frame in frames:
        raw_indices = {detection.detection_index for detection in frame.raw_detections}
        associated_indices = {
            detection.source_detection_index
            for detection in frame.tracked_detections
            if detection.source_detection_index in raw_indices
        }
        associated_raw += len(associated_indices)
    total_raw = len(raw_confidences)
    divisor = total_video_frames or 1
    return DetectionSummary(
        frames_with_raw_person=sum(bool(frame.raw_detections) for frame in frames),
        frames_with_tracked_person=sum(bool(frame.tracked_detections) for frame in frames),
        raw_person_frame_coverage=(
            sum(bool(frame.raw_detections) for frame in frames) / divisor
            if total_video_frames
            else 0.0
        ),
        tracked_person_frame_coverage=(
            sum(bool(frame.tracked_detections) for frame in frames) / divisor
            if total_video_frames
            else 0.0
        ),
        total_raw_detections=total_raw,
        total_tracked_detections=total_tracked,
        untracked_raw_detections=total_raw - associated_raw,
        mean_raw_confidence=mean(raw_confidences) if raw_confidences else 0.0,
        median_raw_confidence=median(raw_confidences) if raw_confidences else 0.0,
        min_raw_confidence=min(raw_confidences, default=0.0),
    )
