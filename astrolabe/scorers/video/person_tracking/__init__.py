"""Person detection and multi-object tracking for video preprocessing."""

from .schemas import Detection, FrameDetections, TrackStatistics, VideoInfo, VideoTrackingResult

__all__ = [
    "Detection",
    "FrameDetections",
    "TrackStatistics",
    "VideoInfo",
    "VideoTrackingResult",
]
