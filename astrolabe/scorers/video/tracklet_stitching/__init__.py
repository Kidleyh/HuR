"""Offline geometric and raw-detection-based ByteTrack tracklet stitching."""

from .io import build_tracklets, load_tracking_input
from .schemas import StitchingConfig, Tracklet
from .stitcher import stitch_tracking

__all__ = [
    "StitchingConfig",
    "Tracklet",
    "build_tracklets",
    "load_tracking_input",
    "stitch_tracking",
]
