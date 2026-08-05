"""High-level offline stitching orchestration and logical-track statistics."""

from __future__ import annotations

import time
from statistics import mean, median
from typing import Dict, List, Sequence

from .candidates import generate_candidates
from .io import TrackingInput, build_tracklets
from .matching import build_logical_chains, select_merged_edges
from .schemas import LogicalTrackStatistics, StitchingConfig, StitchingResult, Tracklet


def _logical_statistics(
    chains: Sequence[Sequence[int]], tracklets: Sequence[Tracklet], total_frames: int
) -> List[LogicalTrackStatistics]:
    by_id = {item.track_id: item for item in tracklets}
    output = []
    for logical_id, chain in enumerate(chains):
        fragments = [by_id[track_id] for track_id in chain]
        observations = sorted(
            (frame, detection)
            for fragment in fragments
            for frame, detection in zip(fragment.frame_indices, fragment.detections)
        )
        indices = [item[0] for item in observations]
        detections = [item[1] for item in observations]
        start, end = indices[0], indices[-1]
        gaps = [right - left - 1 for left, right in zip(indices, indices[1:])]
        output.append(LogicalTrackStatistics(
            logical_track_id=logical_id, source_track_ids=list(chain), start_frame=start,
            end_frame=end, num_fragments=len(chain), num_observed_frames=len(indices),
            global_coverage=len(indices) / total_frames if total_frames else 0.0,
            span_coverage=len(indices) / (end - start + 1), max_internal_gap=max(gaps, default=0),
            mean_confidence=mean(item.confidence for item in detections),
            median_confidence=median(item.confidence for item in detections),
            mean_bbox_area_ratio=mean(item.bbox_area_ratio for item in detections),
            median_bbox_area_ratio=median(item.bbox_area_ratio for item in detections),
        ))
    return output


def stitch_tracking(data: TrackingInput, config: StitchingConfig) -> StitchingResult:
    started = time.perf_counter()
    tracklets = build_tracklets(data.frames)
    edges = generate_candidates(tracklets, data.frames, config)
    merged = select_merged_edges(tracklets, edges, config)
    chains = build_logical_chains(tracklets, merged)
    mapping: Dict[int, int] = {
        track_id: logical_id for logical_id, chain in enumerate(chains) for track_id in chain
    }
    if set(mapping) != {item.track_id for item in tracklets}:
        raise RuntimeError("Each source track must map to exactly one logical track")
    total_frames = int(data.summary["video"]["num_frames"])
    return StitchingResult(
        source_tracking_schema_version=str(data.summary["schema_version"]), config=config,
        track_id_to_logical_track_id=mapping, edges=edges,
        logical_tracks=_logical_statistics(chains, tracklets, total_frames),
        runtime_sec=time.perf_counter() - started,
    )
