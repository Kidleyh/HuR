# Offline Tracklet Stitching

Stage 1.2a repairs short ByteTrack identity fragmentation without rerunning YOLO or
ByteTrack. It reads schema-1.1 tracking artifacts and assigns every source track to
exactly one deterministic logical track. It does not select a main person or remove
background people.

## Identity levels

- `track_id` is the original immutable ByteTrack identity.
- `logical_track_id` is the offline identity. One logical identity may contain a
  chain such as `source_track_ids: [1, 3, 7]`.

Raw detections remain evidence only. They never become tracked observations and do
not change coverage statistics.

## Algorithm

Candidate edges are directional and require the first tracklet to end before the
second starts. Defaults permit gaps of at most five frames. Motion is estimated by
linear regression over the final five observations of normalized `cx`, `cy`,
`log(width)`, and `log(height)`. A single observation uses zero velocity.

Before scoring, candidates must pass hard gates for temporal overlap, gap length,
predicted center distance, area-ratio change, aspect-ratio change, and prediction
validity. Rejected edges retain explicit reasons.

The score combines:

| Component | Default weight | Meaning |
|---|---:|---|
| time | 0.15 | Exponential penalty for gap length |
| motion | 0.35 | Gaussian predicted-center consistency |
| predicted IoU | 0.20 | Predicted box overlap with the next first box |
| scale | 0.10 | Area and aspect-ratio consistency |
| raw bridge | 0.20 | Compatible raw-person evidence inside the gap |

For each gap frame, raw bridge scoring linearly interpolates an expected box and
selects the best compatible raw detection. The same raw detection may support
multiple candidate scores because scoring does not consume or modify detections.

Edges at or above `merge_threshold` enter a global maximum-weight one-to-one
assignment implemented with `scipy.optimize.linear_sum_assignment` and dummy
unmatched columns. Every tracklet has at most one predecessor and successor. A
selected edge is merged only when both its outgoing and incoming assignment margins
meet `minimum_assignment_margin`. Ambiguous selected edges become `uncertain` and
are not joined.

## Installation and configuration

Use the existing `phymotion-track` environment. Defaults are in
`configs/tracklet_stitching.yaml`; all primary decision thresholds can be overridden
by the CLI.

## CLI

Single tracking result:

```bash
python scripts/run_tracklet_stitching.py \
  --input outputs/person_tracking/video_001 \
  --output-dir outputs/tracklet_stitching \
  --config configs/tracklet_stitching.yaml \
  --save-visualization \
  --overwrite
```

Recursive batch mode preserves the input-relative directory layout:

```bash
python scripts/run_tracklet_stitching.py \
  --input outputs/person_tracking \
  --output-dir outputs/tracklet_stitching \
  --recursive \
  --max-gap-frames 5 \
  --merge-threshold 0.75 \
  --uncertain-threshold 0.55 \
  --minimum-assignment-margin 0.08
```

The default config is resolved relative to the project, so the script can be invoked
from another working directory.

## Outputs

The source `detections.jsonl`, CSVs, summary, and visualization are never modified.
Each result receives an independent output directory containing:

- `tracklet_stitching.json`: config, source-to-logical mapping, all explainable
  merged/uncertain/rejected edges, logical tracks, warnings, and runtime.
- `stitched_detections.jsonl`: original per-frame raw data and tracked detections,
  with `logical_track_id` added to each tracked detection.
- `stitched_tracks_summary.json`: logical-track and aggregate counts/statistics.
- `stitched.mp4`: optional H.264 visualization labelled `T3 -> L0`.
- `stitching_error.json`: per-result failure report.

Outputs are built in a temporary directory. Failed serialization or visualization
does not replace prior complete stitching output, and temporary files are cleaned.
If the source video no longer exists, data outputs still succeed, visualization is
skipped, and a warning is recorded.

## Current limitations

- No appearance features or ReID are used.
- Crossings between people can still produce an incorrect identity.
- Long occlusions and long departures/re-entry are intentionally not handled.
- Camera cuts are not detected or handled.
- The conservative defaults prefer missed merges over aggressive false merges.
- Uncertain edges are diagnostic only and never automatically stitched.
- No pose, ViTPose, GVHMR, SMPL, physics scoring, or main-person selection is used.
