# Person tracking

This independent preprocessing module runs YOLOv8x person detection, ByteTrack,
standard exports, statistics, and an OpenCV review video. It does not touch
`astrolabe/rewards.py`.

## Install

```bash
conda create -n phymotion-track python=3.10 -y
conda activate phymotion-track
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-tracking.txt
python scripts/check_tracking_env.py
```

Use `--require-weights` and/or `--require-cuda` for strict deployment checks.
The default check permits CPU mode and missing weights but always fails when a
required Python dependency is unavailable.

Without Conda, use `python3.10 -m venv .venv`, activate it, and run the same pip
commands. Do not install into the global interpreter.

## Weights

YOLOv8x weights are searched in order: `--weights`, `$YOLO_WEIGHTS`,
`$GVHMR_ROOT/inputs/checkpoints/yolo/yolov8x.pt`, then
`checkpoints/yolo/yolov8x.pt`. Network download is disabled unless
`--allow-download` is explicitly passed.

## Usage

```bash
python scripts/run_person_tracking.py \
  --input path/to/video.mp4 --output-dir outputs/person_tracking \
  --weights /path/to/yolov8x.pt \
  --tracker-config configs/bytetrack_person.yaml \
  --device 0 --conf 0.10 --iou 0.70 --imgsz 640 \
  --half --save-visualization
```

For folders add `--recursive`; optional controls include `--max-videos`,
`--overwrite`, `--no-half`, `--no-save-raw-csv`, and `--no-save-visualization`. Supported extensions
are mp4, mov, avi, mkv, and webm. A complete existing result is skipped by default.
Recursive directory outputs preserve the relative input path, so `a/sample.mp4`
and `b/sample.mp4` cannot overwrite each other.

All ByteTrack values can be replaced by another `--tracker-config`, or overridden
with `--track-high-thresh`, `--track-low-thresh`, `--new-track-thresh`,
`--track-buffer`, `--match-thresh`, and `--fuse-score`/`--no-fuse-score`.

## Outputs and schema

Each output contains `detections.jsonl`, `raw_detections.csv`,
`tracked_detections.csv`, `detections.csv`, `tracks_summary.json`, and optionally
`tracked.mp4`. The visualization is encoded as H.264/yuv420p with faststart for
browser and desktop-player compatibility. `detections.csv` is a compatibility copy identical to
`tracked_detections.csv`.

**Raw detections** are direct YOLO person candidates and have a per-frame
`detection_index`, but no identity. They support later human-presence and
low-confidence analysis. **Tracked detections** are candidates successfully
associated by ByteTrack; they have `track_id` and an optional
`source_detection_index` pointing to the raw candidate. They support identity
continuity and trajectory analysis.

Schema 1.1 writes one JSON object per decoded frame, including empty frames:

```json
{
  "frame_index": 1,
  "timestamp_sec": 0.0625,
  "raw_detections": [
    {"detection_index": 0, "class_name": "person", "confidence": 0.18}
  ],
  "tracked_detections": []
}
```

Full records also contain clipped pixel `bbox_xyxy`, top-left-origin `bbox_xywh`,
normalized xyxy, and area ratio. The summary includes raw/tracked frame coverage,
counts, untracked raw count, raw confidence statistics, and existing per-track
coverage/gap statistics. Missing frames are not interpolated and tracks are not
filtered.

Files are completed in a temporary sibling directory and promoted only after
successful inference, video encoding, and serialization. Failed reruns preserve
previous complete outputs. Successful reruns remove stale module-generated files
and `error.json`, while retaining unrelated user files.

## ByteTrack defaults

Detection `conf=0.10` preserves candidates for the second association stage below
`track_high_thresh=0.25`. Other defaults are `track_low_thresh=0.10`,
`new_track_thresh=0.25`, `track_buffer=30`, `match_thresh=0.80`, and
`fuse_score=true`. YOLO loads once per tracker object and runs exactly one predict
call per frame. A fresh `BYTETracker` is constructed for each video so IDs and
Kalman state never leak between videos. A detector confidence above ByteTrack's
high threshold emits a warning because low-score association becomes unavailable.

## CPU and common errors

Use `--device cpu --no-half`; half precision is automatically disabled on CPU.
YOLOv8x will be slow. Errors explicitly report missing weights/person class/lap,
unreadable video, invalid FPS, unavailable CUDA, and VideoWriter failures.
Install `lapx>=0.5.2` if the `lap` import is missing.

## Limitations

- Detection and tracking only; no completeness or main-person decision.
- Raw detections are retained but no human-completeness score is computed.
- No ViTPose, keypoints, GVHMR, SMPL/SMPL-X, physical score, or reward integration.
- ByteTrack has no ReID and cannot eliminate ID switches in severe crossings.
- Crowded scenes may later use BoT-SORT or ReID.
- The rendered video is for review only, never downstream algorithm input.
