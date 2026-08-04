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
`--overwrite`, `--no-half`, and `--no-save-visualization`. Supported extensions
are mp4, mov, avi, mkv, and webm. A complete existing result is skipped by default.

All ByteTrack values can be replaced by another `--tracker-config`, or overridden
with `--track-high-thresh`, `--track-low-thresh`, `--new-track-thresh`,
`--track-buffer`, `--match-thresh`, and `--fuse-score`/`--no-fuse-score`.

## Outputs and schema

Each `<output-dir>/<video-stem>/` contains `detections.jsonl` (one row per frame,
including empty frames), `detections.csv` (one row per tracked box),
`tracks_summary.json`, and optionally `tracked.mp4` at source size and FPS.

Detection records contain integer track/class IDs, class name, confidence, clipped
pixel `bbox_xyxy`, top-left-origin `bbox_xywh`, normalized xyxy, and area ratio.
Schema version 1.0 statistics contain start/end, observed-frame count, global/span
coverage, confidence/area means and medians, and maximum internal missing gap.
Missing frames are not interpolated and tracks are not filtered. A failed video
gets `error.json` and folder processing continues.

## ByteTrack defaults

Detection `conf=0.10` preserves candidates for the second association stage below
`track_high_thresh=0.25`. Other defaults are `track_low_thresh=0.10`,
`new_track_thresh=0.25`, `track_buffer=30`, `match_thresh=0.80`, and
`fuse_score=true`. A fresh model/predictor per video prevents tracker-state leakage.

## CPU and common errors

Use `--device cpu --no-half`; half precision is automatically disabled on CPU.
YOLOv8x will be slow. Errors explicitly report missing weights/person class/lap,
unreadable video, invalid FPS, unavailable CUDA, and VideoWriter failures.
Install `lapx>=0.5.2` if the `lap` import is missing.

## Limitations

- Detection and tracking only; no completeness or main-person decision.
- No ViTPose, keypoints, GVHMR, SMPL/SMPL-X, physical score, or reward integration.
- ByteTrack has no ReID and cannot eliminate ID switches in severe crossings.
- Crowded scenes may later use BoT-SORT or ReID.
- The rendered video is for review only, never downstream algorithm input.
