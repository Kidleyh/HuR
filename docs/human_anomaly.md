# Stage 1.3: Per-Person VBench Human Anomaly

HuR owns orchestration, logical-track identity, manifests, aggregation, validation,
and outputs. VBench 2.0 remains an external read-only dependency that supplies
YOLO-World, Human/Face/Hand classifiers, official transforms, `smart_cut`, weights,
thresholds, and its independent Conda environment. HuR is not installed into that
environment and no VBench source or model weight is copied into this repository.

## Process boundary

The HuR process reads `stitched_detections.jsonl`, clips and deduplicates each
`(frame_index, logical_track_id)` person box, then writes
`human_anomaly_input.jsonl`. It invokes the HuR-owned worker using:

```text
conda run -n vbench2-human-anomaly env ... python scripts/vbench_human_anomaly_worker.py
```

`shell=True` is never used. `PYTHONPATH` contains both the HuR and VBench roots.
The worker does not run a human detector: HuR bboxes are the human inputs. One
YOLO-World instance detects only face and hand inside each human crop, parameterized
with `[["face"], ["hand"], [" "]]`. Official Human/Face/Hand classifiers and
`smart_cut` then score their crops.

## Environment

- External root: `/gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0`
- Conda environment: `/root/miniconda3/envs/vbench2-human-anomaly`
- Cache: `VBench-2.0/.cache/vbench2`
- CLIP variables: `VBENCH2_CLIP_TEXT_MODEL` and `HF_HOME` are set by HuR.
- H100 requires an MMCV build containing CUDA sm90 kernels. An incompatible MMCV
  build is a worker initialization failure, never a skipped crop.

## Official thresholds

```json
{"human": 0.4545454545454546, "face": 0.30303030303030304, "hand": 0.3232}
```

Classifier output dimension 0 is the abnormal probability. A person is abnormal
when human, any detected face, or any detected hand exceeds its official threshold.
No face or hand detection is neutral and is never converted into an anomaly.

## Input and outputs

Each input line contains `frame_index`, `logical_track_id`, original
`source_track_id`, clipped `bbox_xyxy`, and `detection_confidence`. Invalid boxes are
excluded with an auditable reason; duplicate logical person-frames retain the
highest-confidence box.

The output directory contains:

- `human_anomaly_input.jsonl`: HuR-to-worker manifest.
- `human_anomaly_frames.jsonl`: human/face/hand probabilities and person decision.
- `human_anomaly_tracks.json`: logical-track scores and observation quality.
- `human_anomaly_summary.json`: video micro/macro scores and failure counts.
- `run_manifest.json`: commits, environment versions, weights, command, thresholds,
  exit code, and runtime.
- `worker_stdout.log` and `worker_stderr.log`.
- `human_anomaly.mp4` when `--visualize` is requested.

`anatomy_quality_score = 1 - abnormal_frames / scored_frames`.
`video_micro_score` pools scored person-frames; `video_macro_score` averages valid
logical-track anatomy scores. Observation quality separately reports classifier
coverage, box area, boundary truncation, and face/hand detection coverage. Raw
absence of face/hand is not an abnormal observation.

## Command

Run this from the HuR repository; the worker itself is launched in the separate
VBench Conda environment:

```bash
python scripts/run_person_human_anomaly.py \
  --video /absolute/path/to/video.mp4 \
  --stitching-dir outputs/sample_tracklet_stitching \
  --output-dir outputs/sample_human_anomaly \
  --vbench-root /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0 \
  --vbench-cache-dir /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/vbench2 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32 \
  --vbench-conda-env vbench2-human-anomaly \
  --device cuda:0 \
  --visualize \
  --overwrite
```

## Limitations

Human Anomaly evaluates single-frame anatomy and visible parts. It does not evaluate
motion timing, temporal physical plausibility, identity crossings, interactions,
pose, SMPL, or dynamics. It does not change HuR logical tracks and does not solve
multi-person identity ambiguity.
