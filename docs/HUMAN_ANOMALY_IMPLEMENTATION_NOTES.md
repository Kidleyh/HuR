# Stage 1.3 Human Anomaly Implementation Notes

## Scope

Stage 1.3 adds per-logical-track VBench Human Anomaly analysis to HuR. HuR owns
manifest creation, logical identity, subprocess orchestration, validation,
aggregation, visualization, and output persistence. The external VBench checkout
and its `vbench2-human-anomaly` Conda environment remain separate and read-only.

The worker uses existing HuR person boxes instead of running the VBench human
detector. It reuses the official VBench Human, Face, and Hand classifiers,
transforms, `smart_cut`, thresholds, and one YOLO-World instance parameterized for
face and hand detection.

## Main changes

- Added `astrolabe/scorers/video/human_anomaly/` for schemas, manifest conversion,
  aggregation, subprocess execution, and visualization.
- Added `scripts/run_person_human_anomaly.py` as the HuR entry point.
- Added `scripts/vbench_human_anomaly_worker.py`, executed only inside the external
  VBench Conda environment.
- Extended `scripts/run_person_preprocessing_pipeline.py` with the optional
  `--human-anomaly` stage and `outputs/<name>_human_anomaly/` output.
- Added model-free unit tests and CI path filters.
- Added `docs/human_anomaly.md`; all executable commands remain centralized in
  `docs/RUN_COMMANDS.md`.

## Input and output

`human_anomaly_input.jsonl` contains one stable, deduplicated record per
`(frame_index, logical_track_id)` with the original source track, clipped person
bbox, and detector confidence. Invalid boxes are not sent to VBench and retain an
auditable failure reason.

Successful output contains:

- `human_anomaly_input.jsonl`
- `human_anomaly_frames.jsonl`
- `human_anomaly_tracks.json`
- `human_anomaly_summary.json`
- `run_manifest.json`
- `worker_stdout.log`
- `worker_stderr.log`
- `human_anomaly.mp4` when visualization is requested

Worker failure is propagated as a nonzero main-process result. Logs and a failed
run manifest are retained, while fake score files are never generated.

## Official anomaly semantics

The class-0 abnormal probability uses the official thresholds:

```json
{
  "human": 0.4545454545454546,
  "face": 0.30303030303030304,
  "hand": 0.3232
}
```

A person is abnormal when the human crop, any detected face, or any detected hand
is abnormal. A missing face or hand detection remains neutral.

## Validation

Model-free tests on the H100 node:

```text
Stage 1.3 targeted tests: 7 passed
Stage 1.3 plus pipeline tests: 14 passed
All non-integration tests: 74 passed, 1 skipped, 2 deselected
git diff --check: passed
```

The skipped existing environment branch reports that project-local YOLO weights
are installed; it is unrelated to Stage 1.3.

## H100 smoke test

Environment:

```text
GPU: NVIDIA H100 80GB HBM3
PyTorch: 2.5.1+cu118
MMCV: 2.2.0
CUDA available: true
```

Input video:

```text
/gemini/platform/public/aigc/human_guozz2/code/lyh/job/AnyFlow/assets/evaluation/example/videos/2.mp4
```

Input stitching result:

```text
outputs/custom_name_smoke_tracklet_stitching
```

Result:

```text
Logical tracks: 1
Observed person frames: 80
Scored person frames: 80
Failed person frames: 0
Abnormal person frames: 1
Anatomy quality score: 0.9875
Video micro score: 0.9875
Video macro score: 0.9875
Human anomaly rate: 0.0
Face anomaly rate: 0.0
Hand anomaly rate: 0.0125
Face detection coverage: 1.0
Hand detection coverage: 1.0
Worker runtime: 34.25 seconds
Total runtime: 34.30 seconds
Visualization: 80 frames at 16 FPS
```

Output:

```text
outputs/custom_name_smoke_human_anomaly
```

## Issue found during real validation

The official YOLO-World configuration contains paths relative to the VBench root.
Launching the worker with HuR as its current directory caused LVIS metadata lookup
to fail. HuR now starts the subprocess with the external VBench root as `cwd`.
VBench source and configuration files were not modified.

## Limitations

Human Anomaly evaluates single-frame anatomy. It does not evaluate temporal action
plausibility, modify logical tracks, run pose/SMPL, add ReID, or treat missing
face/hand detections as anomalies.
