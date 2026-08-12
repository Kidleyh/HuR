# Paired Human Reward Batch Implementation Notes

## Scope

- Added `scripts/run_human_reward_pairs.py` for directories whose immediate
  child folders contain `gt.mp4` (positive) and `render.mp4` (negative).
- The script ignores unrelated files such as sample JSON metadata.
- All videos are passed to one `HumanRewardModel.score_batch()` call so model
  lifecycle and reuse follow the existing in-memory batch pipeline.
- A single JSON document is written atomically after all pair results are
  available. Each side retains the complete Human Reward result, including
  person-centric frames, Human/Face/Hand details, temporal fields, person
  statistics, and video scores.

## Output schema

The top level contains `schema_version`, the absolute `input_dir`,
`pair_count`, `video_count`, and a stable `pairs` array. Each pair records its
directory `name` and two entries:

- `positive`: `kind=gt`, source video path, and the complete result.
- `negative`: `kind=render`, source video path, and the complete result.

Pair folders and results are deterministically ordered by folder name. An
incomplete pair is rejected explicitly instead of silently producing a partial
dataset.

## Verification (2026-08-10)

- Compile check: passed.
- Paired CLI tests: `3 passed`.
- Human Reward tests: `24 passed`.
- Full repository tests: `144 passed, 2 skipped`.
- `git diff --check`: passed.
- H100 smoke test: first pair `12_(112)_0`, two videos processed in one batch.
  - Positive (`gt.mp4`): valid, reward `0.9976689976689976`, 6 logical tracks,
    429 observed/scored person frames.
  - Negative (`render.mp4`): valid, reward `0.9462616822429907`, 6 logical
    tracks, 428 observed/scored person frames.
  - Output: `/tmp/wuda_pair_smoke.json` (1,640,421 bytes).

The two skipped repository tests are existing integration tests whose optional
external resources are unavailable; no test failed.

## Pair-preserving visualizations

The optional `--visualization-dir` renders model-free composite overlays after
the single batch inference call. It does not rerun tracking, stitching, or
anomaly inference. Files mirror the source layout as
`<root>/<sample-name>/gt.mp4` and `<root>/<sample-name>/render.mp4`, and their
absolute paths are recorded in the corresponding complete JSON results.

H100 verification used pair `12_(112)_0`. Both outputs contained 137 frames at
25 FPS. The GT visualization preserved 1920x1080 resolution, and the render
visualization preserved 1088x768 resolution. Paired CLI tests passed (`4
passed`) and `git diff --check` passed.

## Human Temporal V1 validation update (2026-08-12)

- Motion acceleration frame ownership was corrected from the last frame to the
  middle frame of each `(first, middle, last)` triplet. This also aligns worst
  motion frames and visualization labels with the acceleration event.
- Added `scripts/summarize_human_temporal_pairs.py` to average valid people per
  video and compare GT/render bone and motion p90 distributions per pair and at
  dataset level.
- Human Temporal tests passed (`7 passed`), the summarizer test passed (`1
  passed`), and `git diff --check` passed.
- The requested real RTMPose run could not be performed without inventing
  resources: `/root/miniconda3/envs/human-reward` has no `mmpose` package, and
  no local RTMPose config/checkpoint was found under the existing HuR, VBench,
  job, Conda, or cache locations. No model was downloaded or installed.
- The originally referenced Wuda input contains 6 pair directories, with one
  known 48-byte corrupt `render.mp4`; the existing completed paired JSON
  contains 5 pairs and, because Human Temporal was disabled, correctly produces
  zero valid temporal samples and null distribution statistics.
