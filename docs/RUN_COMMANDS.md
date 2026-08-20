# PhyMotion Preprocessing Run Commands

本文档是人体预处理项目唯一的运行命令清单，统一维护检测跟踪、离线
tracklet stitching、测试和结果检查命令。每次功能修改后，应同步更新本文件。

## 远程 GPU 节点

HuR 工作区后续统一通过本机 SSH 配置中的 GPU 节点连接：

```bash
ssh kaifa-test
```

该节点使用共享工作区：

```text
/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion
```

## 公共环境准备

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate phymotion-track
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion
```

## 指定视频测试

在远程服务器执行：

```bash
python scripts/check_tracking_env.py --require-weights --require-cuda

python scripts/run_person_tracking.py \
  --input /gemini/platform/public/aigc/human_guozz2/code/lyh/job/OmniStream-LTX-dynamic/ltx_experiments/test_outputs/720_1080_249/onestage_motion_compare_step8_cfg1/base_step27000/ltx23_onestage_i2av_motion_001.mp4 \
  --output-dir outputs/person_tracking_omnistream \
  --weights checkpoints/yolo/yolov8x.pt \
  --tracker-config configs/bytetrack_person.yaml \
  --device 0 \
  --conf 0.10 \
  --iou 0.70 \
  --imgsz 640 \
  --half \
  --save-visualization \
  --overwrite
```

结果目录为：

```text
/gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion/outputs/person_tracking_omnistream/ltx23_onestage_i2av_motion_001/
```

其中包括：

- `detections.jsonl`
- `raw_detections.csv`
- `tracked_detections.csv`
- `detections.csv`（兼容文件，内容等同于 tracked detections）
- `tracks_summary.json`
- `tracked.mp4`

## 检查输出

```bash
find outputs/person_tracking_omnistream/ltx23_onestage_i2av_motion_001 \
  -maxdepth 1 -type f -ls

python - <<'PY'
import json
from pathlib import Path

path = Path(
    "outputs/person_tracking_omnistream/"
    "ltx23_onestage_i2av_motion_001/tracks_summary.json"
)
data = json.loads(path.read_text(encoding="utf-8"))
print("video:", data["video"])
print("detection_summary:", data["detection_summary"])
print("track_count:", len(data["tracks"]))
PY
```

## CPU 备用命令

如果服务器当前没有可用 CUDA 设备，跳过 `--require-cuda` 检查，并将运行参数改为：

```bash
python scripts/run_person_tracking.py \
  --input /gemini/platform/public/aigc/human_guozz2/code/lyh/job/OmniStream-LTX-dynamic/ltx_experiments/test_outputs/720_1080_249/onestage_motion_compare_step8_cfg1/base_step27000/ltx23_onestage_i2av_motion_001.mp4 \
  --output-dir outputs/person_tracking_omnistream \
  --weights checkpoints/yolo/yolov8x.pt \
  --tracker-config configs/bytetrack_person.yaml \
  --device cpu \
  --conf 0.10 \
  --iou 0.70 \
  --imgsz 640 \
  --no-half \
  --save-visualization \
  --overwrite
```

CPU 模式速度会明显慢于 GPU 模式。

## Stage 1.2a：离线 Tracklet Stitching

### 编译和非集成测试

```bash
python -m compileall \
  astrolabe/scorers/video/tracklet_stitching \
  scripts/run_tracklet_stitching.py

pytest -q -m "not integration"
```

Stage 1.2a.1 定向测试：

```bash
pytest -q \
  tests/test_tracklet_stitching_candidates.py \
  tests/test_tracklet_stitching_io.py \
  tests/test_tracklet_stitching_matching.py \
  tests/test_tracklet_stitching_features.py
```

### 单个跟踪结果目录

```bash
python scripts/run_tracklet_stitching.py \
  --input outputs/person_tracking_smoke/2 \
  --output-dir outputs/tracklet_stitching \
  --config configs/tracklet_stitching.yaml \
  --save-visualization \
  --overwrite
```

输出目录为：

```text
outputs/tracklet_stitching/2/
```

### 递归批量处理

```bash
python scripts/run_tracklet_stitching.py \
  --input outputs/person_tracking \
  --output-dir outputs/tracklet_stitching \
  --recursive \
  --config configs/tracklet_stitching.yaml \
  --save-visualization \
  --overwrite
```

### Stage 1.2a Smoke test

```bash
python scripts/run_tracklet_stitching.py \
  --input outputs/person_tracking_smoke \
  --output-dir outputs/tracklet_stitching_smoke \
  --recursive \
  --config configs/tracklet_stitching.yaml \
  --no-raw-bridge-allow-associated-raw \
  --save-visualization \
  --overwrite
```

### 检查拼接输出

```bash
find outputs/tracklet_stitching_smoke -maxdepth 5 -type f -ls

python - <<'PY'
import json
from pathlib import Path

paths = list(
    Path("outputs/tracklet_stitching_smoke").rglob("tracklet_stitching.json")
)
assert paths, "tracklet_stitching.json not found"

for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    print(path)
    print("source tracks:", len(data["track_id_to_logical_track_id"]))
    print("logical tracks:", len(data["logical_tracks"]))
    print("merged edges:", len(data["merged_edges"]))
    print("uncertain edges:", len(data["uncertain_edges"]))
    print("rejected edges:", len(data["rejected_edges"]))
    print("visualization:", data.get("visualization"))
    excluded = sum(
        edge.get("raw_bridge_excluded_associated_count", 0)
        for group in ("merged_edges", "uncertain_edges", "rejected_edges")
        for edge in data.get(group, [])
    )
    print("excluded associated raw:", excluded)
    print("mapping:", data["track_id_to_logical_track_id"])
PY
```

为消融实验显式恢复旧的 raw bridge 行为：

```bash
python scripts/run_tracklet_stitching.py \
  --input outputs/person_tracking_smoke \
  --output-dir outputs/tracklet_stitching_allow_associated \
  --recursive \
  --config configs/tracklet_stitching.yaml \
  --raw-bridge-allow-associated-raw \
  --save-visualization \
  --overwrite
```

### Tracklet stitching 集成测试

```bash
pytest -q -m integration tests/test_tracklet_stitching_integration.py
```

如果需要指定另一个已经完成 person tracking 的结果目录：

```bash
TRACKLET_STITCHING_TEST_INPUT=/absolute/path/to/tracking_result \
pytest -q -m integration tests/test_tracklet_stitching_integration.py
```

## 从原始视频连续运行 Stage 1.1 和 Stage 1.2a

推荐使用统一的端到端入口。它只接收最初的视频路径，并按当前实现顺序运行
person tracking 和 tracklet stitching：

```bash
VIDEO=/absolute/path/to/video.mp4

python scripts/run_person_preprocessing_pipeline.py \
  --input "$VIDEO" \
  --name sample \
  --output-root outputs \
  --weights checkpoints/yolo/yolov8x.pt \
  --tracker-config configs/bytetrack_person.yaml \
  --stitching-config configs/tracklet_stitching.yaml \
  --device 0 \
  --conf 0.10 \
  --iou 0.70 \
  --imgsz 640 \
  --half \
  --save-visualization \
  --overwrite
```

`--name sample` 的输出固定为：

```text
outputs/sample_person_tracking/
outputs/sample_tracklet_stitching/
```

当前服务器没有可用 CUDA 时使用：

```bash
python scripts/run_person_preprocessing_pipeline.py \
  --input "$VIDEO" \
  --name sample \
  --output-root outputs \
  --weights checkpoints/yolo/yolov8x.pt \
  --device cpu \
  --no-half \
  --save-visualization \
  --overwrite
```

针对之前指定的 OmniStream 视频：

```bash
python scripts/run_person_preprocessing_pipeline.py \
  --input /gemini/platform/public/aigc/human_guozz2/code/lyh/job/OmniStream-LTX-dynamic/ltx_experiments/test_outputs/720_1080_249/onestage_motion_compare_step8_cfg1/base_step27000/ltx23_onestage_i2av_motion_001.mp4 \
  --name ltx23_onestage_i2av_motion_001 \
  --output-root outputs \
  --weights checkpoints/yolo/yolov8x.pt \
  --device cpu \
  --no-half \
  --save-visualization \
  --overwrite
```

端到端入口测试：

```bash
python -m compileall scripts/run_person_preprocessing_pipeline.py
pytest -q tests/test_person_preprocessing_pipeline.py
```

## Stage 1.3：逐人物 VBench Human Anomaly

当前节点必须能看到 CUDA；模型环境保持独立：

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
  --crop-batch-size 128 \
  --visualize \
  --overwrite
```

包含 Stage 1.3 的统一端到端命令：

```bash
python scripts/run_person_preprocessing_pipeline.py \
  --input /absolute/path/to/video.mp4 \
  --name sample \
  --output-root outputs \
  --weights checkpoints/yolo/yolov8x.pt \
  --device 0 \
  --half \
  --save-visualization \
  --human-anomaly \
  --vbench-root /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0 \
  --vbench-cache-dir /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/vbench2 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32 \
  --vbench-device cuda:0 \
  --visualize-human-anomaly \
  --overwrite
```

Stage 1.3 无模型单元测试：

```bash
python -m compileall \
  astrolabe/scorers/video/human_anomaly \
  scripts/run_person_human_anomaly.py \
  scripts/vbench_human_anomaly_worker.py

pytest -q \
  tests/test_human_anomaly_manifest.py \
  tests/test_human_anomaly_aggregation.py \
  tests/test_human_anomaly_subprocess.py
```

Stage 1.3 鲁棒性加固和全仓验收：

```bash
pytest -q tests/test_human_anomaly*.py
pytest -q tests/test_person_preprocessing_pipeline.py
pytest -q
git diff --check
```

## 单进程内存式 Human Reward

固定使用合并环境，默认只在标准输出返回最终JSON：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate human-reward
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion

python scripts/run_human_reward.py \
  --video /absolute/path/to/input.mp4 \
  --device cuda:0 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32
```

仅原子保存一个最终JSON：

```bash
python scripts/run_human_reward.py \
  --video /absolute/path/to/input.mp4 \
  --output outputs/reward.json \
  --device cuda:0 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32
```

单视频在全部模型释放后可选生成综合H.264/yuv420p可视化：

```bash
python scripts/run_human_reward.py \
  --video /absolute/path/to/input.mp4 \
  --output outputs/reward.json \
  --visualization-output outputs/reward_visualization.mp4 \
  --device cuda:0 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32
```

多个视频共享一次YOLO加载和一次VBench模型加载，输出一个顺序与输入一致的JSON数组：

```bash
python scripts/run_human_reward.py \
  --video /absolute/path/to/a.mp4 \
  --video /absolute/path/to/b.mp4 \
  --output outputs/rewards.json \
  --device cuda:0 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32
```

未指定可视化时只执行两遍视频读取；指定可视化时在模型全部释放后第三次读取原视频。流程不写tracking、stitching、worker日志或其他中间文件。`--vbench-clip-model`应指向已部署的本地模型目录，避免运行时联网下载。

测试：

```bash
pytest -q tests/test_human_reward*.py
pytest -q tests/test_human_anomaly*.py
git diff --check
```

### Person-centric Human Reward result

`HumanRewardModel.score()` and `score_batch()` keep person-frame data once under
`result["persons"]`, grouped by `logical_track_id`. Each person contains the
original stitching `track` summary, observed (non-interpolated) `frames`, the
current binary `score`, and an empty `temporal` extension point. Video-level
values live in `result["video_score"]`; top-level `reward`, `micro_score`, and
`macro_score` remain compatibility aliases sourced from that same aggregation.

The optional visualization reads the person-centric structure through a
lightweight `(logical_track_id, person_frame_index)` frame index. It does not run
tracking or Human Anomaly again.

## Human Temporal Consistency (RTMPose)

Human Temporal is optional and requires an already-installed MMPose plus local
RTMPose config/checkpoint files. It never invokes another person detector and
never downloads model files:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate human-reward
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion

python scripts/run_human_reward.py \
  --video /absolute/path/to/input.mp4 \
  --output outputs/reward_with_human_temporal.json \
  --visualization-output outputs/reward_with_human_temporal.mp4 \
  --device cuda:0 \
  --human-temporal \
  --human-temporal-pose-config /absolute/path/to/rtmpose_config.py \
  --human-temporal-pose-checkpoint /absolute/path/to/rtmpose_checkpoint.pth \
  --human-temporal-keypoint-threshold 0.3 \
  --human-temporal-max-frame-gap 2
```

The result is attached at `person["temporal"]["human"]`; its `score` is currently
`null` and therefore does not change binary person scores or video reward.

### Head/Face and Hand Temporal

Head Temporal reuses the face boxes already emitted by Human Anomaly. Hand
Temporal reuses its hand boxes and associates them only when confident Human
RTMPose left/right wrists make the assignment reliable. For that reason the
hand option must be used together with `--human-temporal`. All three RTMPose
models are loaded sequentially, use local files only, and are released before
visualization:

```bash
python scripts/run_human_reward.py \
  --video /absolute/path/to/input.mp4 \
  --output outputs/reward_all_temporal.json \
  --visualization-output outputs/reward_all_temporal.mp4 \
  --device cuda:0 \
  --human-temporal \
  --human-temporal-pose-config /absolute/path/to/body_config.py \
  --human-temporal-pose-checkpoint /absolute/path/to/body_checkpoint.pth \
  --head-temporal \
  --head-temporal-pose-config /absolute/path/to/face_config.py \
  --head-temporal-pose-checkpoint /absolute/path/to/face_checkpoint.pth \
  --hand-temporal \
  --hand-temporal-pose-config /absolute/path/to/hand_config.py \
  --hand-temporal-pose-checkpoint /absolute/path/to/hand_checkpoint.pth
```

The added results are `person["temporal"]["head"]` and
`person["temporal"]["hand"]`. Their raw metrics and worst-frame lists are for
analysis only; `score` remains `null`, and they do not affect current reward.
Missing paths are errors and never trigger a network download.

Real paired Body/Head/Hand Temporal validation with the deployed local models:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
python scripts/run_human_reward_pairs.py \
  --input-dir /gemini/platform/public/aigc/human_guozz2/code/lyh/job/OmniStream-LTX-dynamic/ltx_experiments/test_outputs/wuda_stage1_pairs_twostage30_auto_frames \
  --output outputs/head_hand_temporal_pairs \
  --visualization-dir outputs/head_hand_temporal_pairs/visualizations \
  --device cuda:0 \
  --human-temporal \
  --human-temporal-pose-config weights/rtmpose/rtmpose-m_8xb256-420e_body8-256x192.py \
  --human-temporal-pose-checkpoint weights/rtmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
  --head-temporal \
  --head-temporal-pose-config weights/rtmpose_face/rtmpose-m_8xb256-120e_face6-256x256.py \
  --head-temporal-pose-checkpoint weights/rtmpose_face/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth \
  --hand-temporal \
  --hand-temporal-pose-config weights/rtmpose_hand/rtmpose-m_8xb256-210e_hand5-256x256.py \
  --hand-temporal-pose-checkpoint weights/rtmpose_hand/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth

python scripts/summarize_human_temporal_pairs.py \
  --input outputs/head_hand_temporal_pairs/human_reward_pairs_full.json \
  --output outputs/head_hand_temporal_pairs/temporal_summary.json
```

Add `--max-pairs 1` to the first command for the one-pair smoke test. The
offline environment variables make accidental model downloads fail explicitly.

Tests that do not require MMPose or a GPU:

```bash
pytest -q tests/test_human_temporal*.py
pytest -q tests/test_human_reward*.py
git diff --check
```

## Batch score paired gt/render directories

Each immediate child directory must contain `gt.mp4` (positive) and
`render.mp4` (negative). Other files such as `sample.json` are ignored. All
complete person-centric results are written atomically to one JSON file:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate human-reward
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion

python scripts/run_human_reward_pairs.py \
  --input-dir /gemini/platform/public/aigc/human_guozz2/code/lyh/job/OmniStream-LTX-dynamic/ltx_experiments/test_outputs/wuda_stage1_pairs_twostage30_auto_frames \
  --output outputs/wuda_stage1_pairs_human_reward \
  --device cuda:0
```

`--output` is a directory. It contains:

```text
human_reward_pairs_full.json
human_reward_pairs_scores.json
```

The full file preserves the complete person-centric and person-frame results.
The score-only file removes person-frame detections, bboxes, keypoints, and
per-frame temporal metrics while retaining video scores/counts, person scores,
track summaries, anomaly/coverage statistics, and aggregated Human Temporal
metrics.

To also generate composite Human Reward visualizations while preserving the
input pair directory layout:

```bash
python scripts/run_human_reward_pairs.py \
  --input-dir /gemini/platform/public/aigc/human_guozz2/code/lyh/job/OmniStream-LTX-dynamic/ltx_experiments/test_outputs/wuda_stage1_pairs_twostage30_auto_frames \
  --output outputs/wuda_stage1_pairs_human_reward \
  --visualization-dir outputs/wuda_stage1_pairs_human_reward_visualizations \
  --device cuda:0
```

This creates `<visualization-dir>/<sample-name>/gt.mp4` and
`<visualization-dir>/<sample-name>/render.mp4`. The aggregate JSON also records
the absolute visualization path in each positive/negative result.

Use `--max-pairs 1` for a smoke test. Human Temporal remains off unless the
explicit local RTMPose flags documented above are also provided.

Summarize GT/render Human Temporal distributions from a completed paired JSON:

```bash
python scripts/summarize_human_temporal_pairs.py \
  --input outputs/paired_human_temporal.json \
  --output outputs/paired_human_temporal_summary.json
```

The summary averages valid people within each video, then reports dataset-level
GT/render means and medians, the fraction of pairs where Render exceeds GT, and
the largest Render-minus-GT pair differences for bone and motion p90 metrics.

## Selection manifest：逐帧 Human Anomaly GT/render 判别评测

以下命令严格使用 `selection_manifest.json` 中有序的 `selected_pairs`，不会扫描
`data_root` 中未被选中的其他视频。Human、Head、Hand 和 3D Temporal 默认均不
开启，因此只评测现有逐帧 Human/Face/Hand anomaly 检测：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate human-reward
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_human_reward_pairs.py \
  --selection-manifest /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion_origin/outputs/minimax_h3_turbo_wuda_v4_ema_8step_736x416_pair_ranking_100_seed20260817/selection_manifest.json \
  --output outputs/minimax_h3_pair_ranking_100_frame_anomaly \
  --device cuda:0 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32
```

先运行一个 pair 的 smoke test：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_human_reward_pairs.py \
  --selection-manifest /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion_origin/outputs/minimax_h3_turbo_wuda_v4_ema_8step_736x416_pair_ranking_100_seed20260817/selection_manifest.json \
  --output outputs/minimax_h3_pair_ranking_frame_anomaly_smoke \
  --max-pairs 1 \
  --device cuda:0
```

输出目录包含：

```text
human_reward_pairs_full.json
human_reward_pairs_scores.json
human_reward_pair_frame_evaluation.json
human_reward_pair_evaluation.json
```

`human_reward_pair_frame_evaluation.json` 保留每个检测人物帧的 Human、Face、
Hand abnormal probability、combined probability 和官方二值 `person_abnormal`。
`human_reward_pair_evaluation.json` 是去掉逐帧明细后的精简统计，分别报告每个
指标的 GT win、render win、tie、不可比较数量、严格准确率和 tie-aware 准确率。
所有 `quality_score` 都是越高越好；Face/Hand 未检测到时保持中性，不会被填成
正常或异常。GT/render 视频帧数可以不同，因此比较的是各视频人物帧观测的聚合
分布，不会假设两个视频的 frame index 具有严格一一语义对应关系。

### 另一节点：每完成一对立即保存的增量运行

增量入口每完成一个完整的 GT/render pair 就原子更新结果和截至当前的准确率。
它使用独立文件名，不会覆盖上面的全量 runner 输出；指定的输出目录如果已有
增量文件，默认也会拒绝运行，必须显式使用 `--resume` 才能续跑。

在另一 GPU 节点首次运行时，请使用一个新的输出目录：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate human-reward
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
python scripts/run_human_reward_pairs_incremental.py \
  --selection-manifest /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion_origin/outputs/minimax_h3_turbo_wuda_v4_ema_8step_736x416_pair_ranking_100_seed20260817/selection_manifest.json \
  --output outputs/minimax_h3_pair_ranking_100_frame_anomaly_incremental_node2 \
  --device cuda:0 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32
```

任务中断后使用完全相同的输出目录续跑：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
python scripts/run_human_reward_pairs_incremental.py \
  --selection-manifest /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion_origin/outputs/minimax_h3_turbo_wuda_v4_ema_8step_736x416_pair_ranking_100_seed20260817/selection_manifest.json \
  --output outputs/minimax_h3_pair_ranking_100_frame_anomaly_incremental_node2 \
  --device cuda:0 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32 \
  --resume
```

随时查看已完成数量和当前各项准确率：

```bash
python -m json.tool \
  outputs/minimax_h3_pair_ranking_100_frame_anomaly_incremental_node2/human_reward_pairs_incremental_progress.json
```

增量目录包含：

```text
human_reward_pairs_incremental_full.json
human_reward_pairs_incremental_scores.json
human_reward_pair_incremental_frame_evaluation.json
human_reward_pair_incremental_evaluation.json
human_reward_pairs_incremental_progress.json
```

注意：为保证每个 pair 结束即可得到 Human/Face/Hand 完整结果，当前增量入口按
pair 调用现有低显存 pipeline，因此每对都会重新加载、释放 YOLO 和 VBench 模型。
它比单次加载处理全部 100 对更慢，但单对失败或节点中断不会丢失之前的结果。
# Human Temporal V2 (GVHMR, local resources only)

Human Temporal V2 is optional and does not run unless `--human-temporal-3d`
is supplied. The checkpoint and all official GVHMR preprocessing assets must
already exist locally; HuR never downloads them.

Required official GVHMR assets include at least:

```text
$GVHMR_ROOT/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
$GVHMR_ROOT/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
$GVHMR_ROOT/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
```

The GVHMR body-model assets referenced by `hmr4d/utils/body_model` must also
be installed according to the upstream GVHMR instructions.

Official one-video check before HuR integration:

```bash
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/GVHMR
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python tools/demo/demo.py \
  --video "/path/to/paired_sample/gt.mp4" \
  --output_root /tmp/gvhmr_hur_v2_smoke \
  --static_cam
```

Single-video HuR run:

```bash
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion
conda run -n human-reward python scripts/run_human_reward.py \
  --video /path/to/input.mp4 \
  --output outputs/sample_human_reward.json \
  --device cuda:0 \
  --human-temporal-3d \
  --gvhmr-root /gemini/platform/public/aigc/human_guozz2/code/lyh/job/GVHMR \
  --gvhmr-checkpoint /path/to/gvhmr_siga24_release.ckpt
```

Paired run and GT/render summary:

```bash
conda run -n human-reward python scripts/run_human_reward_pairs.py \
  --input-dir /path/to/pairs \
  --output outputs/human_temporal_3d_pairs \
  --max-pairs 1 \
  --device cuda:0 \
  --human-temporal-3d \
  --gvhmr-root /gemini/platform/public/aigc/human_guozz2/code/lyh/job/GVHMR \
  --gvhmr-checkpoint /path/to/gvhmr_siga24_release.ckpt

python scripts/summarize_human_temporal_pairs.py \
  --input outputs/human_temporal_3d_pairs/human_reward_pairs_full.json \
  --output outputs/human_temporal_3d_pairs/temporal_summary.json
```

## 小且清晰人脸筛选

该入口只加载 VBench 已有的 YOLO-World FaceHandDetector，直接对整图/视频帧
以及原分辨率 tiles 检测 face，不加载人物检测、跟踪、ViT、RTMPose 或 Temporal：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate human-reward
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion

NO_ALBUMENTATIONS_UPDATE=1 \
python scripts/filter_small_clear_faces.py \
  --input /absolute/path/to/images_or_videos \
  --output outputs/small_clear_faces.json \
  --recursive \
  --frame-stride 5 \
  --device cuda:0 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32 \
  --tile-size 1024 \
  --tile-overlap 0.15 \
  --whole-image-detection \
  --area-threshold 0.01 \
  --min-face-short-side 32 \
  --laplacian-threshold 100 \
  --tenengrad-threshold 1000 \
  --min-qualified-frames 1 \
  --visualization-dir outputs/small_clear_faces_visualizations \
  --copy-selected-to outputs/small_clear_faces_selected
```

阈值是尚未校准的 V1 起点。输出会保留每个检测脸的面积占比、像素尺寸、
Laplacian variance 和 Tenengrad，后续应根据真实数据分布重新选择阈值。
脚本强制 YOLO-World 的 CLIP text model 使用本地目录，并启用 Hugging Face
offline 模式，不会在运行时下载模型。
指定 `--visualization-dir` 后，图片会输出带框标注图，视频会以原分辨率和
原 FPS 输出 H.264/yuv420p MP4。绿色表示满足筛选条件，黄色表示检测到但未满足；
标签包含 detector score、面积占比、短边、Laplacian variance 与 Tenengrad。
输出保留输入相对目录结构，且完全复用检测阶段结果，不会再次运行模型。

### Guangdian V2 数据首帧筛选

只根据 `caption_v2/*.json` 匹配同 stem 的 `video/data/*.mp4` 和
`label/*.json`，每个视频仅解码、检测首帧。满足默认小且清晰人脸条件的样本
按训练清单格式写入 `--output-path`；缺目录、缺配对文件、不满足条件和处理失败
写入独立的 `*_skipped.json`：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate human-reward
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion

NO_ALBUMENTATIONS_UPDATE=1 \
python scripts/filter_guangdian_small_clear_faces.py \
  --input-root /gemini-1/platform/public/human_guozz/hz_data01_new/guangdian_20251114 \
  --output-path outputs/guangdian_20251114_small_clear_faces.json \
  --device cuda:0 \
  --vbench-clip-model /gemini/platform/public/aigc/human_guozz2/code/lyh/job/VBench/VBench-2.0/.cache/huggingface/openai/clip-vit-base-patch32 \
  --tile-size 1024 \
  --tile-overlap 0.15 \
  --whole-image-detection
```

处理过程每完成一个视频都会原子刷新两个 JSON。中断后用完全相同的命令追加
`--resume` 即可跳过已经处理过的 caption。用 `--skipped-output-path` 可显式指定
跳过记录路径，用 `--max-videos 1` 可先做单视频 smoke test。
