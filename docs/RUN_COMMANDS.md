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
  --output outputs/wuda_stage1_pairs_human_reward.json \
  --device cuda:0
```

Use `--max-pairs 1` for a smoke test. Human Temporal remains off unless the
explicit local RTMPose flags documented above are also provided.
