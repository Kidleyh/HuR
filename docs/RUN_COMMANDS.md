# PhyMotion Preprocessing Run Commands

本文档是人体预处理项目唯一的运行命令清单，统一维护检测跟踪、离线
tracklet stitching、测试和结果检查命令。每次功能修改后，应同步更新本文件。

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

先运行本文档中的 person tracking 命令，再运行：

```bash
python scripts/run_tracklet_stitching.py \
  --input outputs/person_tracking_omnistream \
  --output-dir outputs/tracklet_stitching_omnistream \
  --recursive \
  --config configs/tracklet_stitching.yaml \
  --save-visualization \
  --overwrite
```
