# Person Tracking Commands

本文档单独维护人体检测与 ByteTrack 跟踪流水线的可复制运行命令。

## 指定视频测试

在远程服务器执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate phymotion-track
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/PhyMotion

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

