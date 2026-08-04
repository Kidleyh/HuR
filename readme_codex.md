# Codex 工作记录：人体检测与跟踪前处理

## 目标

在不修改 PhyMotion 现有奖励逻辑的前提下，新增一个独立、可测试、可复用的人体视频前处理模块，完成以下流水线：

```text
输入视频
→ YOLOv8x 逐帧人体检测
→ ByteTrack 跨帧人体跟踪
→ 导出逐帧框、置信度和 track ID
→ 导出轨迹统计与可视化视频
```

本里程碑不会接入 `astrolabe/rewards.py`，也不会实现 ViTPose、GVHMR、SMPL/SMPL-X、主体人物选择、人体完整性评分或物理奖励。

## 代码改动

将新增以下独立模块：

- `astrolabe/scorers/video/person_tracking/schemas.py`：检测、帧、视频及轨迹统计数据结构和合法性校验。
- `astrolabe/scorers/video/person_tracking/tracker.py`：通过 Ultralytics YOLOv8x 和官方 `model.track(...)` 接口执行 person 检测与 ByteTrack。
- `astrolabe/scorers/video/person_tracking/statistics.py`：计算轨迹覆盖率、置信度、框面积比例及最大连续缺失间隔。
- `astrolabe/scorers/video/person_tracking/serialization.py`：写出 JSONL、CSV、summary JSON和逐视频错误报告。
- `configs/bytetrack_person.yaml`：ByteTrack默认参数；检测阈值保持为0.10，以保留第二阶段关联所需的低分检测。
- `scripts/run_person_tracking.py`：单视频、目录和递归目录处理CLI。
- `scripts/check_tracking_env.py`：检查Python、PyTorch、CUDA、GPU、Ultralytics、OpenCV、NumPy、lap及YOLO权重。
- `requirements-tracking.txt`：最小跟踪环境依赖。
- `docs/PERSON_TRACKING.md`：安装、权重查找、CLI、输出schema、CPU模式、错误和限制说明。
- `tests/`中的三个测试文件：schema、统计和可选真实模型集成测试。

## 关键行为

- YOLO模型加载后根据 `model.names` 查找名称为 `person` 的类别，不盲目硬编码类别0。
- 权重按CLI、`YOLO_WEIGHTS`、`GVHMR_ROOT`、项目内checkpoint的顺序查找。
- 默认不联网下载；只有显式传入 `--allow-download` 才允许Ultralytics下载权重。
- 每个视频使用一次完整的流式跟踪调用，并为新视频创建新的模型/预测器，避免跟踪状态跨视频污染。
- `boxes.id is None`时输出空检测帧，不抛异常。
- 所有坐标映射并裁剪到原视频范围，不插值缺失帧。
- CPU模式自动关闭half precision；请求GPU但CUDA不可用时清晰报错。
- 单视频失败写入 `<output_dir>/<video_stem>/error.json`，目录批处理继续执行。
- 已有完整输出默认跳过；`--overwrite`允许重新处理。

## 每个视频的输出

```text
<output_dir>/<video_stem>/
├── detections.jsonl
├── detections.csv
├── tracks_summary.json
└── tracked.mp4
```

`detections.jsonl`每帧一行，包括没有激活轨迹的帧。`detections.csv`每个检测框一行。`tracks_summary.json`记录视频、检测器、跟踪器、运行信息和逐轨迹统计。`tracked.mp4`只用于人工检查，不作为后续算法输入。

## 环境

已创建Conda环境 `phymotion-track`：

- Python 3.10.20
- PyTorch 2.6.0+cu124
- torchvision 0.21.0+cu124
- Ultralytics 8.4.102
- OpenCV 4.11.0
- NumPy 1.26.4

当前登录节点没有可见CUDA设备，但CPU模式可运行。完整安装命令见 `docs/PERSON_TRACKING.md`。

## 验证状态

已执行：

```bash
python scripts/check_tracking_env.py
pytest -q tests/test_person_tracking_schemas.py tests/test_track_statistics.py
pytest -q -m integration tests/test_person_tracking_integration.py
```

结果：

- 单元测试：15 passed。
- 集成测试：1 skipped，原因是服务器未找到YOLOv8x权重。
- Python编译检查：通过。
- CLI帮助和缺失输入错误路径：通过。
- `astrolabe/rewards.py`：无修改。

真实smoke test尚未执行，因为当前未找到以下文件：

- `checkpoints/yolo/yolov8x.pt`或其他约定位置的YOLOv8x权重；
- `$GVHMR_ROOT/docs/example_video/tennis.mp4`、`docs/example_video/tennis.mp4`或`tests/assets/person_short.mp4`。

补齐权重和视频后，应执行：

```bash
conda activate phymotion-track
python scripts/run_person_tracking.py \
  --input "$GVHMR_ROOT/docs/example_video/tennis.mp4" \
  --output-dir outputs/person_tracking_smoke \
  --weights "$GVHMR_ROOT/inputs/checkpoints/yolo/yolov8x.pt" \
  --tracker-config configs/bytetrack_person.yaml \
  --device 0 --conf 0.10 --iou 0.70 --imgsz 640 \
  --half --save-visualization --overwrite
```

## 后续工作

后续里程碑可在该前端输出之上继续实现主体轨迹选择、人体完整性或有效性评分、ViTPose、GVHMR、SMPL/SMPL-X恢复和物理奖励接入。本阶段不会自动删除任何轨迹，也不会判断主要人物。多人严重交叉场景可能需要BoT-SORT或ReID。
