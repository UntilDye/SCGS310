# Dynamic Vehicle Detection and Counting System

本项目实现了一个**基于深度学习的动态车辆监测计数系统**，用于对城市道路车流量进行实时监控与统计。系统支持车辆检测、跟踪、车牌识别，以及单位时间内车流量自动统计，为智能交通管理和路况分析提供数据支撑。

---

## 功能简介

1. **车辆目标检测**  
   利用YOLO系列模型（如YOLOv5、YOLOv8）对视频流中的车辆进行高效检测。

2. **车辆跟踪与计数**  
   集成DeepSORT算法，实现车辆跨帧跟踪与ID分配，并在指定区域（如虚拟计数线/区域）自动计数。

3. **车牌识别**  
   支持基于YOLO2/YOLO3与CRNN模型进行车牌检测与字符识别，或直接使用`utils`目录下的预训练检测模块，根据需求进行灵活整合。

4. **可视化与统计**  
   在视频中实时标记每辆车的位置、跟踪ID和车牌号，并输出单位时间内的流量统计。

---

## 技术栈

- Python 3.x
- PyTorch（深度学习框架）
- YOLO系列（车辆检测）
- DeepSORT（多目标跟踪）
- OpenCV（视频处理与可视化）
- Numpy（数据处理）
- 深度学习/神经网络相关技术

---

## 数据集

- **AIC-HCMC-2020**  
  [AIC-HCMC-2020 数据集介绍](https://www.aicitychallenge.org/2020-data-set/)  
  包含城市道路多类型车辆的视频及标注，适合本项目目标检测与计数任务。

---

## 系统架构

```mermaid
graph TD
A[视频输入] --> B[YOLO车辆检测]
B --> C[DeepSORT车辆跟踪]
C --> D[车牌检测与识别]
C --> E[车辆计数]
D --> F[识别结果可视化]
E --> F
F --> G[输出统计&处理后视频]
快速开始
1. 环境准备
bash

复制
# 创建虚拟环境
python -m venv vehicle_count_env
source vehicle_count_env/bin/activate

# 安装依赖
pip install torch torchvision numpy opencv-python yolov5 deep_sort_realtime
2. 数据准备
下载并解压 AIC-HCMC-2020 数据集至 ./data/ 目录。
准备测试视频，命名为 test_video.mp4，放于 ./data/。
3. 车牌识别模块说明
车牌识别模块可根据model目录下提供的YOLO2或YOLO3与CRNN模型自行训练，
或直接使用utils目录下的检测模块（预训练模型）集成到检测系统中，按需选择灵活方便。
4. 运行主程序
推荐直接运行 video_analysis_test.py。主要运行指令在该文件底部，请根据实际检测需求调整参数。
bash

复制
python video_analysis_test.py --video ./data/test_video.mp4 --output ./results/output.mp4
主要模块说明
yolo_detector.py
YOLO检测接口，支持车辆目标检测。

deep_sort_tracker.py
DeepSORT算法实现，多目标跟踪与ID分配。

plate_recognition.py
车牌定位与字符识别模块。支持自训练模型或集成预训练模块。

counter.py
车辆过线（区域）计数逻辑。

visualization.py
检测/跟踪/车牌/统计信息的可视化叠加。

输出内容
可视化视频：每帧展示检测框、跟踪ID、车牌号等信息。
流量统计文件：如 flow_count.csv，记录各时间段车流量。
日志输出：终端实时输出统计信息。
参考
YOLO官方文档
DeepSORT论文及实现
AIC-HCMC-2020数据集
联系方式
如有疑问或合作，欢迎联系

Email: dontlike299@gmail.com
仅供学术交流与测试，禁止用于非法用途。
