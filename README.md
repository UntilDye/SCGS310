# Dynamic Vehicle Detection and Counting System

本项目旨在实现一个**基于深度学习的动态车辆监测计数系统**，用于对道路车流量进行实时监控与统计。系统能够对视频流中的车辆进行检测、跟踪、车牌识别，并统计单位时间内通过车辆的数量，为智能交通管理和道路拥堵分析提供基础数据支持。

## 功能简介

1. **车辆目标检测**  
   使用YOLO系列模型（如YOLOv5、YOLOv8）对视频流中的车辆进行检测。

2. **车辆跟踪与计数**  
   集成DeepSORT算法对检测到的车辆进行多目标跟踪，实现跨帧车辆身份一致性，并记录通过指定区域（如虚拟线/区域）的车辆数。

3. **车牌识别**  
   对检测到的车辆进行车牌区域定位与字符识别，输出车辆唯一标识。

4. **可视化与统计**  
   在原始视频中实时标记每辆车的位置（以边框形式展示），并输出单位时间内通过车辆的数量。

---

## 技术栈

- **Python 3.x**
- **PyTorch**（深度学习框架）
- **YOLO系列（如YOLOv5/YOLOv8）**（车辆检测）
- **DeepSORT**（多目标追踪）
- **OpenCV**（视频流读取、图像处理与可视化）
- **Numpy**（数据处理）
- **深度学习/神经网络**（目标检测与识别）

---

## 数据集

- **AIC-HCMC-2020**  
  [AIC-HCMC-2020 数据集介绍](https://www.aicitychallenge.org/2020-data-set/)  
  包含城市道路下多类交通目标的视频与标注数据，适合车辆检测与计数任务。

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
环境准备
bash

复制
# 创建虚拟环境并激活
python -m venv vehicle_count_env
source vehicle_count_env/bin/activate

# 安装依赖
pip install torch torchvision numpy opencv-python yolov5 deep_sort_realtime
数据准备
下载并解压 AIC-HCMC-2020 数据集至 ./data/ 目录。
准备测试视频，命名为 test_video.mp4 放于 ./data/。
运行主程序
bash

复制
python main.py --video ./data/test_video.mp4 --output ./results/output.mp4
主要模块说明
yolo_detector.py
封装YOLO检测接口，支持车辆目标检测。
deep_sort_tracker.py
实现DeepSORT多目标跟踪和ID分配。
plate_recognition.py
车辆车牌检测与字符识别模块。
counter.py
车辆过线计数逻辑。
visualization.py
结果可视化，包括车辆框、车牌和计数信息叠加。
输出内容
可视化视频：每帧中展示检测框、跟踪ID、车牌号。
流量统计文件：统计不同时间段的车流量（如 flow_count.csv）。
日志输出：实时在终端输出当前统计数据。

#补充:车牌识别模块可根据model目录下提供的yolo2或yolo3与CRNN来自训练，或者使用utils目录下的检测模块(预训练模型)自己整合到检测系统。另外，运行指令在video_analysis_test.py文件底部.
