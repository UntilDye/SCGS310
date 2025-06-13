#video_vehicle_analysis.py
import cv2
import torch
import numpy as np
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from models.yolo_vehicle import VehicleYOLO
from utils.deepsort_tracker import VehicleTracker
from utils.traffic_counter import TrafficCounter

class VideoVehicleAnalyzer:
    """视频车辆分析器"""
    
    def __init__(self, config_path: str, model_path: str, debug: bool = False):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.class_names = self.config['model']['class_names']
        self.model = self._load_model(model_path)
        self.debug = debug
        
        #检测参数 - 降低阈值减少漏检
        detection_cfg = self.config.get('detection', {})
        self.conf_threshold = float(detection_cfg.get('conf_threshold', 0.15))  
        self.iou_threshold = float(detection_cfg.get('iou_threshold', 0.45))    
        self.nms_iou_threshold = float(detection_cfg.get('nms_iou_threshold', 0.5))  # NMS阈值

        # 大幅放宽过滤参数
        filtering_cfg = self.config.get('filtering', {})
        self.min_box_area = filtering_cfg.get('min_box_area', 100)      
        self.max_box_area = filtering_cfg.get('max_box_area', 500000)   # 上限
        self.min_aspect_ratio = filtering_cfg.get('min_aspect_ratio', 0.05)  # 宽松
        self.max_aspect_ratio = filtering_cfg.get('max_aspect_ratio', 20.0)  # 宽松
        
        # 追踪器配置
        tracker_config = self.config.get('tracking', {})
        self.tracker = VehicleTracker(
            class_names=self.class_names,
            max_age=30,        
            n_init=1,          # 降到1，快速初始化追踪
            max_iou_distance=0.7,  # 提高IoU阈值
            debug=debug
        )
        
        # 初始化流量计数器
        counting_config = self.config.get('counting', {})
        counting_lines_config = counting_config.get('counting_line', [[[300, 400], [900, 400]]])
        count_direction = counting_config.get('count_direction', 'both')
        
        formatted_lines = self._parse_counting_lines(counting_lines_config)
        self.traffic_counter = TrafficCounter(formatted_lines, count_direction)
        
        # 可视化颜色
        self.colors = [
            (255, 56, 56),   # 红色 (摩托车)
            (50, 205, 50),   # 绿色 (汽车)
            (70, 130, 180),  # 蓝色 (巴士)
            (255, 165, 0),   # 橙色 (卡车)
        ]

    def _parse_counting_lines(self, counting_lines_config):
        """解析计数线配置"""
        formatted_lines = []
        if isinstance(counting_lines_config, list) and len(counting_lines_config) > 0:
            for line_coords in counting_lines_config:
                try:
                    if (isinstance(line_coords, list) and len(line_coords) == 2 and
                        isinstance(line_coords[0], list) and len(line_coords[0]) == 2 and
                        isinstance(line_coords[1], list) and len(line_coords[1]) == 2):
                        
                        p1 = (int(line_coords[0][0]), int(line_coords[0][1]))
                        p2 = (int(line_coords[1][0]), int(line_coords[1][1]))
                        formatted_lines.append([p1, p2])
                    else:
                        print(f"警告: 计数线格式不正确: {line_coords}. 使用默认计数线.")
                        formatted_lines = [[(300, 400), (900, 400)]]
                        break
                except (IndexError, TypeError, ValueError) as e:
                    print(f"解析计数线配置时出错: {e}. 使用默认计数线.")
                    formatted_lines = [[(300, 400), (900, 400)]]
                    break
        else:
            formatted_lines = [[(300, 400), (900, 400)]]
            
        return formatted_lines

    def _load_model(self, model_path: str):
        """加载车辆检测模型"""
        print(f"加载模型: {model_path}")
        
        input_size_cfg = self.config['model']['input_size']
        try:
            input_shape = (int(input_size_cfg[0]), int(input_size_cfg[1]))
        except (TypeError, IndexError, ValueError) as e:
            print(f"从配置解析 input_size 时出错: {input_size_cfg}. 使用默认 (640, 640)  ")
            input_shape = (640, 640)  # 默认值

        model = VehicleYOLO(
            num_classes=int(self.config['model']['num_classes']),
            input_shape=input_shape
        )
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"加载模型权重时出错: {e}")
            raise
            
        model.to(self.device)
        model.eval()
        print("模型加载完成")
        return model
    
    def detect_vehicles(self, frame: np.ndarray) -> List[List]:
        """车辆检测方法"""
        img_orig_h, img_orig_w = frame.shape[:2]
        
        # 预处理
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_size_cfg = self.config['model']['input_size']
        input_w, input_h = int(input_size_cfg[0]), int(input_size_cfg[1])
        img_resized = cv2.resize(img_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        
        # 归一化
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        img_tensor = (img_tensor.to(self.device) - mean) / std
        img_tensor = img_tensor.unsqueeze(0)
        
        # 模型推理
        with torch.no_grad():
            try:
                if hasattr(self.model, 'predict') and callable(self.model.predict):
                    raw_preds = self.model.predict(
                        img_tensor,
                        conf_threshold=self.conf_threshold,  
                        iou_threshold=self.iou_threshold,
                        device=self.device
                    )
                else:
                    raw_preds = self.model(img_tensor)
            except Exception as e:
                print(f"模型推理失败: {e}")
                return []
        
        # 后处理和过滤
        results = self._post_process_and_filter(
            raw_preds, 
            (img_orig_w, img_orig_h), 
            (input_w, input_h)
        )
        
        return results

    def _post_process_and_filter(self, detections, original_size, input_size):
        """后处理和过滤方法"""
        if not detections:
            return []

        original_width, original_height = original_size
        input_width, input_height = input_size
        scale_x = original_width / input_width
        scale_y = original_height / input_height

        detection = detections[0] if isinstance(detections, list) else detections
        if not isinstance(detection, dict):
            return []

        boxes = detection.get('boxes', [])
        scores = detection.get('scores', [])
        class_ids = detection.get('class_ids', [])

        if not (len(boxes) == len(scores) == len(class_ids)):
            return []

        filtered_detections = []
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            # 转换tensor
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            if isinstance(score, torch.Tensor):
                score = score.cpu().item()
            if isinstance(class_id, torch.Tensor):
                class_id = class_id.cpu().item()

            # 更宽松的置信度过滤
            if score < self.conf_threshold:
                continue

            try:
                if len(box) != 4:
                    continue
                    
                #坐标转换逻辑
                max_coord = float(np.max(box))
                if 0.0 < max_coord <= 1.0:
                    # 归一化坐标 (center_x, center_y, width, height)
                    cx_norm, cy_norm, w_norm, h_norm = box
                    
                    # 转换为原图坐标
                    center_x_orig = cx_norm * original_width
                    center_y_orig = cy_norm * original_height
                    w_orig = w_norm * original_width
                    h_orig = h_norm * original_height
                    
                    # 计算角点坐标
                    x1 = center_x_orig - w_orig / 2.0
                    y1 = center_y_orig - h_orig / 2.0
                    x2 = center_x_orig + w_orig / 2.0
                    y2 = center_y_orig + h_orig / 2.0
                else:
                    # 输入尺寸坐标
                    if max_coord > input_width:  # 可能是 (x1, y1, x2, y2) 格式
                        x1, y1, x2, y2 = box
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y
                    else:  # (center_x, center_y, width, height) 格式
                        cx_input, cy_input, w_input, h_input = box
                        center_x_orig = cx_input * scale_x
                        center_y_orig = cy_input * scale_y
                        w_orig = w_input * scale_x
                        h_orig = h_input * scale_y
                        
                        x1 = center_x_orig - w_orig / 2.0
                        y1 = center_y_orig - h_orig / 2.0
                        x2 = center_x_orig + w_orig / 2.0
                        y2 = center_y_orig + h_orig / 2.0

                # 限制在图像范围内，但保持合理的检测框大小
                x1 = max(0.0, min(x1, original_width - 1))
                y1 = max(0.0, min(y1, original_height - 1))
                x2 = max(x1 + 10, min(x2, original_width))  # 确保最小宽度
                y2 = max(y1 + 10, min(y2, original_height))  # 确保最小高度

                width = x2 - x1
                height = y2 - y1
                
                # 更宽松的几何过滤
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                
                # 放宽面积和宽高比限制
                if (area >= self.min_box_area and area <= self.max_box_area and
                    aspect_ratio >= self.min_aspect_ratio and aspect_ratio <= self.max_aspect_ratio and
                    width >= 10 and height >= 10):  # 确保最小尺寸
                    
                    # 类别过滤
                    if 0 <= class_id < len(self.class_names):
                        filtered_detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'score': float(score),
                            'class_id': int(class_id),
                            'area': area
                        })

            except Exception as e:
                if self.debug:
                    print(f"处理检测框 {i} 时出错: {e}")
                continue

        # 更宽松的NMS，按类别分别处理
        if not filtered_detections:
            return []
        
        final_detections = []
        for class_id in set(det['class_id'] for det in filtered_detections):
            class_detections = [det for det in filtered_detections if det['class_id'] == class_id]
            
            if class_detections:
                class_detections.sort(key=lambda x: x['score'], reverse=True)
                
                boxes_np = np.array([det['bbox'] for det in class_detections])
                scores_np = np.array([det['score'] for det in class_detections])
                
                # 使用更宽松的NMS
                indices = cv2.dnn.NMSBoxes(
                    boxes_np.tolist(), 
                    scores_np.tolist(), 
                    self.conf_threshold * 0.8,  # 进一步降低NMS置信度阈值
                    self.nms_iou_threshold
                )
                
                if len(indices) > 0:
                    indices = indices.flatten()
                    for idx in indices:
                        det = class_detections[idx]
                        final_detections.append([
                            det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3],
                            det['score'], det['class_id']
                        ])

        if self.debug and final_detections:
            print(f"最终检测结果: {len(final_detections)} 个目标")
            for i, det in enumerate(final_detections[:3]):  # 只打印前3个
                print(f"  检测 {i}: bbox={[f'{x:.1f}' for x in det[:4]]}, "
                    f"conf={det[4]:.3f}, class={det[5]}")

        return final_detections

    def process_video(self, video_path: str, output_path: str = None, save_stats: bool = True, 
                 show_realtime: bool = True):
        """处理视频 - 支持实时显示"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames if total_frames > 0 else 'N/A'} 帧")
        
        # 设置输出视频
        out_writer = None
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            fourcc_out = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc_out, fps, (width, height))
            print(f"输出视频将保存到: {output_path}")
        
        # 创建显示窗口
        if show_realtime:
            cv2.namedWindow('车辆检测与计数', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('车辆检测与计数', 1200, 800)
            print("按 'q' 键退出实时显示，按 'p' 键暂停/继续")
        
        frame_count = 0
        processing_start_time = time.time()
        detection_count = 0
        paused = False
        last_stats_print = time.time()
        
        print("开始处理视频...")
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("视频流结束.")
                        break
                    
                    frame_count += 1
                    current_loop_time = time.time()
                    
                    try:
                        # 检测车辆
                        detections = self.detect_vehicles(frame.copy())
                        detection_count += len(detections)
                        
                        # 更新追踪器
                        tracked_objects = self.tracker.update(detections, frame)
                        
                        # 更新流量计数
                        self.traffic_counter.update(tracked_objects, current_loop_time)
                        
                        # 可视化
                        self._visualize_frame(frame, tracked_objects)
                        
                    except Exception as e:
                        print(f"处理第 {frame_count} 帧时出错: {e}")
                        if self.debug:
                            import traceback
                            traceback.print_exc()
                        continue
                    
                    # 绘制计数线和统计信息
                    frame = self.traffic_counter.draw_counting_lines(frame)
                    self._draw_statistics_enhanced(frame, current_loop_time)
                    
                    # 写入输出视频
                    if out_writer:
                        out_writer.write(frame)
                
                # 实时显示
                if show_realtime:
                    if not paused:
                        cv2.imshow('车辆检测与计数', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("用户请求退出")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print(f"{'暂停' if paused else '继续'}播放")
                    elif key == ord('s'):  # 保存当前帧
                        screenshot_path = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        print(f"截图已保存: {screenshot_path}")
                
                # 进度报告（每10秒一次）
                if time.time() - last_stats_print > 10:
                    elapsed_time = time.time() - processing_start_time
                    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    
                    stats = self.traffic_counter.get_statistics()
                    debug_info = stats['debug_info']
                    
                    print(f"进度: {frame_count}/{total_frames if total_frames > 0 else '~'} ({progress:.1f}%), "
                        f"处理FPS: {avg_fps:.1f}, 总计数: {stats['total_count']}, "
                        f"追踪数: {debug_info['total_tracks']}, "
                        f"穿越尝试: {debug_info['crossing_attempts']}, "
                        f"成功计数: {debug_info['successful_counts']}")
                    
                    last_stats_print = time.time()

        except KeyboardInterrupt:
            print("\n用户中断处理")
        
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            if show_realtime:
                cv2.destroyAllWindows()

        print(f"处理完成! 总帧数: {frame_count}, 总检测数: {detection_count}")
        
        if save_stats:
            self._save_statistics(video_path)
        
        self._print_final_statistics()
        print(f"视频处理完成! 总用时: {(time.time() - processing_start_time):.2f} 秒.")

    def _draw_statistics_enhanced(self, frame: np.ndarray, current_time_sec: float):
        """增强的统计信息显示（英文版本）"""
        stats = self.traffic_counter.get_statistics()
        debug_info = stats['debug_info']
        
        # 背景区域
        overlay_x, overlay_y, overlay_w, overlay_h = 10, 10, 450, 200
        sub_img = frame[overlay_y:overlay_y+overlay_h, overlay_x:overlay_x+overlay_w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 0)
        frame[overlay_y:overlay_y+overlay_h, overlay_x:overlay_x+overlay_w] = res

        text_color = (255, 255, 255)
        font_scale = 0.6
        line_height = 22
        current_y = overlay_y + line_height

        # 总车辆数 - 突出显示
        cv2.putText(frame, f"Total Vehicles: {stats['total_count']}", 
                    (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        current_y += line_height + 5
        
        # 流量
        flow_rate = stats['current_flow_rate_per_minute']
        cv2.putText(frame, f"Flow Rate: {flow_rate['total']}/min", 
                    (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, text_color, 1, cv2.LINE_AA)
        current_y += line_height

        # 按类别计数（英文类别名）
        class_name_mapping = {
            'motorbike': 'Motorbike',
            'car': 'Car', 
            'bus': 'Bus',
            'truck': 'Truck'
        }
        
        class_counts_str = ", ".join([
            f"{class_name_mapping.get(self.class_names[cid], self.class_names[cid] if cid < len(self.class_names) else 'Unknown')}: {cnt}" 
            for cid, cnt in stats['count_by_class'].items()
        ])
        cv2.putText(frame, f"By Type: {class_counts_str}", 
                    (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, text_color, 1, cv2.LINE_AA)
        current_y += line_height

        # 方向计数
        direction_mapping = {'up': 'Up', 'down': 'Down', 'both': 'Both'}
        direction_counts_str = ", ".join([
            f"{direction_mapping.get(direction, direction.capitalize())}: {count}" 
            for direction, count in stats['count_by_direction'].items()
        ])
        if direction_counts_str:
            cv2.putText(frame, f"Direction: {direction_counts_str}", 
                        (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, text_color, 1, cv2.LINE_AA)
            current_y += line_height

        # 按线计数
        line_counts_str = ", ".join([
            f"Line{line_idx+1}: {count}" for line_idx, count in stats['count_by_line'].items()
        ])
        if line_counts_str:
            cv2.putText(frame, f"By Line: {line_counts_str}", 
                        (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, text_color, 1, cv2.LINE_AA)
            current_y += line_height
        
        # 调试信息
        cv2.putText(frame, f"Tracks: {debug_info['total_tracks']}, "
                        f"Crossing Attempts: {debug_info['crossing_attempts']}, "
                        f"Success: {debug_info['successful_counts']}", 
                    (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (255, 255, 0), 1, cv2.LINE_AA)
        current_y += line_height
        
        # 时间戳
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time_sec))
        cv2.putText(frame, time_str, (overlay_x + 10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    def _visualize_frame(self, frame, tracked_objects):
        """可视化，显示更多信息"""
        for obj in tracked_objects:
            track_id = obj['track_id']
            x1, y1, x2, y2 = [int(c) for c in obj['bbox']]
            class_id = obj['class_id']
            confidence = obj.get('confidence', 0.0)
            hits = obj.get('hits', 0)
            time_since_update = obj.get('time_since_update', 0)
            
            # 确保类别ID有效
            if 0 <= class_id < len(self.class_names):
                color_idx = class_id % len(self.colors)
                color = self.colors[color_idx]
                class_name = self.class_names[class_id]
            else:
                color = (128, 128, 128)  # 灰色
                class_name = "未知"
            
            # 根据追踪稳定性调整颜色亮度
            if hits < 3:  # 新追踪，颜色稍淡
                color = tuple(int(c * 0.7) for c in color)
            
            # 绘制边界框
            thickness = 3 if hits >= 5 else 2  # 稳定追踪用更粗的线
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签 - 包含更多信息
            label_text = f"{class_name} ID:{track_id}"
            if self.debug:
                label_text += f" H:{hits} T:{time_since_update}"
                
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1 - 5), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 为新追踪添加特殊标记
            if hits < 3:
                cv2.circle(frame, (x1 + 10, y1 + 10), 5, (0, 255, 255), -1)

    def _save_statistics(self, video_path: str):
        """保存统计数据"""
        stats = self.traffic_counter.get_statistics()
        save_data = {
            'video_file': Path(video_path).name,
            'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'configuration': {
                'model_input_size': self.config['model']['input_size'],
                'detection_conf_thresh': self.conf_threshold,
                'detection_iou_thresh': self.iou_threshold,
                'min_box_area': self.min_box_area,
                'max_box_area': self.max_box_area,
                'counting_lines': self.traffic_counter.counting_lines,
                'count_direction': self.traffic_counter.count_direction
            },
            'traffic_statistics': stats,
        }
        
        output_filename = Path(video_path).stem + "_optimized_analysis.json"
        output_dir = Path(self.config.get('training',{}).get('logs_dir', 'experiments/logs')) / "analysis_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        full_output_path = output_dir / output_filename

        try:
            with open(full_output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=4, default=str)
            print(f"分析结果已保存到: {full_output_path}")
        except Exception as e:
            print(f"保存统计数据失败: {e}")

    def _print_final_statistics(self):
        """打印最终统计结果"""
        stats = self.traffic_counter.get_statistics()
        print("\n" + "="*60)
        print("最终统计结果 ")
        print("="*60)
        print(f"总车辆数: {stats['total_count']}")
        
        print("\n按类别统计:")
        for class_id, count in stats['count_by_class'].items():
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"未知类别 {class_id}"
            print(f"  {class_name}: {count}")
        
        print("\n按方向统计:")
        for direction, count in stats['count_by_direction'].items():
            print(f"  {direction.capitalize()}: {count}")
        
        flow_rate = stats['current_flow_rate_per_minute']
        print(f"\n当前流量: {flow_rate['total']} 车辆/分钟")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='的视频车辆分析系统')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, help='输出视频路径')
    parser.add_argument('--no-save-stats', action='store_true', help='不保存统计结果')
    parser.add_argument('--no-display', action='store_true', help='不显示实时视频')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 验证文件存在
    for file_path, name in [(args.config, "配置文件"), (args.model, "模型文件"), (args.video, "视频文件")]:
        if not Path(file_path).exists():
            print(f"{name}不存在: {file_path}")
            return
    
    analyzer = VideoVehicleAnalyzer(args.config, args.model, debug=args.debug)
    analyzer.process_video(
        video_path=args.video,
        output_path=args.output,
        save_stats=not args.no_save_stats,
        show_realtime=not args.no_display
    )

if __name__ == '__main__':
    main()

