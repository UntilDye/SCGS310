import cv2
import torch
import numpy as np
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import tempfile
import os
from PIL import Image
from torchvision import transforms

warnings.filterwarnings('ignore')

from models.yolo_vehicle import VehicleYOLO
from utils.deepsort_tracker import VehicleTracker
from utils.traffic_counter import TrafficCounter

# 车牌相关导入
from models.YOLO2 import YOLO2_MobileNetV2 as YOLO2
from models.Crnn2 import CRNN

class LicensePlateRecognizer:
    """车牌识别器 - 集成YOLO检测和CRNN识别"""
    
    def __init__(self, yolo_config_path: str, yolo_model_path: str, crnn_model_path: str, device, debug=False):
        self.device = device
        self.debug = debug
        self.tmp_dir = tempfile.mkdtemp()
        
        # 首先初始化字符集
        self.CHARACTER_SET = self._get_charset()
        self.INDEX_TO_CHAR = {idx + 1: char for idx, char in enumerate(self.CHARACTER_SET)}
        self.INDEX_TO_CHAR[0] = ''  # blank for CTC
        
        if self.debug:
            print(f"[DEBUG] 字符集大小: {len(self.CHARACTER_SET)}")
            print(f"[DEBUG] 字符集: {self.CHARACTER_SET}")
        
        # 加载YOLO车牌检测模型
        self.yolo_model, self.yolo_cfg = self._load_yolo_model(yolo_config_path, yolo_model_path)
        
        # 加载CRNN车牌识别模型
        self.crnn_model = self._load_crnn_model(crnn_model_path)
        
        print(f"车牌识别器初始化完成，使用设备: {device}")
    
    def _get_charset(self):
        """获取字符集"""
        provincelist = [
            "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京",
            "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "西",
            "陕", "甘", "青", "宁", "新"
        ]
        wordlist = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
            "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
        ]
        merged_string = ''.join(provincelist + wordlist)
        return ''.join(sorted(set(merged_string)))
    
    def _load_yolo_model(self, config_path: str, model_path: str):
        """加载YOLO车牌检测模型"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            if self.debug:
                print(f"[DEBUG] YOLO配置加载成功: {cfg}")
        except Exception as e:
            print(f"加载YOLO配置文件失败: {e}")
            # 使用默认配置
            cfg = {
                'model': {
                    'nc': 1,  # 车牌检测只有一个类别
                    's': 7    # 网格大小
                }
            }
            if self.debug:
                print(f"[DEBUG] 使用默认YOLO配置: {cfg}")
        
        try:
            model = YOLO2(nc=cfg['model']['nc'], S=cfg['model']['s']).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"YOLO车牌检测模型加载成功")
            if self.debug:
                print(f"[DEBUG] YOLO模型参数数量: {sum(p.numel() for p in model.parameters())}")
            return model, cfg
        except Exception as e:
            print(f"加载YOLO车牌检测模型失败: {e}")
            raise
    
    def _load_crnn_model(self, model_path: str):
        """加载CRNN车牌识别模型"""
        try:
            model = CRNN(num_classes=len(self.CHARACTER_SET)+1, input_channels=1)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if self.debug:
                print(f"[DEBUG] CRNN模型结构: {model}")
                print(f"[DEBUG] 检查点键: {checkpoint.keys() if isinstance(checkpoint, dict) else '非字典格式'}")
            
            # 处理不同的保存格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
                
            model = model.to(self.device)
            model.eval()
            print(f"CRNN车牌识别模型加载成功")
            if self.debug:
                print(f"[DEBUG] CRNN模型参数数量: {sum(p.numel() for p in model.parameters())}")
            return model
        except Exception as e:
            print(f"加载CRNN车牌识别模型失败: {e}")
            raise
    
    def _preprocess_yolo_image(self, img_array: np.ndarray, img_size: int):
        """预处理图像用于YOLO检测"""
        try:
            if self.debug:
                print(f"[DEBUG] YOLO预处理输入: shape={img_array.shape}, size={img_size}")
            
            # 转换为PIL图像
            if len(img_array.shape) == 3:
                img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            else:
                img_pil = Image.fromarray(img_array)
            
            # 预处理
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
            img_tensor = transform(img_pil).unsqueeze(0)
            
            if self.debug:
                print(f"[DEBUG] YOLO预处理输出: PIL size={img_pil.size}, tensor shape={img_tensor.shape}")
            
            return img_pil, img_tensor
        except Exception as e:
            print(f"YOLO图像预处理失败: {e}")
            raise
    
    def _yolo_box_to_img_coords(self, cell_row, cell_col, x, y, w, h, S, net_w, net_h, img_w, img_h):
        """YOLO坐标转换"""
        try:
            x_center = ((cell_col + x) / S) * net_w
            y_center = ((cell_row + y) / S) * net_h
            bw = w * net_w
            bh = h * net_h
            
            scale_x = img_w / net_w
            scale_y = img_h / net_h
            
            x_center *= scale_x
            y_center *= scale_y
            bw *= scale_x
            bh *= scale_y
            
            x1 = x_center - bw / 2
            y1 = y_center - bh / 2
            x2 = x_center + bw / 2
            y2 = y_center + bh / 2
            
            if self.debug:
                print(f"[DEBUG] YOLO坐标转换: ({cell_row},{cell_col}) -> ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
            
            return [x1, y1, x2, y2]
        except Exception as e:
            print(f"YOLO坐标转换失败: {e}")
            return [0, 0, 100, 50]  # 返回默认值
    
    def _detect_license_plate(self, img_array: np.ndarray, threshold: float = 0.2):
        """使用YOLO检测车牌"""
        try:
            if self.debug:
                print(f"[DEBUG] 开始车牌检测，输入shape: {img_array.shape}, 阈值: {threshold}")
            
            img_size = self.yolo_cfg.get('training', {}).get('img_size', 416)
            if isinstance(img_size, list):
                img_size = img_size[0]
            
            img_pil, img_tensor = self._preprocess_yolo_image(img_array, img_size)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                pred = self.yolo_model(img_tensor)  # [1, S, S, 5]
                pred = pred[0].cpu()      # [S, S, 5]
                S = pred.shape[0]
                
                conf = torch.sigmoid(pred[..., 0])
                best = torch.argmax(conf)
                i, j = np.unravel_index(best, (S, S))
                
                max_confidence = conf[i, j].item()
                
                if self.debug:
                    print(f"[DEBUG] YOLO预测: S={S}, 最高置信度={max_confidence:.3f} at ({i},{j})")
                
                if max_confidence < threshold:
                    if self.debug:
                        print(f"[DEBUG] 置信度{max_confidence:.3f}低于阈值{threshold}，未检测到车牌")
                    return None
                
                x, y, w, h = pred[i, j, 1:5].numpy()
                net_w, net_h = img_tensor.shape[-1], img_tensor.shape[-2]
                img_w, img_h = img_pil.size
                
                x1, y1, x2, y2 = self._yolo_box_to_img_coords(
                    i, j, x, y, w, h, S, net_w, net_h, img_w, img_h
                )
                
                # 坐标校正
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
                
                # 检查边界框有效性
                if x2 <= x1 or y2 <= y1:
                    if self.debug:
                        print(f"[DEBUG] 无效边界框: ({x1},{y1},{x2},{y2})")
                    return None
                
                if self.debug:
                    print(f"[DEBUG] 检测到车牌: bbox=({x1},{y1},{x2},{y2}), 置信度={max_confidence:.3f}")
                
                return (x1, y1, x2, y2)
                
        except Exception as e:
            print(f"车牌检测失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _crop_license_plate(self, img_array: np.ndarray, bbox: Tuple[int, int, int, int]):
        """裁剪车牌区域"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = img_array.shape[:2]
            
            if self.debug:
                print(f"[DEBUG] 裁剪车牌: 原图{w}x{h}, bbox=({x1},{y1},{x2},{y2})")
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            cropped = img_array[y1:y2, x1:x2]
            
            # 检查裁剪结果
            if cropped.size == 0:
                if self.debug:
                    print(f"[DEBUG] 裁剪结果为空")
                return None
            
            if self.debug:
                print(f"[DEBUG] 裁剪成功: {cropped.shape}")
                
            return cropped
        except Exception as e:
            print(f"车牌裁剪失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _get_infer_transform(self, img_size=(32, 320)):
        """获取CRNN推理预处理变换"""
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    
    def _decode_crnn_output(self, preds):
        """解码CRNN输出"""
        try:
            if self.debug:
                print(f"[DEBUG] CRNN输出shape: {preds.shape}")
            
            pred_indices = preds.argmax(dim=2)
            pred_indices = pred_indices.permute(1, 0)
            results = []
            
            if self.debug:
                print(f"[DEBUG] 预测索引shape: {pred_indices.shape}")
            
            for batch_idx, indices in enumerate(pred_indices):
                prev_idx = -1
                text = ''
                char_details = []
                
                for pos, idx in enumerate(indices):
                    idx = idx.item()
                    char = self.INDEX_TO_CHAR.get(idx, '')
                    
                    if self.debug:
                        char_details.append(f"pos{pos}:idx{idx}->'{char}'")
                    
                    if idx != 0 and idx != prev_idx:  # 不是空白且不重复
                        if char:  # 只添加有效字符
                            text += char
                    prev_idx = idx
                
                if self.debug:
                    print(f"[DEBUG] 批次{batch_idx} 字符详情: {char_details[:10]}...")  # 只显示前10个
                    print(f"[DEBUG] 批次{batch_idx} 最终文本: '{text}'")
                
                results.append(text)
            
            return results
        except Exception as e:
            print(f"CRNN输出解码失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return [""]
    
    def _recognize_license_plate(self, cropped_img: np.ndarray):
        """使用CRNN识别车牌文字"""
        try:
            if cropped_img is None or cropped_img.size == 0:
                if self.debug:
                    print(f"[DEBUG] 裁剪图像无效")
                return ""
            
            if self.debug:
                print(f"[DEBUG] 开始CRNN识别，输入shape: {cropped_img.shape}")
            
            # 转换为PIL图像
            if len(cropped_img.shape) == 3:
                img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            else:
                img_pil = Image.fromarray(cropped_img)
            
            if self.debug:
                print(f"[DEBUG] PIL图像转换完成: {img_pil.size}")
            
            # 预处理
            transform = self._get_infer_transform()
            image = transform(img_pil).unsqueeze(0).to(self.device)
            
            if self.debug:
                print(f"[DEBUG] CRNN输入tensor shape: {image.shape}")
            
            # 推理
            with torch.no_grad():
                output = self.crnn_model(image)
                output = output.permute(1, 0, 2)
                
                if self.debug:
                    print(f"[DEBUG] CRNN原始输出shape: {output.shape}")
            
            # 解码
            texts = self._decode_crnn_output(output)
            result_text = texts[0] if texts else ""
            
            if self.debug:
                print(f"[DEBUG] CRNN最终识别结果: '{result_text}'")
            
            return result_text
            
        except Exception as e:
            print(f"车牌识别出错: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return ""
    
    def recognize_from_vehicle_crop(self, vehicle_img: np.ndarray):
        """从车辆裁剪图像中识别车牌"""
        try:
            if vehicle_img is None or vehicle_img.size == 0:
                if self.debug:
                    print(f"[DEBUG] 车辆图像无效")
                return None
            
            if self.debug:
                print(f"[DEBUG] 开始从车辆图像识别车牌，输入shape: {vehicle_img.shape}")
            
            # 第一步：检测车牌位置
            bbox = self._detect_license_plate(vehicle_img, threshold=0.2)
            if bbox is None:
                if self.debug:
                    print(f"[DEBUG] 未检测到车牌位置")
                return None
            
            # 第二步：裁剪车牌区域
            license_plate_crop = self._crop_license_plate(vehicle_img, bbox)
            if license_plate_crop is None:
                if self.debug:
                    print(f"[DEBUG] 车牌区域裁剪失败")
                return None
            
            # 第三步：识别车牌文字
            license_text = self._recognize_license_plate(license_plate_crop)
            
            if self.debug:
                print(f"[DEBUG] 车牌识别完成: '{license_text}', 长度: {len(license_text.strip())}")
            
            # 过滤无效结果
            if license_text and len(license_text.strip()) >= 6:
                result = {
                    'bbox': bbox,
                    'text': license_text.strip(),
                    'confidence': 0.8  # 可以根据实际情况调整
                }
                if self.debug:
                    print(f"[DEBUG] 车牌识别成功: {result}")
                return result
            else:
                if self.debug:
                    print(f"[DEBUG] 车牌文本长度不足或为空，被过滤")
            
            return None
            
        except Exception as e:
            print(f"车牌识别过程出错: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def __del__(self):
        """清理临时目录"""
        try:
            import shutil
            if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
        except:
            pass

class VideoVehicleAnalyzerWithLicensePlate:
    """带车牌识别的视频车辆分析器"""
    
    def __init__(self, config_path: str, vehicle_model_path: str, 
             license_yolo_config: str, license_yolo_model: str, 
             license_crnn_model: str, debug: bool = False):
    
        self.debug = debug
        
        # 初始化统计变量时确保类型正确
        self.total_license_attempts = 0
        self.successful_license_recognitions = 0
        self.recognized_licenses = set()
        self.license_output_log = []
        self.license_cache = {}
        self.license_cache_ttl = 30
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.class_names = self.config['model']['class_names']
        
        # 加载车辆检测模型
        self.vehicle_model = self._load_vehicle_model(vehicle_model_path)
        
        # 初始化车牌识别器
        try:
            self.license_recognizer = LicensePlateRecognizer(
                license_yolo_config, license_yolo_model, license_crnn_model, 
                self.device, debug=self.debug
            )
            self.license_enabled = True
            print(f"车牌识别功能已启用")
        except Exception as e:
            print(f"车牌识别器初始化失败: {e}")
            print("将继续使用车辆检测功能，但车牌识别功能将被禁用")
            if self.debug:
                import traceback
                traceback.print_exc()
            self.license_recognizer = None
            self.license_enabled = False
        
        # 车辆检测参数
        detection_cfg = self.config.get('detection', {})
        self.conf_threshold = float(detection_cfg.get('conf_threshold', 0.15))
        self.iou_threshold = float(detection_cfg.get('iou_threshold', 0.45))
        self.nms_iou_threshold = float(detection_cfg.get('nms_iou_threshold', 0.5))
        
        # 过滤参数
        filtering_cfg = self.config.get('filtering', {})
        self.min_box_area = filtering_cfg.get('min_box_area', 100)
        self.max_box_area = filtering_cfg.get('max_box_area', 500000)
        self.min_aspect_ratio = filtering_cfg.get('min_aspect_ratio', 0.05)
        self.max_aspect_ratio = filtering_cfg.get('max_aspect_ratio', 20.0)
        
        # 追踪器配置
        tracker_config = self.config.get('tracking', {})
        self.tracker = VehicleTracker(
            class_names=self.class_names,
            max_age=30,
            n_init=1,
            max_iou_distance=0.7,
            debug=debug
        )
        
        # 流量计数器
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
        
        # 车牌识别缓存 - 避免重复识别
        self.license_cache = {}
        self.license_cache_ttl = 30  # 缓存30帧
        
        # 添加车牌输出相关变量
        self.recognized_licenses = set()  # 存储已识别的车牌，避免重复输出
        self.license_output_log = []  # 存储车牌输出日志
        
        # 统计变量
        self.total_license_attempts = 0  # 尝试识别车牌的总次数
        self.successful_license_recognitions = 0  # 成功识别的次数
    
    def _load_vehicle_model(self, model_path: str):
        """加载车辆检测模型"""
        print(f"加载车辆检测模型: {model_path}")
        
        input_size_cfg = self.config['model']['input_size']
        try:
            input_shape = (int(input_size_cfg[0]), int(input_size_cfg[1]))
        except (TypeError, IndexError, ValueError) as e:
            print(f"从配置解析 input_size 时出错: {input_size_cfg}. 使用默认 (640, 640)")
            input_shape = (640, 640)
        
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
            print(f"加载车辆检测模型权重时出错: {e}")
            raise
        
        model.to(self.device)
        model.eval()
        print("车辆检测模型加载完成")
        return model
    
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
                if hasattr(self.vehicle_model, 'predict') and callable(self.vehicle_model.predict):
                    raw_preds = self.vehicle_model.predict(
                        img_tensor,
                        conf_threshold=self.conf_threshold,
                        iou_threshold=self.iou_threshold,
                        device=self.device
                    )
                else:
                    raw_preds = self.vehicle_model(img_tensor)
            except Exception as e:
                print(f"车辆检测模型推理失败: {e}")
                return []
        
        # 后处理和过滤
        results = self._post_process_and_filter(
            raw_preds,
            (img_orig_w, img_orig_h),
            (input_w, input_h)
        )
        
        return results
    
    def _post_process_and_filter(self, detections, original_size, input_size):
        """后处理和过滤方法（保持原有逻辑）"""
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

            # 置信度过滤
            if score < self.conf_threshold:
                continue

            try:
                if len(box) != 4:
                    continue
                
                # 坐标转换逻辑（保持原有逻辑）
                max_coord = float(np.max(box))
                if 0.0 < max_coord <= 1.0:
                    # 归一化坐标
                    cx_norm, cy_norm, w_norm, h_norm = box
                    
                    center_x_orig = cx_norm * original_width
                    center_y_orig = cy_norm * original_height
                    w_orig = w_norm * original_width
                    h_orig = h_norm * original_height
                    
                    x1 = center_x_orig - w_orig / 2.0
                    y1 = center_y_orig - h_orig / 2.0
                    x2 = center_x_orig + w_orig / 2.0
                    y2 = center_y_orig + h_orig / 2.0
                else:
                    # 输入尺寸坐标
                    if max_coord > input_width:
                        x1, y1, x2, y2 = box
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y
                    else:
                        cx_input, cy_input, w_input, h_input = box
                        center_x_orig = cx_input * scale_x
                        center_y_orig = cy_input * scale_y
                        w_orig = w_input * scale_x
                        h_orig = h_input * scale_y
                        
                        x1 = center_x_orig - w_orig / 2.0
                        y1 = center_y_orig - h_orig / 2.0
                        x2 = center_x_orig + w_orig / 2.0
                        y2 = center_y_orig + h_orig / 2.0

                # 限制在图像范围内
                x1 = max(0.0, min(x1, original_width - 1))
                y1 = max(0.0, min(y1, original_height - 1))
                x2 = max(x1 + 10, min(x2, original_width))
                y2 = max(y1 + 10, min(y2, original_height))

                width = x2 - x1
                height = y2 - y1
                
                # 几何过滤
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                
                if (area >= self.min_box_area and area <= self.max_box_area and
                    aspect_ratio >= self.min_aspect_ratio and aspect_ratio <= self.max_aspect_ratio and
                    width >= 10 and height >= 10):
                    
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

        # NMS处理
        if not filtered_detections:
            return []
        
        final_detections = []
        for class_id in set(det['class_id'] for det in filtered_detections):
            class_detections = [det for det in filtered_detections if det['class_id'] == class_id]
            
            if class_detections:
                class_detections.sort(key=lambda x: x['score'], reverse=True)
                
                boxes_np = np.array([det['bbox'] for det in class_detections])
                scores_np = np.array([det['score'] for det in class_detections])
                
                indices = cv2.dnn.NMSBoxes(
                    boxes_np.tolist(),
                    scores_np.tolist(),
                    self.conf_threshold * 0.8,
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

        return final_detections
    
    def _recognize_license_plates(self, frame: np.ndarray, tracked_objects: List, frame_count: int):
        """识别车牌并添加到追踪对象中"""
        if not self.license_enabled:
            return
            
        for obj in tracked_objects:
            # 确保 track_id 是整数类型
            track_id = obj.get('track_id', 0)
            try:
                track_id = int(track_id) if not isinstance(track_id, int) else track_id
            except (ValueError, TypeError):
                if self.debug:
                    print(f"[DEBUG] 无效的 track_id: {obj.get('track_id')}, 跳过车牌识别")
                continue
            
            # 检查缓存，避免频繁识别
            cache_key = f"{track_id}_{frame_count // self.license_cache_ttl}"
            if cache_key in self.license_cache:
                obj['license_plate'] = self.license_cache[cache_key]
                if self.debug:
                    print(f"[DEBUG] 车辆{track_id}使用缓存的车牌: {self.license_cache[cache_key]}")
                continue
            
            # 车辆检测置信度过滤 - 只对高置信度的车辆进行车牌识别
            confidence = obj.get('confidence', 0)
            try:
                confidence = float(confidence) if confidence is not None else 0.0
            except (ValueError, TypeError):
                confidence = 0.0
                
            if confidence < 0.5:
                if self.debug:
                    print(f"[DEBUG] 车辆{track_id}置信度{confidence:.2f}过低，跳过车牌识别")
                continue
            
            self.total_license_attempts += 1
            
            try:
                # 裁剪车辆区域
                bbox = obj.get('bbox', [0, 0, 100, 100])
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    if self.debug:
                        print(f"[DEBUG] 车辆{track_id}边界框无效: {bbox}")
                    continue
                    
                x1, y1, x2, y2 = [int(float(c)) for c in bbox]  # 确保坐标是整数
                
                if self.debug:
                    print(f"[DEBUG] 车辆{track_id}边界框: ({x1},{y1},{x2},{y2})")
                
                # 扩展边界框以包含更多车牌区域
                margin_x = max(1, int((x2 - x1) * 0.1))
                margin_y = max(1, int((y2 - y1) * 0.1))
                
                x1_expanded = max(0, x1 - margin_x)
                y1_expanded = max(0, y1 - margin_y)
                x2_expanded = min(frame.shape[1], x2 + margin_x)
                y2_expanded = min(frame.shape[0], y2 + margin_y)
                
                vehicle_crop = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                
                if self.debug:
                    print(f"[DEBUG] 车辆{track_id}扩展后区域: ({x1_expanded},{y1_expanded},{x2_expanded},{y2_expanded})")
                    print(f"[DEBUG] 车辆{track_id}裁剪区域shape: {vehicle_crop.shape}")
                
                # 检查裁剪区域是否有效
                if vehicle_crop.size == 0 or vehicle_crop.shape[0] < 30 or vehicle_crop.shape[1] < 30:
                    if self.debug:
                        print(f"[DEBUG] 车辆{track_id}裁剪区域太小，跳过车牌识别")
                    continue
                
                # 车牌识别
                license_result = self.license_recognizer.recognize_from_vehicle_crop(vehicle_crop)
                
                if self.debug:
                    print(f"[DEBUG] 车辆{track_id}车牌识别原始结果: {license_result}")
                
                if license_result and license_result.get('text'):
                    # 过滤明显错误的车牌结果
                    license_text = str(license_result['text']).strip()  # 确保是字符串
                    
                    if self.debug:
                        print(f"[DEBUG] 车辆{track_id}识别到车牌文本: '{license_text}', 长度: {len(license_text)}")
                    
                    if len(license_text) >= 6:  # 至少6个字符
                        obj['license_plate'] = license_result
                        self.license_cache[cache_key] = license_result
                        self.successful_license_recognitions += 1
                        
                        # 输出车牌到控制台（避免重复输出）
                        self._output_license_to_console(track_id, license_text, frame_count, obj)
                    else:
                        if self.debug:
                            print(f"[DEBUG] 车辆{track_id}车牌文本太短，被过滤: '{license_text}'")
                else:
                    if self.debug:
                        print(f"[DEBUG] 车辆{track_id}未识别到有效车牌")
                    
            except Exception as e:
                print(f"车辆{track_id}车牌识别出错: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                continue
        
    def _output_license_to_console(self, track_id: int, license_text: str, frame_count: int, vehicle_obj: dict):
        """输出车牌信息到控制台"""
        # 检查车牌文本是否有效
        if not license_text or not license_text.strip():
            if self.debug:
                print(f"[DEBUG] 警告: 车辆 {track_id} 的车牌文本为空")
            return
        
        license_text = license_text.strip()
        
        # 确保 track_id 是整数类型
        try:
            track_id = int(track_id) if not isinstance(track_id, int) else track_id
        except (ValueError, TypeError):
            track_id = 0
            if self.debug:
                print(f"[DEBUG] 警告: track_id 转换失败，使用默认值 0")
        
        # 创建唯一标识符，避免同一车辆的同一车牌重复输出
        license_key = f"{track_id}_{license_text}"
        
        if license_key not in self.recognized_licenses:
            self.recognized_licenses.add(license_key)
            
            # 获取当前时间戳
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # 获取车辆信息 - 添加类型检查和默认值
            vehicle_class = vehicle_obj.get('class_id', -1)
            try:
                vehicle_class = int(vehicle_class) if vehicle_class is not None else -1
            except (ValueError, TypeError):
                vehicle_class = -1
                
            vehicle_class_name = self.class_names[vehicle_class] if 0 <= vehicle_class < len(self.class_names) else "未知"
            
            vehicle_confidence = vehicle_obj.get('confidence', 0.0)
            try:
                vehicle_confidence = float(vehicle_confidence) if vehicle_confidence is not None else 0.0
            except (ValueError, TypeError):
                vehicle_confidence = 0.0
                
            vehicle_bbox = vehicle_obj.get('bbox', [0, 0, 0, 0])
            if not isinstance(vehicle_bbox, (list, tuple)) or len(vehicle_bbox) != 4:
                vehicle_bbox = [0, 0, 0, 0]
            
            # 确保 frame_count 是整数
            try:
                frame_count = int(frame_count) if not isinstance(frame_count, int) else frame_count
            except (ValueError, TypeError):
                frame_count = 0
            
            # 突出显示的控制台输出
            print("\n" + "="*70)
            print("🚗 车牌识别成功!")
            print("="*70)
            print(f"   时间: {timestamp}")
            print(f"   车辆ID: {track_id:03d}")  # 确保 track_id 是整数
            print(f"   车辆类型: {vehicle_class_name}")
            print(f"   车辆置信度: {vehicle_confidence:.3f}")
            print(f"   车辆位置: [{int(vehicle_bbox[0])}, {int(vehicle_bbox[1])}, {int(vehicle_bbox[2])}, {int(vehicle_bbox[3])}]")
            print(f"   🏷️  车牌号: 【{license_text}】")  # 用【】包围车牌号增加可见性
            print(f"   检测帧: {frame_count}")
            print(f"   车牌长度: {len(license_text)} 字符")
            
            # 车牌质量评估
            quality_score = self._assess_license_quality(license_text)
            print(f"   车牌质量: {quality_score}")
            
            print("="*70)
            
            # 保存到日志
            log_entry = {
                'timestamp': timestamp,
                'track_id': track_id,
                'vehicle_class': vehicle_class_name,
                'vehicle_confidence': vehicle_confidence,
                'vehicle_bbox': vehicle_bbox,
                'license_plate': license_text,
                'license_quality': quality_score,
                'frame_count': frame_count
            }
            self.license_output_log.append(log_entry)
            
            # 如果开启调试模式，输出更详细的信息
            if self.debug:
                license_plate_info = vehicle_obj.get('license_plate', {})
                print(f"   └─ 调试信息:")
                print(f"      车牌检测置信度: {license_plate_info.get('confidence', 'N/A')}")
                print(f"      车牌检测框: {license_plate_info.get('bbox', 'N/A')}")
                print(f"      字符详细: {[c for c in license_text]}")
                print(f"      识别尝试统计: {self.total_license_attempts} 次尝试, {self.successful_license_recognitions} 次成功")
                
                # 安全的成功率计算
                if self.total_license_attempts > 0:
                    success_rate = (self.successful_license_recognitions/self.total_license_attempts*100)
                    print(f"      成功率: {success_rate:.1f}%")
                else:
                    print(f"      成功率: 0.0%")
                print("="*70)
    
    def _assess_license_quality(self, license_text: str) -> str:
        """评估车牌质量"""
        if len(license_text) < 6:
            return "差"
        elif len(license_text) < 7:
            return "一般"
        elif len(license_text) == 7:
            # 检查是否符合中国车牌格式
            if len(license_text) >= 2:
                # 第一个字符应该是中文省份
                first_char = license_text[0]
                is_chinese = '\u4e00' <= first_char <= '\u9fff'
                if is_chinese:
                    return "优秀"
                else:
                    return "良好"
            return "良好"
        else:
            return "良好"
    
    def _visualize_frame_with_license(self, frame, tracked_objects):
        """可视化，包含车牌信息"""
        for obj in tracked_objects:
            track_id = obj['track_id']
            x1, y1, x2, y2 = [int(c) for c in obj['bbox']]
            class_id = obj['class_id']
            confidence = obj.get('confidence', 0.0)
            hits = obj.get('hits', 0)
            license_plate = obj.get('license_plate', None)
            
            # 确保类别ID有效
            if 0 <= class_id < len(self.class_names):
                color_idx = class_id % len(self.colors)
                color = self.colors[color_idx]
                class_name = self.class_names[class_id]
            else:
                color = (128, 128, 128)
                class_name = "未知"
            
            # 根据追踪稳定性调整颜色亮度
            if hits < 3:
                color = tuple(int(c * 0.7) for c in color)
            
            # 绘制边界框
            thickness = 3 if hits >= 5 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # 准备标签文本
            label_lines = []
            label_lines.append(f"{class_name} ID:{track_id}")
            
            if license_plate:
                license_text = license_plate['text']
                label_lines.append(f"车牌: {license_text}")
            
            if self.debug:
                label_lines.append(f"H:{hits} C:{confidence:.2f}")
            
            # 绘制多行标签
            font_scale = 0.6
            font_thickness = 1
            line_height = 20
            
            # 计算标签背景大小
            max_width = 0
            total_height = len(label_lines) * line_height + 5
            
            for line in label_lines:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                max_width = max(max_width, tw)
            
            # 绘制标签背景
            label_bg_color = color
            cv2.rectangle(frame, (x1, y1 - total_height), (x1 + max_width + 10, y1), label_bg_color, -1)
            
            # 绘制标签文本
            for i, line in enumerate(label_lines):
                text_y = y1 - total_height + (i + 1) * line_height - 5
                text_color = (255, 255, 255) if i == 0 else (255, 255, 0)  # 车牌信息用黄色
                cv2.putText(frame, line, (x1 + 5, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
            # 为新追踪添加特殊标记
            if hits < 3:
                cv2.circle(frame, (x1 + 10, y1 + 10), 5, (0, 255, 255), -1)
            
            # 如果有车牌信息，在车牌位置绘制小框
            if license_plate and 'bbox' in license_plate:
                lp_bbox = license_plate['bbox']
                # 将车牌坐标转换为原图坐标（相对于车辆裁剪区域的偏移）
                margin_x = int((x2 - x1) * 0.1)
                margin_y = int((y2 - y1) * 0.1)
                x1_expanded = max(0, x1 - margin_x)
                y1_expanded = max(0, y1 - margin_y)
                
                lp_x1 = int(x1_expanded + lp_bbox[0])
                lp_y1 = int(y1_expanded + lp_bbox[1])
                lp_x2 = int(x1_expanded + lp_bbox[2])
                lp_y2 = int(y1_expanded + lp_bbox[3])
                
                # 确保坐标在有效范围内
                h, w = frame.shape[:2]
                lp_x1 = max(0, min(lp_x1, w-1))
                lp_y1 = max(0, min(lp_y1, h-1))
                lp_x2 = max(lp_x1+1, min(lp_x2, w))
                lp_y2 = max(lp_y1+1, min(lp_y2, h))
                
                # 绘制车牌框
                cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 255, 255), 2)
    
    def process_video(self, video_path: str, output_path: str = None, save_stats: bool = True,
                     show_realtime: bool = True):
        """处理视频 - 支持车牌识别"""
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
        print(f"车牌识别: {'启用' if self.license_enabled else '禁用'}")
        print(f"调试模式: {'启用' if self.debug else '禁用'}")
        
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
            window_title = '车辆检测与车牌识别' if self.license_enabled else '车辆检测'
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_title, 1200, 800)
            print("按 'q' 键退出，按 'p' 键暂停/继续，按 's' 键截图")
        
        frame_count = 0
        processing_start_time = time.time()
        detection_count = 0
        license_recognition_count = 0
        paused = False
        last_stats_print = time.time()
        
        print("开始处理视频...")
        print("="*80)
        
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
                        
                        if self.debug and detections:
                            print(f"[DEBUG] 第{frame_count}帧检测到{len(detections)}个车辆")
                        
                        # 更新追踪器
                        tracked_objects = self.tracker.update(detections, frame)
                        
                        if self.debug and tracked_objects:
                            print(f"[DEBUG] 第{frame_count}帧追踪到{len(tracked_objects)}个对象")
                        
                        # 车牌识别（每5帧识别一次以提高性能）
                        if self.license_enabled and frame_count % 5 == 0:
                            if self.debug:
                                print(f"[DEBUG] 第{frame_count}帧开始车牌识别")
                            self._recognize_license_plates(frame, tracked_objects, frame_count)
                            license_recognition_count = len(self.recognized_licenses)
                        
                        # 更新流量计数
                        self.traffic_counter.update(tracked_objects, current_loop_time)
                        
                        # 可视化（包含车牌信息）
                        self._visualize_frame_with_license(frame, tracked_objects)
                        
                    except Exception as e:
                        print(f"处理第 {frame_count} 帧时出错: {e}")
                        if self.debug:
                            import traceback
                            traceback.print_exc()
                        continue
                    
                    # 绘制计数线和统计信息
                    frame = self.traffic_counter.draw_counting_lines(frame)
                    self._draw_statistics_enhanced(frame, current_loop_time, license_recognition_count)
                    
                    # 写入输出视频
                    if out_writer:
                        out_writer.write(frame)
                
                # 实时显示
                if show_realtime:
                    if not paused:
                        window_title = '车辆检测与车牌识别' if self.license_enabled else '车辆检测'
                        cv2.imshow(window_title, frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("用户请求退出")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print(f"{'暂停' if paused else '继续'}播放")
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_with_license_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        print(f"截图已保存: {screenshot_path}")
                
                # 进度报告
                if time.time() - last_stats_print > 10:
                    elapsed_time = time.time() - processing_start_time
                    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    
                    stats = self.traffic_counter.get_statistics()
                    
                    status_msg = f"进度: {frame_count}/{total_frames if total_frames > 0 else '~'} ({progress:.1f}%), " \
                               f"处理FPS: {avg_fps:.1f}, 总计数: {stats['total_count']}"
                    
                    if self.license_enabled:
                        recognition_rate = (self.successful_license_recognitions / self.total_license_attempts * 100) if self.total_license_attempts > 0 else 0
                        status_msg += f", 车牌识别: {len(self.recognized_licenses)} 个 (成功率: {recognition_rate:.1f}%)"
                    
                    print(status_msg)
                    last_stats_print = time.time()

        except KeyboardInterrupt:
            print("\n用户中断处理")
        
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            if show_realtime:
                cv2.destroyAllWindows()

        print("="*80)
        print(f"处理完成! 总帧数: {frame_count}, 总检测数: {detection_count}")
        if self.license_enabled:
            print(f"车牌识别尝试: {self.total_license_attempts} 次")
            print(f"车牌识别成功: {self.successful_license_recognitions} 次")
            recognition_rate = (self.successful_license_recognitions / self.total_license_attempts * 100) if self.total_license_attempts > 0 else 0
            print(f"车牌识别成功率: {recognition_rate:.1f}%")
            print(f"唯一车牌总数: {len(self.recognized_licenses)}")
            self._print_license_summary()
        
        if save_stats:
            self._save_statistics_with_license(video_path, len(self.recognized_licenses))
        
        self._print_final_statistics()
        print(f"视频处理完成! 总用时: {(time.time() - processing_start_time):.2f} 秒.")
    
    def _print_license_summary(self):
        """打印车牌识别汇总"""
        if not self.license_output_log:
            return
            
        print("\n" + "="*80)
        print("🏷️  车牌识别详细汇总")
        print("="*80)
        
        for i, entry in enumerate(self.license_output_log, 1):
            print(f"{i:3d}. {entry['timestamp']} - 车辆ID: {entry['track_id']:03d} ({entry['vehicle_class']}) - 车牌: {entry['license_plate']} - 质量: {entry['license_quality']}")
        
        print("="*80)
        print(f"总计识别到 {len(self.license_output_log)} 个不同的车牌")
        
        # 按车辆类型统计
        type_stats = {}
        for entry in self.license_output_log:
            vehicle_type = entry['vehicle_class']
            type_stats[vehicle_type] = type_stats.get(vehicle_type, 0) + 1
        
        print("\n车牌识别按车辆类型统计:")
        for vtype, count in type_stats.items():
            print(f"  {vtype}: {count} 个车牌")
        
        print("="*80)
    
    def _draw_statistics_enhanced(self, frame: np.ndarray, current_time_sec: float, license_count: int):
        """统计信息显示（包含车牌识别信息）"""
        stats = self.traffic_counter.get_statistics()
        debug_info = stats['debug_info']
        
        # 背景区域 - 根据是否启用车牌识别调整高度
        overlay_h = 260 if self.license_enabled else 200
        overlay_x, overlay_y, overlay_w = 10, 10, 500
        sub_img = frame[overlay_y:overlay_y+overlay_h, overlay_x:overlay_x+overlay_w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 0)
        frame[overlay_y:overlay_y+overlay_h, overlay_x:overlay_x+overlay_w] = res

        text_color = (255, 255, 255)
        font_scale = 0.6
        line_height = 22
        current_y = overlay_y + line_height

        # 总车辆数
        cv2.putText(frame, f"Total Vehicles: {stats['total_count']}", 
                    (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        current_y += line_height + 5
        
        # 车牌识别统计 - 仅在启用时显示
        if self.license_enabled:
            recognition_rate = (self.successful_license_recognitions / self.total_license_attempts * 100) if self.total_license_attempts > 0 else 0
            cv2.putText(frame, f"License Plates: {len(self.recognized_licenses)} ({recognition_rate:.1f}%)", 
                        (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 0), 1, cv2.LINE_AA)
            current_y += line_height
            
            # 显示识别尝试次数
            cv2.putText(frame, f"Recognition Attempts: {self.total_license_attempts}", 
                        (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)
            current_y += line_height
        
        # 流量
        flow_rate = stats['current_flow_rate_per_minute']
        cv2.putText(frame, f"Flow Rate: {flow_rate['total']}/min", 
                    (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, text_color, 1, cv2.LINE_AA)
        current_y += line_height

        # 按类别计数
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
                        f"Crossing: {debug_info['crossing_attempts']}, "
                        f"Success: {debug_info['successful_counts']}", 
                    (overlay_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (255, 255, 0), 1, cv2.LINE_AA)
        current_y += line_height
        
        # 时间戳
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time_sec))
        cv2.putText(frame, time_str, (overlay_x + 10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
    def _save_statistics_with_license(self, video_path: str, license_count: int):
        """保存包含车牌识别的统计数据"""
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
                'count_direction': self.traffic_counter.count_direction,
                'license_recognition_enabled': self.license_enabled,
                'debug_mode': self.debug
            },
            'traffic_statistics': stats
        }
        
        if self.license_enabled:
            recognition_rate = (self.successful_license_recognitions / self.total_license_attempts * 100) if self.total_license_attempts > 0 else 0
            save_data['license_plate_statistics'] = {
                'total_unique_plates': license_count,
                'total_recognition_attempts': self.total_license_attempts,
                'successful_recognitions': self.successful_license_recognitions,
                'recognition_success_rate': f"{recognition_rate:.2f}%",
                'unique_plate_rate': f"{(license_count / stats['total_count'] * 100):.1f}%" if stats['total_count'] > 0 else "0%",
                'recognized_plates': self.license_output_log
            }
        
        suffix = "_with_license_analysis" if self.license_enabled else "_vehicle_analysis"
        output_filename = Path(video_path).stem + suffix + ".json"
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
        title = "最终统计结果（包含车牌识别）" if self.license_enabled else "最终统计结果"
        print(title)
        print("="*60)
        print(f"总车辆数: {stats['total_count']}")
        
        if self.license_enabled:
            recognition_rate = (self.successful_license_recognitions / self.total_license_attempts * 100) if self.total_license_attempts > 0 else 0
            print(f"\n车牌识别统计:")
            print(f"  识别尝试: {self.total_license_attempts} 次")
            print(f"  识别成功: {self.successful_license_recognitions} 次")
            print(f"  成功率: {recognition_rate:.1f}%")
            print(f"  唯一车牌: {len(self.recognized_licenses)} 个")
        
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
    parser = argparse.ArgumentParser(description='带车牌识别的视频车辆分析系统')
    parser.add_argument('--config', type=str, default='config.yaml', help='车辆检测配置文件路径')
    parser.add_argument('--vehicle-model', type=str, required=True, help='车辆检测模型文件路径')
    parser.add_argument('--license-yolo-config', type=str, required=True, help='车牌YOLO配置文件路径')
    parser.add_argument('--license-yolo-model', type=str, required=True, help='车牌YOLO模型文件路径')
    parser.add_argument('--license-crnn-model', type=str, required=True, help='车牌CRNN模型文件路径')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, help='输出视频路径')
    parser.add_argument('--no-save-stats', action='store_true', help='不保存统计结果')
    parser.add_argument('--no-display', action='store_true', help='不显示实时视频')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 验证文件存在
    files_to_check = [
        (args.config, "车辆检测配置文件"),
        (args.vehicle_model, "车辆检测模型文件"),
        (args.license_yolo_config, "车牌YOLO配置文件"),
        (args.license_yolo_model, "车牌YOLO模型文件"),
        (args.license_crnn_model, "车牌CRNN模型文件"),
        (args.video, "视频文件")
    ]
    
    for file_path, name in files_to_check:
        if not Path(file_path).exists():
            print(f"{name}不存在: {file_path}")
            return
    
    # 创建分析器实例
    analyzer = VideoVehicleAnalyzerWithLicensePlate(
        config_path=args.config,
        vehicle_model_path=args.vehicle_model,
        license_yolo_config=args.license_yolo_config,
        license_yolo_model=args.license_yolo_model,
        license_crnn_model=args.license_crnn_model,
        debug=args.debug
    )
    
    # 处理视频
    analyzer.process_video(
        video_path=args.video,
        output_path=args.output,
        save_stats=not args.no_save_stats,
        show_realtime=not args.no_display
    )

if __name__ == '__main__':
    main()