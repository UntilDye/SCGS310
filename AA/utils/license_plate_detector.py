# utils/license_plate_detector.py
import cv2
import numpy as np
import easyocr
from typing import List, Tuple, Optional

class LicensePlateDetector:
    """车牌检测和识别"""
    
    def __init__(self, use_gpu=True):
        # 初始化OCR读取器
        self.reader = easyocr.Reader(['en', 'ch_sim'], gpu=use_gpu)
        
        # 车牌检测的Haar级联分类器（可选）
        self.plate_cascade = None
        try:
            # 如果有预训练的车牌检测模型
            self.plate_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_licence_plate_rus_16stages.xml')
        except:
            print("警告: 未找到车牌检测级联分类器，将使用基于颜色的检测")
    
    def detect_and_recognize_plates(self, image: np.ndarray, vehicle_bbox: List[int]) -> List[dict]:
        """
        在车辆区域内检测和识别车牌
        
        Args:
            image: 原始图像
            vehicle_bbox: 车辆边界框 [x1, y1, x2, y2]
            
        Returns:
            List[dict]: 车牌信息列表，每个包含 {'bbox': [x1,y1,x2,y2], 'text': str, 'confidence': float}
        """
        x1, y1, x2, y2 = vehicle_bbox
        
        # 扩展车辆区域以确保包含车牌
        h, w = image.shape[:2]
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # 提取车辆区域
        vehicle_roi = image[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return []
        
        plates = []
        
        # 方法1: 使用Haar级联检测车牌
        if self.plate_cascade is not None:
            plates.extend(self._detect_plates_haar(vehicle_roi, (x1, y1)))
        
        # 方法2: 基于颜色和形状的车牌检测
        plates.extend(self._detect_plates_color_shape(vehicle_roi, (x1, y1)))
        
        # 去重并识别文字
        unique_plates = self._remove_duplicate_plates(plates)
        recognized_plates = []
        
        for plate in unique_plates:
            text, confidence = self._recognize_plate_text(image, plate['bbox'])
            if text and confidence > 0.3:  # 设置置信度阈值
                recognized_plates.append({
                    'bbox': plate['bbox'],
                    'text': text,
                    'confidence': confidence
                })
        
        return recognized_plates
    
    def _detect_plates_haar(self, roi: np.ndarray, offset: Tuple[int, int]) -> List[dict]:
        """使用Haar级联检测车牌"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 15)
        )
        
        result = []
        for (x, y, w, h) in plates:
            # 转换为全局坐标
            global_x = x + offset[0]
            global_y = y + offset[1]
            result.append({
                'bbox': [global_x, global_y, global_x + w, global_y + h],
                'method': 'haar'
            })
        
        return result
    
    def _detect_plates_color_shape(self, roi: np.ndarray, offset: Tuple[int, int]) -> List[dict]:
        """基于颜色和形状检测车牌"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 定义蓝色车牌的HSV范围（中国车牌）
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 定义白色/黄色车牌的HSV范围
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        lower_yellow = np.array([15, 150, 150])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 合并掩码
        combined_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(white_mask, yellow_mask))
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            # 计算轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 车牌比例检查（宽高比通常在2-5之间）
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if (2.0 <= aspect_ratio <= 6.0 and 
                area > 500 and area < 50000 and  # 面积限制
                w > 50 and h > 15):  # 最小尺寸限制
                
                # 转换为全局坐标
                global_x = x + offset[0]
                global_y = y + offset[1]
                plates.append({
                    'bbox': [global_x, global_y, global_x + w, global_y + h],
                    'method': 'color_shape'
                })
        
        return plates
    
    def _remove_duplicate_plates(self, plates: List[dict]) -> List[dict]:
        """移除重复的车牌检测"""
        if len(plates) <= 1:
            return plates
        
        # 简单的IoU去重
        unique_plates = []
        for plate in plates:
            is_duplicate = False
            for unique_plate in unique_plates:
                iou = self._calculate_iou(plate['bbox'], unique_plate['bbox'])
                if iou > 0.5:  # 如果IoU > 0.5则认为是重复
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_plates.append(plate)
        
        return unique_plates
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _recognize_plate_text(self, image: np.ndarray, bbox: List[int]) -> Tuple[str, float]:
        """识别车牌文字"""
        x1, y1, x2, y2 = bbox
        plate_roi = image[y1:y2, x1:x2]
        
        if plate_roi.size == 0:
            return "", 0.0
        
        # 预处理车牌图像
        plate_roi = self._preprocess_plate_image(plate_roi)
        
        try:
            # 使用EasyOCR识别文字
            results = self.reader.readtext(plate_roi)
            
            if not results:
                return "", 0.0
            
            # 找到置信度最高的结果
            best_result = max(results, key=lambda x: x[2])  # x[2]是置信度
            text = best_result[1].upper().replace(' ', '')  # 移除空格并转大写
            confidence = best_result[2]
            
            # 简单的车牌格式验证
            if self._validate_plate_text(text):
                return text, confidence
            else:
                return "", 0.0
                
        except Exception as e:
            print(f"车牌识别错误: {e}")
            return "", 0.0
    
    def _preprocess_plate_image(self, plate_image: np.ndarray) -> np.ndarray:
        """预处理车牌图像以提高OCR准确性"""
        # 调整大小
        height, width = plate_image.shape[:2]
        if width < 200:  # 如果太小则放大
            scale = 200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            plate_image = cv2.resize(plate_image, (new_width, new_height))
        
        # 转换为灰度
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def _validate_plate_text(self, text: str) -> bool:
        """验证车牌文字格式"""
        if not text or len(text) < 6:
            return False
        
        # 移除所有非字母数字字符
        clean_text = ''.join(c for c in text if c.isalnum())
        
        # 基本长度检查（中国车牌通常6-8位）
        if len(clean_text) < 6 or len(clean_text) > 8:
            return False
        
        # 检查是否包含合理的字符组合
        has_letter = any(c.isalpha() for c in clean_text)
        has_digit = any(c.isdigit() for c in clean_text)
        
        return has_letter and has_digit