#deep_sort_tracker.py
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    """车辆追踪器，基于DeepSORT"""
    
    def __init__(self, class_names: list, max_age=50, n_init=1, max_iou_distance=0.8, debug=False):
        self.debug = debug
        self.class_names = class_names
        
        self.tracker = DeepSort(
            max_age=max_age,        # 追踪保持时间
            n_init=n_init,          # 初始化要求
            max_iou_distance=max_iou_distance,  # IoU阈值
            max_cosine_distance=0.3,  # 特征匹配
            nn_budget=150,          # 特征预算
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=torch.cuda.is_available(),
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )

    def _extract_confirmed_tracks(self, tracks):
        """提取已确认的追踪结果"""
        tracked_objects = []
        for track in tracks:
            # 确认条件
                
            # 允许最近几帧内的追踪
            if track.time_since_update > 3:  # 容忍度
                continue
            
            track_id = track.track_id
            
            try:
                ltrb_float = track.to_ltrb()
                x1, y1, x2, y2 = ltrb_float
            except Exception as e:
                if self.debug:
                    print(f"获取track边界框失败: {e}")
                try:
                    bbox = track.to_tlwh()
                    x1, y1, w, h = bbox
                    x2, y2 = x1 + w, y1 + h
                except Exception as e2:
                    if self.debug:
                        print(f"备用方法也失败: {e2}")
                    continue

            class_name_from_track = track.get_det_class()
            cls_id = self._get_class_id_from_name(class_name_from_track)
            
            # 添加追踪质量信息
            tracked_objects.append({
                'track_id': track_id,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class_id': int(cls_id),
                'confidence': getattr(track, 'confidence', 0.5),  
                'hits': getattr(track, 'hits', 1),
                'time_since_update': track.time_since_update
            })
        
        if self.debug and tracked_objects:
            print(f"返回追踪对象数量: {len(tracked_objects)}")
        
        return tracked_objects  
    def update(self, detections, frame):
        if frame is None:
            if self.debug:
                print("VehicleTracker: Frame is None, cannot update tracker.")
            self.tracker.update_tracks([], frame=np.zeros((100,100,3), dtype=np.uint8))
            return []

        if not detections:
            tracks = self.tracker.update_tracks([], frame=frame)
            # 即使没有新检测，也要返回现有的稳定追踪
            return self._extract_confirmed_tracks(tracks)
        
        det_list_for_deepsort = []
        for det in detections:
            try:
                x1, y1, x2, y2, conf, cls_id = det
                
                x1, y1, x2, y2, conf = float(x1), float(y1), float(x2), float(y2), float(conf)
                cls_id = int(cls_id)

                if not (x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1):
                    continue
                
                frame_h, frame_w = frame.shape[:2]
                x1 = max(0.0, min(x1, frame_w - 1))
                y1 = max(0.0, min(y1, frame_h - 1))
                x2 = max(x1 + 1, min(x2, frame_w))
                y2 = max(y1 + 1, min(y2, frame_h))

                if x2 <= x1 or y2 <= y1:
                    continue

                w = x2 - x1
                h = y2 - y1
                
                if 0 <= cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = "unknown"
                
                det_list_for_deepsort.append(
                    ([x1, y1, w, h], conf, class_name)
                )

            except (ValueError, TypeError, IndexError) as e:
                if self.debug:
                    print(f"Detection processing error: {det}, Error: {e}")
                continue
        
        try:
            tracks = self.tracker.update_tracks(det_list_for_deepsort, frame=frame)
            return self._extract_confirmed_tracks(tracks)
        except Exception as e:
            if self.debug:
                print(f"DeepSORT 更新失败: {e}")
            return []

    
    def _get_class_id_from_name(self, class_name: str) -> int:
        """辅助方法：从class_name_str获取class_id"""
        try:
            return self.class_names.index(class_name)
        except ValueError:
            if self.debug and class_name != "unknown": # 避免为故意设为unknown的情况重复打印
                 print(f"Warning: Class name '{class_name}' not found in self.class_names.") # 警告：类别名称未在self.class_names中找到
            return -1 # 或一个默认的"unknown"类别ID