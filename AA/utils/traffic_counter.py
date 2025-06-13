import numpy as np
import cv2
from collections import defaultdict, deque
from typing import List, Tuple, Dict
import time
import math

class TrafficCounter:
    """车流量统计器"""
    
    def __init__(self, counting_lines: List[List[Tuple[int, int]]], count_direction="both"):
        self.counting_lines = []
        for line in counting_lines:
            try:
                p1 = (int(line[0][0]), int(line[0][1]))
                p2 = (int(line[1][0]), int(line[1][1]))
                self.counting_lines.append((p1, p2))
            except (TypeError, IndexError) as e:
                print(f"解析计数线时出错: {line}. 错误: {e}. 跳过此线.")
                continue

        self.count_direction = count_direction
        
        # 增加历史长度，提高检测精度
        self.track_history = defaultdict(lambda: deque(maxlen=50))  
        self.crossed_tracks = {}  # 字典：track_id -> {line_idx: last_cross_time}
        self.track_last_positions = {}  # track_id -> 最后已知位置
        
        # 统计数据
        self.count_data = {
            'total': 0,
            'by_class': defaultdict(int),
            'by_direction': defaultdict(int),
            'by_line': defaultdict(int),  # 新增：按线统计
            'by_time': defaultdict(int),
            'hourly_stats': defaultdict(lambda: defaultdict(int))
        }
        
        self.time_window = 60
        self.recent_counts = deque()
        
        # 调试信息
        self.debug_info = {
            'total_tracks': 0,
            'crossing_attempts': 0,
            'successful_counts': 0
        }
    
    def update(self, tracked_objects: List[dict], frame_time: float = None):
        if frame_time is None:
            frame_time = time.time()
        
        current_track_ids = set()
        self.debug_info['total_tracks'] = len(tracked_objects)
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            current_track_ids.add(track_id)
            bbox = obj['bbox']
            class_id = obj.get('class_id', -1)
            
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            center = (center_x, center_y)
            
            # 更新历史
            self.track_history[track_id].append({
                'center': center,
                'class_id': class_id,
                'timestamp': frame_time,
                'bbox': bbox
            })
            
            # 更新最后位置
            self.track_last_positions[track_id] = center
            
            # 检查所有计数线的穿过情况
            self._check_all_line_crossings(track_id, frame_time)
        
        self._cleanup_old_tracks(frame_time, current_track_ids)
        self._update_time_window_stats(frame_time)
    
    def _check_all_line_crossings(self, track_id: int, frame_time: float):
        """检查单个追踪对象与所有计数线的交叉情况"""
        history = self.track_history[track_id]
        if len(history) < 2:  # 降低要求到2个点
            return
        
        # 获取最近的位置点
        positions = [item['center'] for item in list(history)[-8:]]  # 增加到8个点
        class_id = history[-1]['class_id']
        
        for line_idx, line in enumerate(self.counting_lines):
            if self._is_in_cooldown(track_id, line_idx, frame_time):
                continue
            
            # 多点检测穿越
            if self._detect_line_crossing_multipoint(positions, line):
                direction = self._get_crossing_direction_improved(positions, line)
                
                if self._is_valid_direction(direction):
                    self._count_vehicle(track_id, class_id, direction, frame_time, line_idx)
                    self.debug_info['successful_counts'] += 1
                    # 添加调试输出
                    print(f"✓ 成功计数: Track {track_id}, 类别 {class_id}, 方向 {direction}, Line {line_idx}")
                
                self.debug_info['crossing_attempts'] += 1
    
    def _detect_line_crossing_multipoint(self, positions: List[Tuple[float, float]], 
                                       line: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """使用多点检测线段穿越，提高检测精度"""
        if len(positions) < 2:
            return False
        
        p3, p4 = line
        
        # 检查连续的线段对
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i + 1]
            
            if self._line_intersect_robust(p1, p2, p3, p4):
                return True
        
        # 额外检查：如果物体从线的一侧移动到另一侧
        if len(positions) >= 3:
            first_side = self._point_line_side(positions[0], line)
            last_side = self._point_line_side(positions[-1], line)
            
            # 如果起始和结束在线的不同侧，认为发生了穿越
            if first_side != 0 and last_side != 0 and first_side != last_side:
                return True
        
        return False
    
    def _line_intersect_robust(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                              p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """更稳健的线段相交检测"""
        def orientation(a, b, c):
            """计算三点的方向"""
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if abs(val) < 1e-6:  # 共线
                return 0
            return 1 if val > 0 else 2  # 顺时针或逆时针
        
        def on_segment(a, b, c):
            """检查点b是否在线段ac上"""
            return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                    min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # 一般情况
        if o1 != o2 and o3 != o4:
            return True
        
        # 特殊情况：点在线段上
        if (o1 == 0 and on_segment(p1, p3, p2)) or \
           (o2 == 0 and on_segment(p1, p4, p2)) or \
           (o3 == 0 and on_segment(p3, p1, p4)) or \
           (o4 == 0 and on_segment(p3, p2, p4)):
            return True
        
        return False
    
    def _point_line_side(self, point: Tuple[float, float], 
                        line: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        """判断点在直线的哪一侧 (-1, 0, 1)"""
        (x1, y1), (x2, y2) = line
        px, py = point
        
        # 使用叉积判断
        cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        
        if abs(cross_product) < 1e-6:
            return 0  # 在线上
        return 1 if cross_product > 0 else -1
    
    def _get_crossing_direction_improved(self, positions: List[Tuple[float, float]], 
                                       line: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
        """改进的方向判断"""
        if len(positions) < 2:
            return "unknown"
        
        # 使用首尾位置计算主要运动方向
        start_pos = positions[0]
        end_pos = positions[-1]
        
        (x1, y1), (x2, y2) = line
        
        # 线的方向向量
        line_vec = np.array([x2 - x1, y2 - y1], dtype=float)
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-6:
            return "unknown"
        
        # 标准化线向量
        line_vec_normalized = line_vec / line_length
        
        # 运动向量
        motion_vec = np.array([end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]], dtype=float)
        
        # 线的法向量（垂直向量）
        normal_vec = np.array([-line_vec_normalized[1], line_vec_normalized[0]])
        
        # 计算运动方向与法向量的点积
        dot_product = np.dot(motion_vec, normal_vec)
        
        # 根据点积判断方向
        return "up" if dot_product > 0 else "down"
    
    def _is_valid_direction(self, direction: str) -> bool:
        """验证方向是否符合计数要求"""
        if self.count_direction == "both":
            return direction in ["up", "down"]
        return direction == self.count_direction
    
    def _is_in_cooldown(self, track_id: int, line_idx: int, current_time: float) -> bool:
        """检查是否在冷却期内（防止重复计数）"""
        cooldown_time = 2.0  # 降低到2秒冷却期
        
        if track_id not in self.crossed_tracks:
            return False
        
        if line_idx not in self.crossed_tracks[track_id]:
            return False
        
        last_cross_time = self.crossed_tracks[track_id][line_idx]
        return (current_time - last_cross_time) < cooldown_time
    
    def _count_vehicle(self, track_id: int, class_id: int, direction: str, 
                      timestamp: float, line_idx: int):
        """计数车辆"""
        # 记录穿越时间
        if track_id not in self.crossed_tracks:
            self.crossed_tracks[track_id] = {}
        self.crossed_tracks[track_id][line_idx] = timestamp
        
        # 更新统计
        self.count_data['total'] += 1
        self.count_data['by_class'][class_id] += 1
        self.count_data['by_direction'][direction] += 1
        self.count_data['by_line'][line_idx] += 1
        
        # 时间统计
        current_time = time.gmtime(timestamp)
        hour = current_time.tm_hour
        minute_key = f"{hour:02d}:{current_time.tm_min:02d}"
        
        self.count_data['by_time'][minute_key] += 1
        self.count_data['hourly_stats'][hour][class_id] += 1
        
        self.recent_counts.append((timestamp, class_id, direction))
        
        print(f"✓ 车辆计数: Line {line_idx+1}, Track ID {track_id}, "
              f"类别 {class_id}, 方向 {direction}, 总计 {self.count_data['total']}")
    
    def _cleanup_old_tracks(self, current_time: float, current_track_ids: set):
        """清理旧的追踪数据"""
        timeout_history = 30.0
        timeout_crossed = 120.0  # 增加到2分钟
        
        # 清理历史
        expired_tracks = []
        for tid, hist in self.track_history.items():
            if tid not in current_track_ids and hist:
                if current_time - hist[-1]['timestamp'] > timeout_history:
                    expired_tracks.append(tid)
        
        for tid in expired_tracks:
            del self.track_history[tid]
            if tid in self.track_last_positions:
                del self.track_last_positions[tid]
        
        # 清理穿越记录
        expired_crossed = []
        for tid in list(self.crossed_tracks.keys()):
            if tid not in self.track_history:
                expired_crossed.append(tid)
            else:
                # 清理过期的线穿越记录
                for line_idx in list(self.crossed_tracks[tid].keys()):
                    if current_time - self.crossed_tracks[tid][line_idx] > timeout_crossed:
                        del self.crossed_tracks[tid][line_idx]
                
                # 如果该track的所有线记录都被清理了，删除整个track记录
                if not self.crossed_tracks[tid]:
                    expired_crossed.append(tid)
        
        for tid in expired_crossed:
            if tid in self.crossed_tracks:
                del self.crossed_tracks[tid]
    
    def _update_time_window_stats(self, current_time: float):
        """更新时间窗口统计"""
        cutoff_time = current_time - self.time_window
        while self.recent_counts and self.recent_counts[0][0] < cutoff_time:
            self.recent_counts.popleft()

    def get_current_flow_rate(self) -> Dict:
        """获取当前流量"""
        if not self.recent_counts or self.time_window == 0:
            return {'total': 0, 'by_class': defaultdict(int)}
        
        multiplier = 60.0 / self.time_window
        flow_rate_total = int(len(self.recent_counts) * multiplier)
        
        flow_by_class = defaultdict(int)
        for _, class_id, _ in self.recent_counts:
            flow_by_class[class_id] += 1
        
        for class_id in flow_by_class:
            flow_by_class[class_id] = int(flow_by_class[class_id] * multiplier)
            
        return {
            'total': flow_rate_total,
            'by_class': dict(flow_by_class)
        }

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_count': self.count_data['total'],
            'count_by_class': dict(self.count_data['by_class']),
            'count_by_direction': dict(self.count_data['by_direction']),
            'count_by_line': dict(self.count_data['by_line']),
            'count_by_time_minute': dict(self.count_data['by_time']),
            'hourly_stats': {hour: dict(counts) for hour, counts in self.count_data['hourly_stats'].items()},
            'current_flow_rate_per_minute': self.get_current_flow_rate(),
            'debug_info': self.debug_info.copy()
        }

    def draw_counting_lines(self, frame: np.ndarray) -> np.ndarray:
        """绘制计数线和调试信息"""
        for i, line in enumerate(self.counting_lines):
            p1, p2 = line
            
            # 绘制计数线
            cv2.line(frame, p1, p2, (0, 255, 255), 3)
            
            # 添加线标签和统计
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            
            line_count = self.count_data['by_line'].get(i, 0)
            label = f"L{i+1}: {line_count}"
            
            cv2.putText(frame, label, (mid_x + 10, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 绘制方向箭头
            self._draw_direction_arrow(frame, p1, p2)
        
        return frame
    
    def _draw_direction_arrow(self, frame: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]):
        """绘制方向箭头"""
        # 计算线的中点和方向
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        # 线向量
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # 标准化
            dx /= length
            dy /= length
            
            # 法向量（垂直向量）
            arrow_length = 20
            arrow_x = int(mid_x + dy * arrow_length)
            arrow_y = int(mid_y - dx * arrow_length)
            
            # 绘制箭头（表示"up"方向）
            cv2.arrowedLine(frame, (mid_x, mid_y), (arrow_x, arrow_y), 
                           (255, 255, 0), 2, tipLength=0.3)