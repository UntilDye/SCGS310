#video_analysis_test.py
import cv2
import torch
import numpy as np
from pathlib import Path

def test_single_frame_from_video(video_path, model_path, frame_number=10):
    """从视频中提取单帧进行测试"""
    
    # 导入必要的模块
    from video_vehicle_analysis import VideoVehicleAnalyzer
    
    # 提取视频帧
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"无法读取第 {frame_number} 帧")
        return
    
    # 保存帧用于对比
    frame_path = f"video_frame_{frame_number}.jpg"
    cv2.imwrite(frame_path, frame)
    print(f"提取的帧保存为: {frame_path}")
    
    # 使用分析器检测
    analyzer = VideoVehicleAnalyzer("config.yaml", model_path, debug=True)
    
    print("使用视频分析器检测...")
    detections = analyzer.detect_vehicles(frame)
    print(f"检测结果数量: {len(detections)}")
    
    # 绘制结果
    if detections:
        result_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{analyzer.class_names[cls_id]}: {conf:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(f"video_detection_result_{frame_number}.jpg", result_frame)
        print(f"检测结果保存为: video_detection_result_{frame_number}.jpg")
    
    return detections

if __name__ == "__main__":
    video_path = "data/aic_hcmc2020/video/aic_hcmc2020_video.mp4"
    model_path = "experiments/model/best_vehicle_model.pth"
    
    # 测试多个帧
    for frame_num in [10, 50, 100, 200]:
        print(f"\n{'='*50}")
        print(f"测试第 {frame_num} 帧")
        print(f"{'='*50}")
        test_single_frame_from_video(video_path, model_path, frame_num)
        """ # 带实时显示
        python video_vehicle_analysis.py --config config.yaml --model experiments\model\best_vehicle_model.pth --video data\aic_hcmc2020\video\test.mp4 --output output.mp4
        
        python detector_counter_liscense_vechicle.py --config config.yaml --vehicle-model experiments\model\best_vehicle_model.pth --license-yolo-config utils\config.yaml --license-yolo-model "experiments\model\MobileNet_Yolo3 epoch_100.pt" --license-crnn-model experiments\model\Crnn_model—OCR.pth --video data\aic_hcmc2020\video\test.mp4 --output output_video.mp4 
    --debug              """