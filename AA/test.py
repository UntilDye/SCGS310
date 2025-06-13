#test.py
import torch
import cv2
import numpy as np
from models.yolo_vehicle import VehicleYOLO 
import yaml

def analyze_checkpoint(model_path):
    """分析检查点中的实际配置"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("检查点分析:")
    print(f"训练轮次: {checkpoint.get('epoch', 'Unknown')}")
    print(f"损失: {checkpoint.get('loss', 'Unknown')}")
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"配置信息: {config}")
        return config
    
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    num_classes = 4 # 默认值
    if state_dict: # 确保 state_dict 不是 None
        head_weights_keys = [k for k in state_dict.keys() if k.endswith('.weight') and ('detect.' in k or 'head.' in k or '.m.' in k)]
        if head_weights_keys:
            specific_head_weights = [k for k in state_dict.keys() if 'head' in k and 'weight' in k and k.endswith('.3.weight')]
            if specific_head_weights:
                output_channels = state_dict[specific_head_weights[0]].shape[0]
                print(f"检测头特定层输出通道数: {output_channels}")
                if output_channels == 27: 
                    num_classes = 4
                    print("推断类别数 (逻辑1): 4")
                elif output_channels == 39: 
                    num_classes = 8
                    print("推断类别数 (逻辑1): 8")
                else:
                    potential_num_classes = output_channels / 3 - 5
                    if potential_num_classes > 0 and potential_num_classes.is_integer():
                        num_classes = int(potential_num_classes)
                        print(f"推断类别数 (通用逻辑, 假设3个锚点): {num_classes}")
                    else:
                        print(f"无法从 {output_channels} 通道精确推断类别数，使用默认值: 4")
                        num_classes = 4
            else:
                print("未找到特定检测头权重来推断类别数，使用默认值: 4")
        else:
            print("无法找到检测头权重来推断类别数，使用默认值: 4")
    else:
        print("Checkpoint中未找到模型权重 (state_dict)，使用默认类别数: 4")

    inferred_config = {
        'model': {
            'num_classes': num_classes,
            'input_size': [416, 416],  
            'class_names': ['motorbike', 'car', 'bus', 'truck'] if num_classes == 4 else \
                           (['motorbike', 'car', 'bus', 'truck', 'bicycle', 'person', 'traffic_light', 'traffic_sign'] if num_classes == 8 else \
                            [f'class_{i}' for i in range(num_classes)]) 
        }
    }
    print(f"最终使用的类别数: {num_classes}, 类别名: {inferred_config['model']['class_names']}")
    return inferred_config

def create_model_with_correct_config(model_path):
    """使用正确配置创建模型"""
    config = analyze_checkpoint(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else \
                     ('state_dict' if 'state_dict' in checkpoint else None)

    if not state_dict_key:
        raise KeyError("Checkpoint does not contain 'model_state_dict' or 'state_dict'.")

    model = VehicleYOLO(
        num_classes=config['model']['num_classes'],
        input_shape=tuple(config['model']['input_size'])
    )
    
    model.load_state_dict(checkpoint[state_dict_key])
    model.to(device)
    model.eval()
    
    print(f"模型创建成功，类别数: {config['model']['num_classes']}")
    return model, config, device

def improved_post_process(detections, original_size, input_size, conf_threshold=0.3, nms_threshold=0.4):
    """
    改进后的后处理函数
    """
    if not detections or len(detections) == 0:
        return []

    original_width, original_height = original_size 
    input_width, input_height = input_size          
    scale_x = original_width / input_width          
    scale_y = original_height / input_height        

    detection = detections[0] if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict) else \
                (detections if isinstance(detections, dict) else None)
    
    if not detection:
        print("Post-process: Detections format not recognized or empty.")
        return []

    boxes     = detection.get('boxes', [])
    scores    = detection.get('scores', [])
    class_ids = detection.get('class_ids', [])

    if not (hasattr(boxes, '__len__') and len(boxes) == len(scores) == len(class_ids)):
        print("Post-process: Mismatch in length of boxes, scores, or class_ids.")
        return []

    valid_detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        if isinstance(box, torch.Tensor): box = box.cpu().numpy()
        if isinstance(score, torch.Tensor): score = score.cpu().item()
        if isinstance(class_id, torch.Tensor): class_id = class_id.cpu().item()

        if score < conf_threshold: continue
        if not (hasattr(box, '__len__') and len(box) == 4):
            print(f"Post-process: Invalid box format: {box}")
            continue

        max_coord = float(np.max(box))
        if 0.0 < max_coord <= 1.0 and all(b >= 0 for b in box):
            cx_norm, cy_norm, w_norm, h_norm = box
            center_x_orig = cx_norm * original_width
            center_y_orig = cy_norm * original_height
            w_orig        = w_norm  * original_width
            h_orig        = h_norm  * original_height
        elif max_coord > 1.0:
            cx_input, cy_input, w_input, h_input = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            center_x_orig = cx_input * scale_x
            center_y_orig = cy_input * scale_y
            w_orig        = w_input  * scale_x
            h_orig        = h_input  * scale_y
        else: 
            print(f"Post-process: Ambiguous box values, skipping: {box}")
            continue
            
        x1 = center_x_orig - w_orig / 2.0
        y1 = center_y_orig - h_orig / 2.0
        x2 = center_x_orig + w_orig / 2.0
        y2 = center_y_orig + h_orig / 2.0

        x1_c = max(0, min(int(round(x1)), original_width  - 1))
        y1_c = max(0, min(int(round(y1)), original_height - 1))
        x2_c = max(x1_c + 1, min(int(round(x2)), original_width )) 
        y2_c = max(y1_c + 1, min(int(round(y2)), original_height))
        
        if x1_c >= x2_c or y1_c >= y2_c: continue

        width_final  = x2_c - x1_c
        height_final = y2_c - y1_c

        min_dim_filter = min(original_width, original_height) * 0.01  
        max_dim_filter = min(original_width, original_height) * 0.8  
        if not (width_final >= min_dim_filter and height_final >= min_dim_filter and
                width_final <= max_dim_filter and height_final <= max_dim_filter):
            continue

        area = width_final * height_final
        image_area = original_width * original_height
        if image_area == 0: continue 
        area_ratio = area / image_area
        if not (0.00005 <= area_ratio <= 0.5): 
            continue

        valid_detections.append({
            'bbox':  [x1_c, y1_c, x2_c, y2_c],
            'score': float(score),
            'class_id': int(class_id),
            'area': area
        })

    if len(valid_detections) == 0: return []

    class_groups = {}
    for det in valid_detections:
        cid = det['class_id']
        class_groups.setdefault(cid, []).append(det)

    final_results = []
    for cid, dets in class_groups.items():
        if not dets: continue
        boxes_tensor = torch.tensor([d['bbox'] for d in dets], dtype=torch.float32)
        scores_tensor = torch.tensor([d['score'] for d in dets], dtype=torch.float32)
        
        keep_inds = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, nms_threshold)
        for idx in keep_inds:
            final_results.append(dets[idx.item()]) 

    final_results.sort(key=lambda x: x['score'], reverse=True)
    return final_results

def size_consistency_filter(detections, size_threshold_ratio=0.5):
    """基于尺寸一致性过滤检测结果"""
    if len(detections) <= 1: return detections
    
    class_groups = {}
    for det in detections:
        class_id = det['class_id']
        class_groups.setdefault(class_id, []).append(det)
    
    filtered_results = []
    for class_id, class_detections in class_groups.items():
        if len(class_detections) <= 1:
            filtered_results.extend(class_detections)
            continue
        
        areas = [det['area'] for det in class_detections]
        if not areas: continue
        median_area = np.median(areas)
        
        if median_area == 0: 
             filtered_results.extend(class_detections)
             continue

        for det in class_detections:
            if det['area'] == 0 and median_area == 0: 
                 filtered_results.append(det)
                 continue
            if median_area == 0: continue

            area_ratio = det['area'] / median_area
            if 0.2 <= area_ratio <= 5.0: 
                filtered_results.append(det)
    
    return filtered_results

def enhanced_detect_image(image_path, model_path, output_path=None, conf_threshold=0.4):
    """改进版检测函数"""
    print("=" * 60)
    print("改进版车辆检测 v2.2 - 类别名黑色, 类别ID红色") # 版本更新
    print("=" * 60)
    
    model, config, device = create_model_with_correct_config(model_path)
    class_names = config['model']['class_names']
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    print(f"图像尺寸: {image.shape}")
    original_height, original_width = image.shape[:2]
    input_size_cfg = config['model']['input_size']
    # 目标尺寸的索引，确保宽度和高度对应正确
    target_width, target_height = int(input_size_cfg[0]), int(input_size_cfg[1]) 
    print(f"目标尺寸: {target_width}x{target_height}")
    
    image_resized = cv2.resize(image, (target_width, target_height))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_tensor_chw = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    image_tensor_norm = (image_tensor_chw.to(device) - mean) / std
    image_tensor_batch = image_tensor_norm.unsqueeze(0)
    
    print("开始推理...")
    detections_raw = None
    with torch.no_grad():
        try:
            if hasattr(model, 'predict') and callable(model.predict):
                detections_raw = model.predict(
                    image_tensor_batch, 
                    conf_threshold=conf_threshold, 
                    iou_threshold=0.45, 
                    device=device
                )
                print("使用model.predict()方法进行推理")
            else: 
                outputs = model(image_tensor_batch)
                if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
                    detections_raw = outputs
                else: 
                    print("模型没有 predict 方法，且直接输出格式未知，请适配此部分代码。")
                    detections_raw = [] 
                print("使用模型直接调用进行推理")

        except Exception as e:
            print(f"推理错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    if detections_raw and (isinstance(detections_raw, list) and len(detections_raw) > 0 and detections_raw[0].get('boxes') is not None or isinstance(detections_raw, dict) and detections_raw.get('boxes') is not None):
        processed_detections = improved_post_process(
            detections_raw,
            (original_width, original_height),
            (target_width, target_height),
            conf_threshold=conf_threshold, 
            nms_threshold=0.4 # 尝试略微降低NMS阈值
        )
        
        # 将尺寸一致性过滤这行代码注释掉，或者移除，先观察效果
        # final_detections = size_consistency_filter(processed_detections)
        final_detections = processed_detections # 直接使用NMS后的结果

        num_raw_boxes = 0
        if isinstance(detections_raw, list) and len(detections_raw) > 0:
            num_raw_boxes = len(detections_raw[0].get('boxes', []))
        elif isinstance(detections_raw, dict):
            num_raw_boxes = len(detections_raw.get('boxes', []))

        print(f"原始检测框数: {num_raw_boxes} -> NMS后: {len(processed_detections)} -> 尺寸一致性过滤后: {len(final_detections)}")
        
        result_image = draw_enhanced_boxes(image, final_detections, class_names)
    else:
        print("推理后没有有效的原始检测结果或格式不正确")
        result_image = image
    
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"结果保存到: {output_path}")
    
    # 移除或注释掉显示结果的函数调用，因为在某些环境下可能会导致错误
    # display_result(result_image)
    return result_image

def draw_enhanced_boxes(image, detections, class_names):
    """改进的绘制函数，类别名黑色, 类别ID红色, 分数白色"""
    result_image = image.copy()
    
    colors = [ # BGR format
        (0, 255, 0),   # 绿色 
        (255, 0, 0),   # 蓝色 
        (0, 0, 255),   # 红色 
        (255, 255, 0), # 青色 
        (255, 0, 255), # 品红
        (0, 255, 255), # 黄色
        (128, 0, 128), # 紫色
        (255, 165, 0)  # 橙色
    ]
    
    detections_sorted = sorted(detections, key=lambda x: x.get('area', 0), reverse=True)
    
    for i, det in enumerate(detections_sorted):
        x1, y1, x2, y2 = map(int, det['bbox']) 
        score = det['score']
        class_id = det['class_id']
        area = det.get('area', (x2-x1)*(y2-y1)) 
        
        width = x2 - x1
        height = y2 - y1
        
        det_class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f'cls_{class_id}' # Renamed to avoid conflict
        bg_color = colors[class_id % len(colors)] 
        
        thickness = max(1, min(6, int(np.sqrt(area) / 50)))
        font_scale = max(0.4, min(1.0, np.sqrt(area) / 200)) 
        text_thickness = max(1, int(font_scale * 1.5)) 

        cv2.rectangle(result_image, (x1, y1), (x2, y2), bg_color, thickness)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        # --- 颜色定义修改 ---
        text_color_classname = (0, 0, 0)     # 黑色字体 (BGR) for class name
        text_color_id = (0, 0, 255)          # 红色字体 (BGR) for class ID
        text_color_score = (255, 255, 255)   # 白色字体 (BGR) for score

        text_part1_name = f'{det_class_name} '
        text_part2_id = f'(ID:{class_id}) ' 
        text_part3_score = f'{score:.2f}'

        (tw1, th1), bl1 = cv2.getTextSize(text_part1_name, font, font_scale, text_thickness)
        (tw2, th2), bl2 = cv2.getTextSize(text_part2_id, font, font_scale, text_thickness)
        (tw3, th3), bl3 = cv2.getTextSize(text_part3_score, font, font_scale, text_thickness)

        total_text_width = tw1 + tw2 + tw3
        text_height = th1 
        baseline = bl1

        label_bg_y1 = y1 - text_height - baseline - 8
        label_bg_y2 = y1
        if label_bg_y1 < 0: 
            label_bg_y1 = y2 - text_height - baseline - 8
            label_bg_y2 = y2
            if label_bg_y1 < y1 + height / 2 : 
                 label_bg_y1 = y1 + baseline + 4 
                 label_bg_y2 = y1 + text_height + baseline + 8

        cv2.rectangle(
            result_image,
            (x1, label_bg_y1),
            (x1 + total_text_width + 8, label_bg_y2), 
            bg_color,
            -1
        )
        
        text_y = label_bg_y2 - baseline - 4 

        current_x = x1 + 4 
        # --- 绘制时应用新颜色 ---
        cv2.putText(result_image, text_part1_name, (current_x, text_y), font, font_scale, text_color_classname, text_thickness, cv2.LINE_AA) # 类别名用黑色
        
        current_x += tw1
        cv2.putText(result_image, text_part2_id, (current_x, text_y), font, font_scale, text_color_id, text_thickness, cv2.LINE_AA) # 类别ID用红色

        current_x += tw2
        cv2.putText(result_image, text_part3_score, (current_x, text_y), font, font_scale, text_color_score, text_thickness, cv2.LINE_AA) # 分数用白色
        
        print(f"绘制检测框 {i+1}: {det_class_name} (ID:{class_id}) - Score: {score:.3f} - BBox: [{x1}, {y1}, {x2}, {y2}] - Size: {width}x{height} - Area: {area:.0f}")
    
    return result_image

def display_result(image):
    """显示结果图像"""
    height, width = image.shape[:2]
    
    max_display_width = 1200
    max_display_height = 800 
    
    scale_w = 1.0
    if width > max_display_width: scale_w = max_display_width / width
    scale_h = 1.0
    if height > max_display_height: scale_h = max_display_height / height
    scale = min(scale_w, scale_h) 

    if scale < 1.0:
        display_width = int(width * scale)
        display_height = int(height * scale)
        display_image = cv2.resize(image, (display_width, display_height))
    else:
        display_image = image
    
    cv2.imshow('Enhanced Detection Result', display_image)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        from models.yolo_vehicle import VehicleYOLO
    except ImportError:
        print("错误：无法从 models.yolo_vehicle 导入 VehicleYOLO。")
        print("请确保 'models' 文件夹在PYTHONPATH中，或者 'models/yolo_vehicle.py' 文件存在且 VehicleYOLO 类已定义。")
        class VehicleYOLO(torch.nn.Module): # Placeholder
            def __init__(self, num_classes, input_shape):
                super().__init__()
                print(f"警告: 使用了占位符 VehicleYOLO (num_classes={num_classes}, input_shape={input_shape})")
                self.fc = torch.nn.Linear(10, num_classes) 
            def forward(self, x):
                print("警告: 占位符 VehicleYOLO forward() 被调用")
                return [{'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 'class_ids': torch.empty(0)}]
            def predict(self, x, conf_threshold, iou_threshold, device): 
                print("警告: 占位符 VehicleYOLO predict() 被调用")
                return self.forward(x)

    model_path = r'A\experiments\model\best_vehicle_model.pth'
    image_path = r"data/aic_hcmc2020/images/val/cam_01_000011.jpg" 
    output_path = "enhanced_result_name_black_id_red.jpg" # 更新输出文件名
    
    print("分析模型结构...")
    try:
        analyze_checkpoint(model_path)
    except FileNotFoundError:
        print(f"错误: 模型文件 {model_path} 未找到。请检查路径。")
        exit()
    except Exception as e:
        print(f"分析模型时发生错误: {e}")

    print("\n" + "="*60)
    
    try:
        enhanced_detect_image(image_path, model_path, output_path, conf_threshold=0.1) # 置信度阈值
    except FileNotFoundError:
        print(f"错误: 图片文件 {image_path} 或模型文件 {model_path} 未找到。请检查路径。")
    except NameError as e:
        if 'VehicleYOLO' in str(e): print("错误：VehicleYOLO 类未定义。请确保相关模型代码可用。")
        else: print(f"发生名称错误：{e}")
    except Exception as e:
        print(f"检测过程中发生未知错误: {e}")
        import traceback
        traceback.print_exc()