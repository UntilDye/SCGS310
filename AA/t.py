import cv2
import matplotlib.pyplot as plt

def show_bbox_with_class_id(image_path, label_path):
    # 读取图像并保留原始副本
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 读取标注文件
    with open(label_path) as f:
        annotations = [list(map(float, line.strip().split())) for line in f]
    
    height, width = image.shape[:2]
    for ann in annotations:
        if len(ann) < 5: continue
        class_id, x_center, y_center, w, h = ann
        
        # 计算边界框坐标
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)
        
        # 绘制边界框（青色）
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # 计算文本位置
        class_id = str(int(class_id))
        (tw, th), _ = cv2.getTextSize(class_id, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # 自适应文本位置
        text_x = x1 + 5 if x1 + tw < width else x2 - tw - 5
        text_y = y1 + th + 5 if y1 + th < height else y2 - 5
        
        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (text_x-2, text_y-th-2), 
                     (text_x+tw+2, text_y+2), (0, 0, 0), -1)
        alpha = 0.8  # 透明度系数
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # 添加黄色文本
        cv2.putText(image, class_id, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # 转换颜色空间并显示
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# 使用示例
show_bbox_with_class_id(
    r'C:\codee\A\data\aic_hcmc2020\images\val\cam_01_000011.jpg',
    r'C:\codee\A\data\aic_hcmc2020\labels\val\cam_01_000011.txt'
)
