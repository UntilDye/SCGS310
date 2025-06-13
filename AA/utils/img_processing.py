import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_single_image_for_crnn(input_image_path, imgH=32, imgW=160, show=False):
    input_path = Path(input_image_path)

    if not input_path.is_file():
        print(f"输入文件不存在: {input_path}")
        return

    filename = input_path.name
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        print(f"跳过非图片文件: {filename}")
        return

    try:
        with open(input_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            print(f"警告: 无法解码图片 {filename}，可能文件已损坏或格式不支持。")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        scale = imgH / h
        new_w = int(w * scale)
        if new_w > imgW:
            new_w = imgW
            scale = imgW / w
            img_resized_h = int(h * scale)
            img = cv2.resize(img, (new_w, img_resized_h))
        else:
            img = cv2.resize(img, (new_w, imgH))

        padded = np.ones((imgH, imgW), dtype=np.uint8) * 255
        h_resized, w_resized = img.shape
        start_y = (imgH - h_resized) // 2
        padded[start_y:start_y + h_resized, :w_resized] = img

        if show:
            plt.imshow(padded, cmap='gray')
            plt.title(f"预处理后图片: {filename}")
            plt.axis('off')
            plt.show()

    except Exception as e:
        print(f"处理图片 {filename} 时发生错误: {str(e)}")
    return padded
if __name__ == '__main__':

    #img = preprocess_single_image_for_crnn(r"E:\AISYSTEM\YOLO\dataset\val\cropped_images\沪AD07225_251&427_491&538.jpg")
    pass
