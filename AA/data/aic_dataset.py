# data/aic_dataset.py
import os
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AICDataset(Dataset):
    """AIC-HCMC-2020数据集处理类 (仅支持 YOLO TXT 格式标注)"""
    def __init__(self, images_dir, labels_dir, img_size=640, class_mapping=None, is_training=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.is_training = is_training

        # AIC数据集的类别映射
        self.class_mapping = class_mapping or {
            'motorbike': 0, 'car': 1, 'bus': 2, 'truck': 3,
        }
        self.num_classes = len(self.class_mapping)

        # 获取所有图像文件
        self.image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.image_files.sort()

        # 预加载所有标注
        self.annotations_cache = self._preload_annotations()
        print(f"Dataset initialized with {len(self.image_files)} images and preloaded annotations.")

        # 数据增强配置
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """获取数据增强变换"""
        if self.is_training:
            return A.Compose([
                # 基础变换
                A.Resize(height=self.img_size, width=self.img_size),

                # 几何变换 
                A.HorizontalFlip(p=0.1),  # 水平翻转
                A.OneOf([
                    A.Affine(
                        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # 平移范围
                        scale=(0.9, 1.1),  # 缩放范围
                        rotate=(-10, 10), # 旋转范围
                        p=1.0
                    ),
                    A.Perspective(scale=(0.02, 0.05), p=1.0),  # 透视变换强度
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),  # 网格畸变
                ], p=0.1),  # 几何变换概率

                # 颜色增强 
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,  # 亮度变化
                        contrast_limit=0.15,    # 对比度变化
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,     # 整数值，轻微色相变化
                        sat_shift_limit=20,     # 适度饱和度变化
                        val_shift_limit=15,     # 明度变化
                        p=1.0
                    ),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
                ], p=0.6),  # 颜色增强概率

                # 噪声和模糊 
                A.OneOf([
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),  # 模糊程度
                    A.GaussianBlur(blur_limit=(1, 3), p=1.0),  # 轻微高斯模糊
                    A.GaussNoise(var_limit=(5.0, 15.0), mean=0, p=1.0),  # 噪声强度
                ], p=0.2),  # 模糊/噪声概率

                # 天气效果 
                A.OneOf([
                    A.RandomRain(
                        slant_range=(-10, 10),  # 雨滴角度范围
                        drop_length=1, 
                        drop_width=1, 
                        drop_color=(200, 200, 200),
                        blur_value=1,
                        brightness_coefficient=0.8,
                        rain_type="drizzle",  # 使用小雨效果
                        p=1.0
                    ),
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1), 
                        num_shadows_limit=(1, 3),  # 阴影数量
                        shadow_dimension=5,
                        p=1.0
                    ),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_lower=0,
                        angle_upper=1,
                        num_flare_circles_lower=1,
                        num_flare_circles_upper=2,
                        src_radius=50,
                        p=1.0
                    ),
                ], p=0.001),  # 天气效果概率

                # 遮挡增强 
                A.CoarseDropout(
                    max_holes=5,      # hole数量
                    max_height=20,    # 最大高度
                    max_width=20,     # 最大宽度
                    min_holes=1,
                    min_height=8,     # 最小高度
                    min_width=8,      # 最小宽度
                    fill_value=0,
                    p=0.1             # 遮挡概率
                ),

                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),  # 伽马校正
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # 对比度限制自适应直方图均衡
                    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),  # 锐化
                ], p=0.1),

                # 标准化
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_area=100,      # 最小面积要求
                min_visibility=0.4, # 最小可见性要求
                check_each_transform=True
            ))
        else:
            # 验证集变换
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                # 验证时轻微增强以提高鲁棒性
                A.HorizontalFlip(p=0.1),  # 水平翻转
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))

    def _preload_annotations(self):
        """预加载所有标注数据"""
        annotations = []
        for i, img_file in enumerate(self.image_files):
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(self.labels_dir, base_name + '.txt')
            bboxes, class_labels = self._load_txt_annotations(txt_path)
            annotations.append({'bboxes': bboxes, 'class_labels': class_labels})
            
            if (i + 1) % 1000 == 0:
                print(f"Preloaded {i + 1}/{len(self.image_files)} annotations.")
        return annotations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Could not read image {img_path}. Returning a black image.")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取标注
        annotations_data = self.annotations_cache[idx]
        bboxes = annotations_data['bboxes']
        class_labels = annotations_data['class_labels']

        # 处理边界框
        if len(bboxes) > 0:
            bboxes = self._fix_bbox_coordinates(bboxes)
            valid_bboxes, valid_labels = self._filter_valid_bboxes(bboxes, class_labels)
            bboxes, class_labels = valid_bboxes, valid_labels

        # 应用变换
        try:
            bboxes_array = np.array(bboxes, dtype=np.float32)

            if not self._validate_bboxes(bboxes_array):
                return self._fallback_processing(image, img_path)

            transformed = self.transform(
                image=image,
                bboxes=bboxes_array.tolist(),
                class_labels=class_labels
            )

            image_tensor = transformed['image']
            transformed_bboxes = transformed.get('bboxes', [])
            transformed_labels = transformed.get('class_labels', [])

            # 构建标签张量
            if len(transformed_bboxes) > 0:
                transformed_bboxes = self._fix_bbox_coordinates(transformed_bboxes)
                valid_bboxes, valid_labels = self._filter_valid_bboxes(transformed_bboxes, transformed_labels)

                if len(valid_bboxes) > 0:
                    labels_tensor = torch.tensor([
                        [cls] + bbox for cls, bbox in zip(valid_labels, valid_bboxes)
                    ], dtype=torch.float32)
                else:
                    labels_tensor = torch.zeros((0, 5))
            else:
                labels_tensor = torch.zeros((0, 5))

        except Exception as e:
            print(f"Error in augmentation for {img_path}: {e}")
            return self._fallback_processing(image, img_path)

        return image_tensor, labels_tensor, img_path

    def _validate_bboxes(self, bboxes):
        """验证边界框是否有效"""
        if len(bboxes) == 0:
            return True

        # 检查数值范围和有效性
        if (np.any(bboxes < -0.01) or np.any(bboxes > 1.01) or
            np.any(bboxes[:, 2] <= 0) or np.any(bboxes[:, 3] <= 0)):
            return False

        # 检查边界框是否在图像内
        half_w = bboxes[:, 2] / 2
        half_h = bboxes[:, 3] / 2

        return not (np.any(bboxes[:, 0] - half_w < -0.01) or 
                   np.any(bboxes[:, 0] + half_w > 1.01) or
                   np.any(bboxes[:, 1] - half_h < -0.01) or 
                   np.any(bboxes[:, 1] + half_h > 1.01))

    def _fallback_processing(self, image, img_path):
        """fallback处理"""
        try:
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

            # 标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

            labels_tensor = torch.zeros((0, 5))
            return image_tensor, labels_tensor, img_path

        except Exception as e:
            print(f"Error in fallback processing for {img_path}: {e}")
            image_tensor = torch.zeros((3, self.img_size, self.img_size))
            labels_tensor = torch.zeros((0, 5))
            return image_tensor, labels_tensor, img_path

    def _fix_bbox_coordinates(self, bboxes):
        """边界框坐标"""
        if len(bboxes) == 0:
            return []

        bboxes = np.array(bboxes, dtype=np.float32)
        eps = 1e-6
        
        # 限制在有效范围内
        bboxes = np.clip(bboxes, eps, 1.0 - eps)
        
        # 确保宽高有效
        bboxes[:, 2] = np.maximum(bboxes[:, 2], eps)
        bboxes[:, 3] = np.maximum(bboxes[:, 3], eps)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], 1.0 - eps)
        bboxes[:, 3] = np.minimum(bboxes[:, 3], 1.0 - eps)

        # 调整中心点
        half_w = bboxes[:, 2] / 2
        half_h = bboxes[:, 3] / 2
        bboxes[:, 0] = np.clip(bboxes[:, 0], half_w + eps, 1.0 - half_w - eps)
        bboxes[:, 1] = np.clip(bboxes[:, 1], half_h + eps, 1.0 - half_h - eps)

        return bboxes.tolist()

    def _filter_valid_bboxes(self, bboxes, class_labels):
        """过滤有效的边界框"""
        if len(bboxes) == 0:
            return [], []

        bboxes = np.array(bboxes, dtype=np.float32)
        class_labels = np.array(class_labels)

        min_dim_ratio = 0.001 if self.is_training else 0.0005

        valid_mask = (
            (bboxes[:, 2] > min_dim_ratio) &
            (bboxes[:, 3] > min_dim_ratio) &
            (bboxes[:, 0] >= 0 - 1e-5) & (bboxes[:, 0] <= 1 + 1e-5) &
            (bboxes[:, 1] >= 0 - 1e-5) & (bboxes[:, 1] <= 1 + 1e-5) &
            (class_labels >= 0) & (class_labels < self.num_classes)
        )

        return bboxes[valid_mask].tolist(), class_labels[valid_mask].tolist()

    def _load_txt_annotations(self, txt_path):
        """加载TXT格式标注"""
        bboxes = []
        class_labels = []

        if not os.path.exists(txt_path):
            return [], []

        try:
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            try:
                                class_id, x, y, w, h = map(float, parts)

                                if (0 <= x <= 1 and 0 <= y <= 1 and
                                    0 < w <= 1 and 0 < h <= 1):
                                    bboxes.append([x, y, w, h])
                                    class_labels.append(int(class_id))
                            except ValueError:
                                continue

        except Exception as e:
            print(f"Error loading TXT annotations from {txt_path}: {e}")

        return bboxes, class_labels


def aic_collate_fn(batch):
    """优化的批处理函数"""
    images, labels, img_paths = zip(*batch)

    # 过滤有效项目
    valid_items = [(img, lbl, path) for img, lbl, path in zip(images, labels, img_paths)
                   if img is not None and lbl is not None]

    if not valid_items:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return (torch.zeros(1, 3, 640, 640, device=device),
                torch.zeros(0, 6, device=device),
                ['empty'])

    images, labels, img_paths = zip(*valid_items)

    try:
        images = torch.stack(images, 0)
    except Exception as e:
        print(f"Error stacking images: {e}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return (torch.zeros(len(images), 3, 640, 640, device=device),
                torch.zeros(0, 6, device=device),
                list(img_paths))

    device = images.device
    batch_labels = []
    
    for i, label in enumerate(labels):
        if isinstance(label, torch.Tensor) and label.shape[0] > 0:
            if label.ndim == 1:
                label = label.unsqueeze(0)
            label = label.to(device)

            batch_idx = torch.full((label.shape[0], 1), i, dtype=torch.float32, device=device)
            label_with_batch = torch.cat([batch_idx, label], dim=1)
            batch_labels.append(label_with_batch)

    if batch_labels:
        batch_labels = torch.cat(batch_labels, 0)
    else:
        batch_labels = torch.zeros((0, 6), dtype=torch.float32, device=device)

    return images, batch_labels, list(img_paths)