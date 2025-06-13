# models/yolo.py
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torch
class ConvBlock(nn.Module):
    """正则化"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0.2):  
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
    def forward(self, x):
        return self.block(x)

class CSPBlock(nn.Module):
    """增加dropout正则化"""
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True, dropout=0.1):
        super().__init__()
        self.shortcut = shortcut and in_channels == out_channels
        mid_channels = out_channels // 2
        
        self.conv1 = ConvBlock(in_channels, mid_channels, 1, 1, dropout=dropout)
        self.conv2 = ConvBlock(in_channels, mid_channels, 1, 1, dropout=dropout)
        
        self.blocks = nn.Sequential(*[
            ConvBlock(mid_channels, mid_channels, 3, 1, dropout=dropout) 
            for _ in range(num_blocks)
        ])
        
        self.conv3 = ConvBlock(mid_channels * 2, out_channels, 1, 1, dropout=dropout)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        out = self.conv3(torch.cat([y1, y2], dim=1))
        out = self.dropout(out)
        return x + out if self.shortcut else out

class YOLO(nn.Module):
    def __init__(self, num_classes, input_shape=(640, 640)):
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.stride = [8, 16, 32]
        self.anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]

        # backbone
        self.backbone = self._build_backbone()
        
        # neck
        self.neck = self._build_neck()
        
        # 检测头
        self.head_large = self._build_detection_head(256, num_classes)   # P3
        self.head_medium = self._build_detection_head(512, num_classes)  # P4
        self.head_small = self._build_detection_head(1024, num_classes)  # P5

    def _build_backbone(self):
        """backbone构建 - 确保正确的通道数和特征层"""
        layers = nn.ModuleList()
        
        # Stem - 初始特征提取
        layers.append(ConvBlock(3, 32, 6, 2, dropout=0.05))      # 0: 640->320, 32 channels
        layers.append(ConvBlock(32, 64, 3, 2, dropout=0.05))     # 1: 320->160, 64 channels
        
        # Stage 1 - P3特征层准备
        layers.append(CSPBlock(64, 128, num_blocks=2))           # 2: 160x160, 128 channels
        layers.append(ConvBlock(128, 256, 3, 2, dropout=0.1))    # 3: 160->80, 256 channels (P3特征)
        
        # Stage 2 - P4特征层准备
        layers.append(CSPBlock(256, 256, num_blocks=3))          # 4: 80x80, 256 channels
        layers.append(ConvBlock(256, 512, 3, 2, dropout=0.1))    # 5: 80->40, 512 channels (P4特征)
        
        # Stage 3 - P5特征层准备
        layers.append(CSPBlock(512, 512, num_blocks=4))          # 6: 40x40, 512 channels
        layers.append(ConvBlock(512, 1024, 3, 2, dropout=0.1))   # 7: 40->20, 1024 channels (P5特征)
        
        # 最终特征增强
        layers.append(CSPBlock(1024, 1024, num_blocks=2))        # 8: 20x20, 1024 channels
        
        return layers

    def _build_neck(self):
        """neck构建 - 确保通道数匹配"""
        return nn.ModuleDict({
            # Top-down lateral convolutions
            'lateral_conv1': ConvBlock(1024, 512, 1, 1),  # P5: 1024->512
            'lateral_conv2': ConvBlock(512, 256, 1, 1),   # P4: 512->256
            
            # Bottom-up downsampling
            'downsample1': ConvBlock(256, 256, 3, 2),     # P3->P4: stride=2
            'downsample2': ConvBlock(512, 512, 3, 2),     # P4->P5: stride=2
            
            # Feature fusion convolutions
            'fusion_conv1': CSPBlock(512 + 512, 512, num_blocks=1),   # P4融合: 512+512->512
            'fusion_conv2': CSPBlock(256 + 256, 256, num_blocks=1),   # P3融合: 256+256->256
            'fusion_conv3': CSPBlock(256 + 256, 512, num_blocks=1),   # P4增强: 256+256->512
            'fusion_conv4': CSPBlock(512 + 512, 1024, num_blocks=1),  # P5增强: 512+512->1024
        })

    def _build_detection_head(self, in_channels, num_classes):
        """构建检测头"""
        return nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, 3, 1, dropout=0.1),
            ConvBlock(in_channels // 2, in_channels // 2, 3, 1, dropout=0.1),
            nn.Conv2d(in_channels // 2, 3 * (num_classes + 5), 1, 1, 0)
        )

    def forward(self, x):
        # 确保输入尺寸
        if x.shape[-1] != self.input_shape[0] or x.shape[-2] != self.input_shape[1]:
            x = F.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)
        
        # Backbone forward -特征提取
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # 保存正确的特征层 - P3(80x80,256), P4(40x40,512), P5(20x20,1024)
            if i == 3:  # P3: 80x80, 256 channels
                features.append(x)
            elif i == 5:  # P4: 40x40, 512 channels
                features.append(x)
            elif i == 8:  # P5: 20x20, 1024 channels (经过最终CSP增强)
                features.append(x)
        
        # 确保有3个特征层
        if len(features) != 3:
            raise RuntimeError(f"Expected 3 feature maps, got {len(features)}. Feature shapes: {[f.shape for f in features]}")
        
        # 特征图: C3(80x80,256), C4(40x40,512), C5(20x20,1024)
        c3, c4, c5 = features
        
        
        # PANet neck - 特征融合
        # Top-down path
        p5 = self.neck['lateral_conv1'](c5)  # 1024 -> 512
        # print(f"Debug - P5 after lateral: {p5.shape}")
        
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        # print(f"Debug - P5 upsampled: {p5_up.shape}, C4: {c4.shape}")
        
        p4 = self.neck['fusion_conv1'](torch.cat([c4, p5_up], dim=1))  # 512+512 -> 512
        # print(f"Debug - P4 after fusion: {p4.shape}")
        
        p4_lateral = self.neck['lateral_conv2'](p4)  # 512 -> 256
        p4_up = F.interpolate(p4_lateral, size=c3.shape[-2:], mode='nearest')
        # print(f"Debug - P4 upsampled: {p4_up.shape}, C3: {c3.shape}")
        
        p3 = self.neck['fusion_conv2'](torch.cat([c3, p4_up], dim=1))  # 256+256 -> 256
        # print(f"Debug - P3 after fusion: {p3.shape}")
        
        # Bottom-up path
        p3_down = self.neck['downsample1'](p3)  # 256 -> 256, stride=2
        # print(f"Debug - P3 downsampled: {p3_down.shape}")
        
        p4_enhanced = self.neck['fusion_conv3'](torch.cat([p4_lateral, p3_down], dim=1))  # 256+256 -> 512
        # print(f"Debug - P4 enhanced: {p4_enhanced.shape}")
        
        p4_down = self.neck['downsample2'](p4_enhanced)  # 512 -> 512, stride=2
        # print(f"Debug - P4 downsampled: {p4_down.shape}")
        
        p5_enhanced = self.neck['fusion_conv4'](torch.cat([p5, p4_down], dim=1))  # 512+512 -> 1024
        # print(f"Debug - P5 enhanced: {p5_enhanced.shape}")
        
        # 检测头输出
        outputs = []
        outputs.append(self.head_large(p3))          # P3 - 大目标
        outputs.append(self.head_medium(p4_enhanced)) # P4 - 中目标  
        outputs.append(self.head_small(p5_enhanced))  # P5 - 小目标

        return outputs

    def predict(self, x, conf_threshold=0.4, iou_threshold=0.5, device='cpu'):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float() / 255.0
                if x.ndim == 3:
                    x = x.unsqueeze(0)
            x = x.to(device)
            
            predictions = self.forward(x)
            detections = self.post_process(predictions, conf_threshold, iou_threshold)
            return detections

    def post_process(self, predictions, conf_threshold=0.4, iou_threshold=0.5, 
                max_box_ratio=0.4, min_box_ratio=0.005):  # 降低max_box_ratio
        """后处理，严格的尺寸过滤"""
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        all_detections = []

        for b in range(batch_size):
            all_boxes, all_scores, all_classes = [], [], []
            
            for i, pred in enumerate(predictions):
                pred_b = pred[b]
                h, w = pred_b.shape[1], pred_b.shape[2]
                
                pred_b = pred_b.view(3, self.num_classes + 5, h, w)
                pred_b = pred_b.permute(0, 2, 3, 1).contiguous()
                
                grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                grid = torch.stack([grid_x, grid_y], dim=-1).float().to(device)
                
                for a in range(3):
                    anchor = torch.tensor(self.anchors[i][a]).float().to(device)
                    pred_anchor = pred_b[a]
                    
                    # 解码边界框 - 添加约束
                    xy = (pred_anchor[..., :2].sigmoid() * 2 - 0.5 + grid) * self.stride[i]
                    # 限制宽高增长
                    wh_raw = pred_anchor[..., 2:4].sigmoid() * 2
                    wh = torch.clamp(wh_raw, max=4.0) ** 2 * anchor  # 限制最大增长倍数
                    
                    conf = pred_anchor[..., 4].sigmoid()
                    cls_scores = pred_anchor[..., 5:].sigmoid()
                    cls_conf, cls_ids = cls_scores.max(dim=-1)
                    
                    total_conf = conf * cls_conf
                    
                    # 提高置信度阈值
                    mask = total_conf > (conf_threshold * 0.8)  # 更严格的初始过滤
                    if mask.sum() > 0:
                        xy_filtered = xy[mask]
                        wh_filtered = wh[mask]
                        conf_filtered = total_conf[mask]
                        cls_filtered = cls_ids[mask]
                        
                        # 更严格的尺寸过滤
                        img_size = self.input_shape[0]
                        wh_norm = wh_filtered / img_size
                        
                        # 多重尺寸检查
                        size_mask = (
                            (wh_norm[:, 0] >= min_box_ratio) & 
                            (wh_norm[:, 0] <= max_box_ratio) &
                            (wh_norm[:, 1] >= min_box_ratio) & 
                            (wh_norm[:, 1] <= max_box_ratio) &
                            # 添加面积检查
                            (wh_norm[:, 0] * wh_norm[:, 1] <= max_box_ratio * max_box_ratio) &
                            (wh_norm[:, 0] * wh_norm[:, 1] >= min_box_ratio * min_box_ratio * 4)
                        )
                        
                        # 长宽比过滤
                        aspect_ratios = wh_norm[:, 0] / (wh_norm[:, 1] + 1e-6)
                        aspect_mask = (aspect_ratios >= 0.33) & (aspect_ratios <= 3.0)  
                        
                        # 中心点位置检查
                        xy_norm = xy_filtered / img_size
                        center_mask = (
                            (xy_norm[:, 0] >= wh_norm[:, 0]/2) & 
                            (xy_norm[:, 0] <= 1 - wh_norm[:, 0]/2) &
                            (xy_norm[:, 1] >= wh_norm[:, 1]/2) & 
                            (xy_norm[:, 1] <= 1 - wh_norm[:, 1]/2)
                        )
                        
                        # 组合所有过滤条件
                        final_mask = size_mask & aspect_mask & center_mask
                        
                        if final_mask.sum() > 0:
                            xy_filtered = xy_filtered[final_mask]
                            wh_filtered = wh_filtered[final_mask]
                            conf_filtered = conf_filtered[final_mask]
                            cls_filtered = cls_filtered[final_mask]
                            
                            x1 = xy_filtered[:, 0] - wh_filtered[:, 0] / 2
                            y1 = xy_filtered[:, 1] - wh_filtered[:, 1] / 2
                            x2 = xy_filtered[:, 0] + wh_filtered[:, 0] / 2
                            y2 = xy_filtered[:, 1] + wh_filtered[:, 1] / 2
                            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                            
                            all_boxes.append(boxes)
                            all_scores.append(conf_filtered)
                            all_classes.append(cls_filtered)
            
            if all_boxes:
                all_boxes = torch.cat(all_boxes, dim=0)
                all_scores = torch.cat(all_scores, dim=0)
                all_classes = torch.cat(all_classes, dim=0)
                
                # 最终置信度过滤
                final_mask = all_scores > conf_threshold
                if final_mask.sum() > 0:
                    all_boxes = all_boxes[final_mask]
                    all_scores = all_scores[final_mask]
                    all_classes = all_classes[final_mask]
                    
                    # 降低IoU阈值，减少重复框
                    keep = torchvision.ops.nms(all_boxes, all_scores, iou_threshold * 0.8)
                    
                    final_boxes = all_boxes[keep].cpu().numpy()
                    final_scores = all_scores[keep].cpu().numpy()
                    final_classes = all_classes[keep].cpu().numpy()
                else:
                    final_boxes = np.zeros((0, 4))
                    final_scores = np.zeros((0,))
                    final_classes = np.zeros((0,))
            else:
                final_boxes = np.zeros((0, 4))
                final_scores = np.zeros((0,))
                final_classes = np.zeros((0,))
            
            all_detections.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "class_ids": final_classes.astype(int)
            })
        
        return all_detections