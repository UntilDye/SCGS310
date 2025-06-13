# models/yolo_vehicle.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.yolo import YOLO, ConvBlock

class VehicleYOLO(YOLO):
    """用于车辆检测的YOLO模型"""
    
    def __init__(self, num_classes=4, input_shape=(640, 640)):
        super().__init__(num_classes, input_shape)
        
        # 在特征提取阶段添加注意力机制
        self.vehicle_attention_p3 = VehicleAttentionModule(256)  # P3特征层
        self.vehicle_attention_p4 = VehicleAttentionModule(512)  # P4特征层  
        self.vehicle_attention_p5 = VehicleAttentionModule(1024) # P5特征层
        
        # 构建检测头以适应增强的特征
        self.head_large = self._build_enhanced_detection_head(256, num_classes)
        self.head_medium = self._build_enhanced_detection_head(512, num_classes)
        self.head_small = self._build_enhanced_detection_head(1024, num_classes)
    
    def _build_enhanced_detection_head(self, in_channels, num_classes):
        """构建增强的检测头"""
        return nn.Sequential(
            # 车辆特征增强
            VehicleFeatureEnhancer(in_channels),
            ConvBlock(in_channels, in_channels // 2, 3, 1, dropout=0.1),
            ConvBlock(in_channels // 2, in_channels // 2, 3, 1, dropout=0.1),
            nn.Conv2d(in_channels // 2, 3 * (num_classes + 5), 1, 1, 0)
        )
    
    def forward(self, x):
        # 确保输入尺寸
        if x.shape[-1] != self.input_shape[0] or x.shape[-2] != self.input_shape[1]:
            x = F.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)
        
        # Backbone forward - 提取特征
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 3:  # P3: 80x80, 256 channels
                features.append(x)
            elif i == 5:  # P4: 40x40, 512 channels
                features.append(x)
            elif i == 8:  # P5: 20x20, 1024 channels
                features.append(x)
        
        if len(features) != 3:
            raise RuntimeError(f"Expected 3 feature maps, got {len(features)}")
        
        c3, c4, c5 = features
        
        # 应用车辆注意力机制到原始特征
        c3_enhanced = self.vehicle_attention_p3(c3)
        c4_enhanced = self.vehicle_attention_p4(c4)
        c5_enhanced = self.vehicle_attention_p5(c5)
        
        # PANet neck - 使用增强的特征
        # Top-down path
        p5 = self.neck['lateral_conv1'](c5_enhanced)
        p5_up = F.interpolate(p5, size=c4_enhanced.shape[-2:], mode='nearest')
        
        p4 = self.neck['fusion_conv1'](torch.cat([c4_enhanced, p5_up], dim=1))
        p4_lateral = self.neck['lateral_conv2'](p4)
        p4_up = F.interpolate(p4_lateral, size=c3_enhanced.shape[-2:], mode='nearest')
        
        p3 = self.neck['fusion_conv2'](torch.cat([c3_enhanced, p4_up], dim=1))
        
        # Bottom-up path
        p3_down = self.neck['downsample1'](p3)
        p4_enhanced = self.neck['fusion_conv3'](torch.cat([p4_lateral, p3_down], dim=1))
        
        p4_down = self.neck['downsample2'](p4_enhanced)
        p5_enhanced = self.neck['fusion_conv4'](torch.cat([p5, p4_down], dim=1))
        
        # 检测头输出
        outputs = []
        outputs.append(self.head_large(p3))
        outputs.append(self.head_medium(p4_enhanced))
        outputs.append(self.head_small(p5_enhanced))
        
        return outputs

class VehicleAttentionModule(nn.Module):
    """单个特征层的车辆注意力模块"""
    
    def __init__(self, channels):
        super().__init__()
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(channels)
        
        # 车辆特定的特征增强
        self.vehicle_enhancer = nn.Sequential(
            ConvBlock(channels, channels, 3, 1, dropout=0.1),
            ConvBlock(channels, channels, 1, 1)
        )

    def forward(self, x):
        # 应用注意力机制
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        
        # 组合注意力
        attended = x * channel_att * spatial_att
        
        # 车辆特征增强
        enhanced = self.vehicle_enhancer(attended)
        
        # 残差连接
        return x + enhanced

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention)

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 确保reduced_channels至少为1
        reduced_channels = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = avg_out + max_out
        return self.sigmoid(attention)

class VehicleFeatureEnhancer(nn.Module):
    """车辆特征增强模块"""
    
    def __init__(self, channels):
        super().__init__()
        
        # 车辆形状特征检测
        self.shape_detector = nn.Sequential(
            ConvBlock(channels, channels//4, 1, 1),
            ConvBlock(channels//4, channels//4, 3, 1),
            ConvBlock(channels//4, channels//2, 1, 1)
        )
        
        # 车辆纹理特征检测  
        self.texture_detector = nn.Sequential(
            ConvBlock(channels, channels//4, 1, 1),
            ConvBlock(channels//4, channels//4, 3, 1, dropout=0.1),
            ConvBlock(channels//4, channels//2, 1, 1)
        )
        
        # 特征融合
        self.fusion = ConvBlock(channels, channels, 1, 1)
        
    def forward(self, x):
        shape_features = self.shape_detector(x)
        texture_features = self.texture_detector(x)
        
        # 融合特征
        combined = torch.cat([shape_features, texture_features], dim=1)
        enhanced = self.fusion(combined)
        
        return enhanced
        

class VehicleSpecificLoss(nn.Module):
    """车辆检测专用损失函数"""
    
    def __init__(self, base_loss, vehicle_weight=1.5):
        super().__init__()
        self.base_loss = base_loss
        self.vehicle_weight = vehicle_weight
        
        # 车辆类别
        self.vehicle_classes = [0, 1, 2, 3]  # car, truck, bus, motorbike
    
    def forward(self, predictions, targets):
        # 基础损失
        total_loss, loss_items = self.base_loss(predictions, targets)
        
        # 对车辆目标增加权重
        if targets.shape[0] > 0:
            vehicle_mask = torch.isin(targets[:, 1].long(), 
                                    torch.tensor(self.vehicle_classes, device=targets.device))
            if vehicle_mask.sum() > 0:
                # 为车辆目标增加额外的权重
                vehicle_penalty = self.vehicle_weight * vehicle_mask.float().mean()
                total_loss = total_loss * (1 + vehicle_penalty)
        
        return total_loss, loss_items