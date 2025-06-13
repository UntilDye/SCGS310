import torch
import torch.nn as nn
from models.MyMobileNet import MobileNetV2Backbone
""" from MyMobileNet import MobileNetV2Backbone """
import yaml

class YOLO2_MobileNetV2(nn.Module):
    def __init__(self, S=7, nc=1):
        with open(r'C:\code\AISYSTEM\YOLO\config.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        super(YOLO2_MobileNetV2, self).__init__()
        self.S = cfg['model']['s']  # 16
        self.backbone = MobileNetV2Backbone()  # [B, 1280, 16, 16]
        self.conv = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(size=(S, S), mode='bilinear', align_corners=False),  # 输出S x S特征图
        )
        self.pred = nn.Conv2d(256, 5, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = self.pred(x)
        x = x.permute(0, 2, 3, 1)  # [B, S, S, 5]
        # 对 x_cell, y_cell, w, h 都做 sigmoid
        x[..., 1:5] = torch.sigmoid(x[..., 1:5])
        return x

if __name__ == '__main__':
    net = YOLO2_MobileNetV2(S=16)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.shape)  # torch.Size([2, 16, 16, 5])