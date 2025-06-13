import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        planes = expansion * in_planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride == 1:
            out = out + self.shortcut(x)
        return out

class MobileNetV2Backbone(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),  # 注意第二阶段步长要为2，才能降维
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1)
    ]

    def __init__(self):
        super(MobileNetV2Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)  # 步长2
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(32)
        self.conv2 = nn.Conv2d(320, 1280, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for s in strides:
                layers.append(Block(in_planes, out_planes, expansion, s))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # [B, 32, 112, 112]
        out = self.layers(out)                  # [B, 320, 7, 7]
        out = F.relu(self.bn2(self.conv2(out))) # [B, 1280, 7, 7]
        return out


if __name__ == '__main__':
    net = MobileNetV2Backbone()
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(y.shape)  # 期望: torch.Size([1, 1280, 7, 7])