# 文件：models/backbone.py

# 对应参考：mmdet mmdet/models/backbones/resnet.py + mmdet/models/necks/fpn.py；
#           detectron2 detectron2/modeling/backbone/resnet.py + fpn.py
# 预期输出形状（IMG_SIZE=320）：P3 (B,256,40,40)、P4 (B,256,20,20)、P5 (B,256,10,10)
# 状态：占位文件，待实现

# 核心作用：ResNet-18 只提取 C3/C4/C5（对应 mmdet 的 models/backbones/resnet.py）
# 调用关系：models/neck.py 的 FPN.forward 接收本类输出；models/retinanet.py 的 RetinaNet.forward 先调本类
import torch.nn as nn                                # 网络层
from torchvision.models import resnet18, ResNet18_Weights   # 官方 ResNet-18 + 预训练权重


class Backbone(nn.Module):
    """功能：骨干网络，输出 stride 8/16/32 的三层特征
    输入：x (B,3,H,W)，H/W 建议 320
    输出：[C3(B,128,H/8,W/8), C4(B,256,H/16,W/16), C5(B,512,H/32,W/32)]
    调用位置：RetinaNet.forward()（models/retinanet.py）第一行
    """

    def __init__(self, pretrained=True):
        super().__init__()                            # 初始化 nn.Module
        # 预训练权重能显著加快收敛；没有网络时会自动下载到 ~/.cache/torch
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        rn = resnet18(weights=weights)                # torchvision完整分类版ResNet18，自带ImageNet1000类分类头，本项目不使用最后的fc分类层-目标检测
        # 按 stride 拆开：stem=/4，layer1=/4，layer2=/8，layer3=/16，layer4=/32
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool) # (B, 64, 80, 80),stride=4（conv+pool）
        self.layer1 = rn.layer1                       #conv2_x， 输出 (B, 64, 80, 80),stride=4(下采样倍率) → C2
        self.layer2 = rn.layer2                       #conv3_x， 输出 (B,128, 40, 40),stride=8 → C3
        self.layer3 = rn.layer3                       #conv4_x， 输出 (B,256, 20, 20),stride=16 → C4
        self.layer4 = rn.layer4                       #conv5_x， 输出 (B,512, 10, 10),stride=32 → C5

    def forward(self, x):
        """功能：一次前向，返回 C3/C4/C5
        输入：x (B,3,H,W)
        输出：[C3, C4, C5]
        调用位置：RetinaNest.forward()；neck.FPN.forward()
        """
        c1 = self.stem(x)                             # (B, 64, 80, 80)
        c2 = self.layer1(c1)                          # (B, 64, 80, 80)
        c3 = self.layer2(c2)                          # (B,128, 40, 40)
        c4 = self.layer3(c3)                          # (B,256, 20, 20)
        c5 = self.layer4(c4)                          # (B,512, 10, 10)
        return [c3, c4, c5]