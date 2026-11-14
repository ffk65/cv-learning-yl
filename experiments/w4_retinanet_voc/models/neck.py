# 文件：models/neck.py
# 核心作用：FPN 把 C3/C4/C5 融合成 P3/P4/P5（对应 mmdet 的 models/necks/fpn.py）
# 调用关系：RetinaNet.__init__ 创建本类；RetinaNet.forward 把 backbone 的输出喂给本类
import torch.nn.functional as F                      # 用于最近邻上采样
import torch.nn as nn                                # 网络层
from config.config import fpn_channels              # 统一通道数 256


class FPN(nn.Module):
    """功能：自顶向下 + 横向连接，输出三层 256 通道特征
    输入：feats = [C3, C4, C5]（来自 Backbone）
    输出：[P3(B,256,40,40), P4(B,256,20,20), P5(B,256,10,10)]
    调用位置：RetinaNet.forward()（models/retinanet.py）
    """

    def __init__(self, in_channels=(128, 256, 512), out_channels=fpn_channels):
        super().__init__()                            # 初始化 nn.Module
        c = out_channels                              # 简写，256
        self.lat3 = nn.Conv2d(in_channels[0], c, kernel_size=1)  # 横向 1×1：C3 128→256
        self.lat4 = nn.Conv2d(in_channels[1], c, kernel_size=1)  # 横向 1×1：C4 256→256
        self.lat5 = nn.Conv2d(in_channels[2], c, kernel_size=1)  # 横向 1×1：C5 512→256
        # 融合后再过 3×3，消除上采样带来的"网格感"（对应 6.8 FPN 的输出卷积）
        self.out3 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.out4 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.out5 = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, feats):
        """功能：自顶向下融合，返回三层特征
        输入：feats=[C3,C4,C5]
        输出：[P3,P4,P5]
        调用位置：RetinaNet.forward()（models/retinanet.py）
        """
        c3, c4, c5 = feats
        p5 = self.lat5(c5)                            # 顶层直接用横向卷积
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")  # 上采样后逐元素相加
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")  # 与 p4 相同的自顶向下融合
        return [self.out3(p3), self.out4(p4), self.out5(p5)]   # 3×3 平滑后输出