# 文件：models/head.py
# 核心作用：RetinaNet 的检测头（分类塔 + 回归塔，各层共享权重）
# 调用关系：models/retinanet.py 的 RetinaNet.__init__ 创建本类；RetinaNet.forward 调用本类
import math                                          # log
import torch.nn as nn                                # 层
from config.config import num_classes, anchor_ratios, fpn_channels, scales_per_octave   # 常量


class Head(nn.Module):
    """功能：RetinaNet 的检测头（各层共享权重）
    输入：feats = [P3,P4,P5]（每个 256 通道）
    输出：cls_list = [ (B,9×20,H,W) ×3 ]；reg_list = [ (B,9×4,H,W) ×3 ]
    调用位置：RetinaNet.forward()
    """

    def __init__(self, num_classes=num_classes, num_anchors=None, ch=fpn_channels):
        super().__init__()                            # 初始化
        num_anchors = num_anchors or scales_per_octave * len(anchor_ratios)   # 每位置 anchor 数 = 3×3=9
        # 分类塔：两层 3×3（原文是 4 层，这里减半，6GB 更快）
        self.cls_tower = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),           # str=3，pad=1--特征图宽高不变
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), 
        )
        # 回归塔：结构独立（原文就是分类/回归分开，不共享）
        self.reg_tower = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), 
        )
        self.cls_logits = nn.Conv2d(ch, num_anchors * num_classes, 1)   # 1×1 → 每位置 9×20=180 个分类分数
        self.bbox_pred = nn.Conv2d(ch, num_anchors * 4, 1)              # 1×1 → 每位置 9×4=36 个回归偏移
        # 初始化权重（RetinaNet 原文做法）
        for m in self.modules():                      
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)    # 小方差，训练初期稳定
                nn.init.constant_(m.bias, 0)
        # ===================== RetinaNet 重点初始化 =====================
        # 分类头输出是sigmoid logit，不是概率
        # 目标：初始状态下，每个anchor预测为前景的概率≈0.01
        # 公式推导：p = sigmoid(z) → z = ln(p/(1-p))
        # p=0.01代入，分类 bias 初始化为z = ln(0.01/0.99) = -ln(0.99/0.01) ≈ -4.595
        # 为什么这么做：目标检测anchor正负样本极不均衡，背景anchor占99%以上
        # 如果bias=0，初始预测前景概率0.5，大量负样本会产生巨大loss，模型学崩、震荡不收敛
        # 把bias置负，模型一开始倾向输出背景，和Focal Loss配合，大幅稳定前期训练
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, feats):
        """功能：逐层出分类/回归预测
        输入：feats = [P3,P4,P5]
        输出：cls_list、reg_list（见类注释）
        调用位置：RetinaNet.forward()
        """
        cls_outs, reg_outs = [], []                    # 收集三层结果
        for f in feats:                                # 逐层（权重共享）
            cls_outs.append(self.cls_logits(self.cls_tower(f)))   # 分类
            reg_outs.append(self.bbox_pred(self.reg_tower(f)))    # 回归
        return cls_outs, reg_outs