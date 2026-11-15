# 文件：models/retinanet.py
# 核心作用：把 backbone / neck / head 组装成整机；提供 forward 与"逐层结果展平"工具
# 运行自检：python -m models.retinanet
# 调用关系：trainer/train.py、trainer/eval.py、tools/predict.py 都 import RetinaNet
import torch                                         # 张量
import torch.nn as nn                                # 层
from models.backbone import Backbone         # 骨干（models/backbone.py）
from models.neck import FPN                          # 颈部（models/neck.py）
from models.head import Head                   # 检测头（models/head.py）
from models.anchor import generate_anchors           # anchor 生成（models/anchor.py）
from config.config import num_classes                # 类别数


class RetinaNet(nn.Module):
    """功能：整机 = backbone + neck + head + anchor
    输入：x (B,3,H,W)
    输出：(cls_list, reg_list, anchors)
    调用位置：trainer/train.py 训练循环；trainer/eval.py；tools/predict.py
    """

    def __init__(self, pretrained=True):
        super().__init__()                            # 初始化
        self.backbone = Backbone(pretrained)  # models/backbone.py
        self.neck = FPN()                             # models/neck.py
        self.head = Head()                      # models/head.py

    def forward(self, x):
        """功能：一次前向
        输入：x (B,3,320,320)
        输出：cls_list (3 个 (B,180,H,W))、reg_list (3 个 (B,36,H,W))、anchors (18900,4)
        调用位置：trainer/train.py 的 train_one_epoch()；trainer/eval.py
        """
        c3, c4, c5 = self.backbone(x)                                  # 骨干输出 C3/C4/C5
        feats = self.neck([c3, c4, c5])                                # FPN → (P3,P4,P5)  输出：[P3(B,256,40,40), P4(B,256,20,20), P5(B,256,10,10)]
        cls_list, reg_list = self.head(feats)                           # 逐层预测  输出：cls_list = [ (B,9×20,H,W) ×3 ]；reg_list = [ (B,9×4,H,W) ×3 ]
        sizes = [(c.shape[-2], c.shape[-1]) for c in cls_list]          # 各层高宽H，W
        anchors = generate_anchors(sizes, x.device)                     # 生成所有 anchor（18900,4）左手/右下坐标  生成顺序 = 层 → 网格位置(h,w) → 尺度 → 比例
        return cls_list, reg_list, anchors


def flatten_outputs(cls_list, reg_list):
    """功能：把逐层输出拼成"每张图一条长表"，与 anchors 顺序一一对应
    输入：cls_list=[(B,A*C,H,W)×3]、reg_list=[(B,A*4,H,W)×3]，A=9，C=20
    输出：cls (B, N, C)、reg (B, N, 4)，N = 18900 = 40*40*9 + 20*20*9 + 10*10*9
    调用位置：losses/loss.py 的 Loss.forward()；trainer/eval.py
    """
    cls_flat, reg_flat = [], []                       # 收集展平后的层
    for c, r in zip(cls_list, reg_list):              # cls_list张量 = [(B, A*C, 40, 40), (B, A*C, 20, 20), (B, A*C, 10, 10)]
        b, ac, h, w = c.shape                         # shape = (B, A*C, H, W) = (B, 9*20, H, W)
        a = ac // num_classes                         # 反推 anchor 数 = 9
        # (B,9*C,40,40) → (B,9,C,40,40) → (B,40,40,9,C) → (B,40*40*9,C)
        cls_flat.append(c.view(b, a, num_classes, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, num_classes))
        # (B,9*4,40,40) → (B,9,4,40,40) → (B,40,40,9,4) → (B,40*40*9,4)
        reg_flat.append(r.view(b, a, 4, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4))
        # 输出展平cls_flat = [(B, 40*40*9, 20), (B, 20*20*9, 20), (B, 10*10*9, 20)]，reg_flat = [(B, 40*40*9, 4), (B, 20*20*9, 4), (B, 10*10*9, 4)]
    # dim=1,按 anchor 维度拼接，输出 cls (B, 18900, 20)、reg (B, 18900, 4)，所有层拼接，每张图一条长表，顺序 = 层 → 网格位置(h,w) → 尺度 → 比例，与 anchors 顺序一一对应
    return torch.cat(cls_flat, dim=1), torch.cat(reg_flat, dim=1)      


if __name__ == "__main__":
    # 自检：造一个假 batch，看三层输出形状 + anchor 总数，python -m models.retinanet
    net = RetinaNet(pretrained=False)                 # 自检不下载权重
    x = torch.randn(2, 3, 320, 320)                   # 假输入
    cls_list, reg_list, anchors = net(x)              # 前向，cls_list = [(B, 9*20, H, W) ×3]，reg_list = [(B, 9*4, H, W) ×3]，anchors = (18900,4)
    for i, (c, r) in enumerate(zip(cls_list, reg_list)): # 逐层打印,cls-三个张量
        print(f"[retinanet] P{i + 3} cls={tuple(c.shape)} reg={tuple(r.shape)}")
    print("[retinanet] anchors:", tuple(anchors.shape))   # 期望 (18900,4)
    # 第一种尺度s=32：anchor[0] = [-7.3137, -18.6274, 15.3137, 26.6274],anchor[1] = [-12.0, -12.0, 20.0, 20.0],anchor[2] = [-18.6274, -7.3137, 26.6274, 15.3137]，
    # 第二个尺度s=40.3175: anchor[3] = [-10.2542, -24.5084, 18.2542, 32.5084],anchor[4] = [-16.1587, -16.1587, 24.1587, 24.1587],anchor[5] = [-24.5084, -10.2542, 32.5084, 18.2542]
    print("[retinanet] anchors[3]:", anchors[3].tolist()) # 打印 ，顺序 = 层 → 网格位置(h,w) → 尺度 → 比例，第一层特征图的一个网格，第一个尺度3个anchor（0.5，1，1.5），第二个尺度（0.5，1.1.5）