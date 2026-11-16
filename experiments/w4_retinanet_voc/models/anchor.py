# 文件：models/anchor.py
# 核心作用：① 为 P3/P4/P5 生成 anchor；② 把 GT 编码成回归目标；③ 把预测解码成真实框
# 调用关系：RetinaNet.forward → generate_anchors；losses/loss.py → encode；trainer/eval.py、tools/predict.py → decode
# 对应参考：mmdet mmdet/models/task_modules/prior_generators/anchor_generator.py + coders/delta_xywh_bbox_coder.py；
#           detectron2 detectron2/modeling/anchor_generator.py + box2box_transform.py
# 计划参数：strides=[8,16,32]，scales=[4*2^(0,1/3,2/3)]（相对 stride），ratios=[0.5,1,2] -> 每位置 9 个
# 计划数量（IMG_SIZE=320）：1600×9 + 400×9 + 100×9 = 18900 个/图
# 状态：占位文件，待实现

import math                                          # sqrt / 2 的幂
import torch                                         # 张量
from config.config import (                          # 全部取自 config，数值对齐 mmdet 默认
    anchor_ratios, anchor_strides, bbox_std, octave_base_scale, scales_per_octave,
)


def generate_anchors(feature_sizes, device):
    """功能：生成所有 anchor（x1y1，x2y2 绝对坐标--左上角右上角）--特征图上映射原图坐标的瞄准框
    输入：feature_sizes=[(40,40),(20,20),(10,10)]（P3/P4/P5 的高宽）；device
    输出：(18900, 4) float32 —— 顺序 = 层 → 位置(h,w) → 尺度 → 比例, (N,4) = [x1,y1,x2,y2]
    调用位置：RetinaNet.forward()（models/retinanet.py）
    """
    # 一共 40x40 + 20x20 + 10x10 = 2100 个位置，每位置 3×3=9 个 anchor → 18900 个
    anchors = []                                     # 用 Python list 收集，最后一次性转张量
    for (fh, fw), stride in zip(feature_sizes, anchor_strides):   # 逐层：P3→8, P4→16, P5→32
        base = stride * octave_base_scale            # 当前 FPN 层 anchor 的基础边长 = stride×4（mmdet: base_size=1, octave_base_scale=4）
        # 每层 3 个尺度scales_per_octave：base × 2^(i/3)；P3=32/40.3/50.8  P4=64/80.6/101.6  P5=128/161.3/203.2
        level_scales = [base * 2 ** (i / scales_per_octave) for i in range(scales_per_octave)]
        for i in range(fh):                          # 行（y 方向）
            for j in range(fw):                      # 列（x 方向）
                # 计算每个网格位置的中心坐标（原图绝对坐标，左上角为原点） = (j+0.5)*stride
                cx = (j + 0.5) * stride              # (j+0.5)特征图坐标 x 倍率 → 原图中心绝对坐标
                cy = (i + 0.5) * stride              
                # 逐层每个位置生成 3×3=9 个 anchor（3 种尺度 × 3 种比例），输出为 xyxy 绝对坐标
                for s in level_scales:               # 3 种尺度（逐层不同，不能全层共用一套）
                    for r in anchor_ratios:          # 3 种比例（正方/竖长方/横长方） 
                        h = s / math.sqrt(r)         # 由"面积≈s² + 比例"反推高
                        w = s * math.sqrt(r)         # 反推宽
                        anchors.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return torch.tensor(anchors, dtype=torch.float32, device=device)


def encode(anchors, gt):
    """功能：把"anchor → GT"的偏移编码成回归目标（RetinaNet 的 4 参数形式）
    输入：anchors (N,4) xyxy；gt (N,4) xyxy（长度必须相同，已一一配对）
    输出：(N,4) = [dx, dy, dw, dh]
    调用位置：losses/loss.py 的 assign_targets()
    """
    aw = anchors[:, 2] - anchors[:, 0]               # anchor 宽
    ah = anchors[:, 3] - anchors[:, 1]               # anchor 高
    acx = anchors[:, 0] + 0.5 * aw                   # anchor 中心 x
    acy = anchors[:, 1] + 0.5 * ah                   # anchor 中心 y
    gw = gt[:, 2] - gt[:, 0]                         # GT 宽
    gh = gt[:, 3] - gt[:, 1]                         # GT 高
    gcx = gt[:, 0] + 0.5 * gw                        # GT 中心 x
    gcy = gt[:, 1] + 0.5 * gh                        # GT 中心 y
    # 此偏移是训练时的回归目标，训练时会乘以 std 放大 10 倍，便于梯度下降
    # 推理时偏移用reg_pred预测出来，乘以 std 再还原成真实框，不是此处偏移变量
    dx = (gcx - acx) / aw / bbox_std[0]              # 中心偏移 ÷ std_x=0.1
    dy = (gcy - acy) / ah / bbox_std[1]              #     放大10倍，信号增强，便于训练
    dw = torch.log(gw / aw) / bbox_std[2]            # 宽比取 log ÷ std_w=0.2
    dh = torch.log(gh / ah) / bbox_std[3]            # 高比取 log ÷ std_h=0.2
    return torch.stack([dx, dy, dw, dh], dim=1)      # (N,4)


def decode(anchors, deltas):# 推理时输入anchor和输出的偏移
    """功能：把网络输出的偏移还原成真实框（encode 的逆运算）
    输入：anchors (N,4) xyxy；deltas (N,4) = [dx,dy,dw,dh]
    输出：(N,4) xyxy
    调用位置：trainer/eval.py 的 postprocess()；tools/predict.py
    """
    aw = anchors[:, 2] - anchors[:, 0]               # anchor 宽
    ah = anchors[:, 3] - anchors[:, 1]               # anchor 高
    acx = anchors[:, 0] + 0.5 * aw                   # anchor 中心
    acy = anchors[:, 1] + 0.5 * ah
    gcx = deltas[:, 0] * bbox_std[0] * aw + acx      # 先乘回 std 再还原中心 x（encode 的逆）
    gcy = deltas[:, 1] * bbox_std[1] * ah + acy      # 先乘回 std 再还原中心 y
    gw = torch.exp(deltas[:, 2] * bbox_std[2]) * aw  # 先乘回 std 再 exp 还原宽
    gh = torch.exp(deltas[:, 3] * bbox_std[3]) * ah  # 先乘回 std 再 exp 还原高
    return torch.stack([gcx - gw / 2, gcy - gh / 2, gcx + gw / 2, gcy + gh / 2], dim=1) # 还原应该输出真实框 xyxy