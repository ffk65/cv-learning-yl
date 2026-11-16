# 文件：losses/loss.py
# 核心作用：检测的损失与正负样本定义：IoU 分配 + Focal Loss（分类）+ smooth L1（回归，仅正样本）
# 调用关系：trainer/train.py 训练循环 -> compute_loss(cls_scores, bbox_preds, anchors, gt_boxes, gt_labels)
# 对应参考：mmdet mmdet/models/task_modules/assigners/max_iou_assigner.py + models/losses/focal_loss.py；
#           detectron2 detectron2/modeling/matcher.py + detectron2/layers/sigmoid_focal_loss
# 计划实现：assign_targets()（pos_iou=0.5 / neg_iou=0.4）、sigmoid_focal_loss()（gamma=2.0、alpha=0.25）、
#           smooth_l1_loss()（beta=1/9）




# 文件：losses/loss.py
# 核心作用：① 给 18900 个 anchor 分配正/负/忽略；② 算 Focal Loss（分类）+ smooth L1（回归）
# 调用关系：trainer/train.py 的 train_one_epoch() 调 Loss.forward()
#          依赖 utils/box.py 的 box_iou()、models/anchor.py 的 encode()
import torch                                         # 张量
import torch.nn as nn                                # Module
import torch.nn.functional as F                      # 损失函数
from utils.box import box_iou                        # IoU 矩阵
from models.anchor import encode                     # 框编码
from config.config import pos_iou, neg_iou, num_classes   # 阈值与类别数


def assign_targets(anchors, gt_boxes, gt_labels):
    """功能：为一张图的所有 anchor 分配类别标签与回归目标偏移，用来loss
    输入：anchors (N,4)；gt_boxes (M,4)；gt_labels (M,)
    输出：cls_t (N,C) float（正样本对应类=1，其余全 0）、reg_t (N,4)、pos (N,) bool、ignore (N,) bool
    调用位置：Loss.forward()（本文件下方）
    """
    n = anchors.shape[0]                             # anchor 数 18900
    device = anchors.device                          # 统一设备
    # 边界情况：这张图没有目标（VOC 里极少，但必须处理，否则 max() 会报错）
    # 返回全 0 分类、全 0 回归、全 False 正样本、全 False 忽略, 所有 anchor 都是负样本，都不参与 loss
    if gt_boxes.numel() == 0:
        return (torch.zeros((n, num_classes), device=device),
                torch.zeros((n, 4), device=device),
                torch.zeros((n,), dtype=torch.bool, device=device),
                torch.zeros((n,), dtype=torch.bool, device=device))
    # N = 18900肯定大于 M
    # 1. 找出与 每个 anchor Iou最匹配的一个 GT。输出：一维元组max_iou = [0.7, 0.8,···], N个值
    ious = box_iou(anchors, gt_boxes)                # 每个 anchor 与每个 GT 的 IoU。(N,M)：NxM 矩阵--N个小元组
    max_iou, idx = ious.max(dim=1)                   # 每个 anchor 最匹配的 GT 及其 IoU。max_iou: [N]
    ''' gt_boxes：[2,4]
        ious = [
            [0.4, 0.3],   # anchor0
            [0.8, 0.2],   # anchor1
            [0.1, 0.2]    # anchor2
            [0.1, 0.2]    # anchor3
             ···
            ]
        max_iou = [0.4, 0.8, 0.2, 0.2, ···]
        idx     = [0,   0,   1,   1,   ···], 表示每个 anchor 与哪个 GT(值) 最匹配
    '''
    # 2. IoU 转为正/负样本
    pos = max_iou >= pos_iou                         # IoU ≥0.5 → 正样本。 pos: [N] = [1,0,1,0,0,0,1,1,1...]
    '''anchor的属性 pos = [false,true,false,false,···]  只有第二个anchor转正，并且和第一个GT匹配   '''
    # 3. 兜底：每个 GT 至少给它"最匹配的那个 anchor"一个正样本，避免小目标被漏, M个值。 GT1与第四个框最匹配，但是第四个框最大IOU也比较小被转负了
    best_anchor = ious.argmax(dim=0)                 # 每个 GT 的最佳 anchor。best_anchor: [M] = [4,1,2]
    pos[best_anchor] = True                          # 强制设为正样本
    '''GT的属性 best_anchor = [1,0], 第一个GT还是和第二个anchor最匹配，把第二个anchor正仍转正；第二个 GT 和第一个 anchor 最匹配，把第一个anchor 从负转正 
       best_anchor值代表第 idx 个 anchor
       此处idx匹配需要更改为idx[1]=0,idx[0]=1,idx序号代表第几个anchor，值代表第几个GT
       pos[0]更改为true
     '''
    # 关键：被强制的 anchor 可能原本匹配到别的 GT，这里把它的目标改回"它服务的那个 GT"
    idx[best_anchor] = torch.arange(gt_boxes.shape[0], device=device)   # =[0,1,2,3,···]GT顺序
    '''anchor的属性 idx = [1,0,1,1,···]  每个 anchor 对应的 GT 的下标  '''
    # 4. 上面得到pos每个anchor的正负，idx每个anchor对应的GT序号，下面matched得到每个anchor对应的GT坐标
    matched = gt_boxes[idx]                          
    '''gt_boxes = torch.tensor([[10,10,50,50], [60,60,100,100]]) ，gt_labels = torch.tensor([2, 5])   # gt0类别2，gt1类别5
       anchor的属性：matched: [          通过 idx 算 每个 anchor 对应的 GT 框坐标
                             [60,60,100,100],
                             [10,10,50,50],
                             [60,60,100,100],
                             [60,60,100,100]
                             ]
             '''
    cls_t = torch.zeros((n, num_classes), device=device)          # 分类目标：多标签 one-hot
    ''' cls_t = [                        每个anchor的20类的概率
                [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # anchor1
                [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # anchor2
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # anchor3
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # anchor4
                ···
                ]
       pos = [True, True, False, False]         
    '''
    cls_t[pos, gt_labels[idx[pos]]] = 1.0                          # 正样本：对应类置 1
    ''' 真实样本有2个GT框,序号0类别2, 序号2类别5
        pos中a个正样本anchor序号:0,1,···(5,200,18900)---分别得到匹配的a个GT序号1,0,···(1,0,1,1,0)---
        GT0对应gt_labels[0] = 类别2,
        结果: 正样本第0个anchor对应第一个GT框--类别5,正样本第1个anchor对应第0个GT--类别2概率置1
        每个正样本anchor得到一个类别的概率
    '''
    reg_t = torch.zeros((n, 4), device=device)                     # 回归目标，4个偏移
    if pos.any():                                                   # 只有正样本才有回归目标
        reg_t[pos] = encode(anchors[pos], matched[pos])             # 调 models/anchor.py，将正样本的4个偏移量插入(初始anchor和anchor匹配的GT框坐标)

    ignore = (max_iou >= neg_iou) & (~pos)           # 0.4~0.5 之间：不参与 loss
    '''ignore = [False, False, False, False], 没有一个样本IOU在之间'''
    return cls_t, reg_t, pos, ignore # 返回18900个框的20个类别概率（0，1），18900个框的4个偏移，只有正样本的概率和偏移才插入


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):   # 【标准】paper 值；【可改】gamma 1.5~2.5、alpha 0.25~0.5（类别极不均衡时才动）
    """功能：Focal Loss（逐元素，不求和），公式 FL = -α_t (1-p_t)^γ log p_t
    输入：logits 任意形状；targets 同形状（0/1 软标签）
    输出：同形状的逐元素 loss
    调用位置：Loss.forward()
    """
    p = torch.sigmoid(logits)                                      # 预测概率
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")   # 逐元素交叉熵
    p_t = p * targets + (1 - p) * (1 - targets)                    # 取"对应那一边"的概率
    loss = ce * (1 - p_t).pow(gamma)                               # 关键：易分样本 ×(1-p_t)^2；gamma 越大对易分样本压制越强
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)        # 类别权重 α
    return alpha_t * loss                                          # 返回逐元素值


class Loss(nn.Module):
    """功能：总损失 = (Focal Loss + smooth L1) / 正样本数
    输入：cls (B,N,C)、reg (B,N,4)、anchors (N,4)、gt_boxes(list[M,4])、gt_labels(list[M])
    输出：(loss 标量, stats 字典)
    调用位置：trainer/train.py 的 train_one_epoch()
    """

    def forward(self, cls, reg, anchors, gt_boxes, gt_labels):
        # 用 0 维张量累加（不能写 0.0：若整批没有正样本，float 没有 .item() 会报错）
        cls_sum = torch.zeros((), device=cls.device)  # 累计分类损失
        reg_sum = torch.zeros((), device=cls.device)  # 累计回归损失
        num_pos = 0                                  # 累计正样本数（用于归一化）
        for b in range(cls.shape[0]):                # 逐张图（框数不同，不能向量化）
            cls_t, reg_t, pos, ignore = assign_targets(anchors, gt_boxes[b], gt_labels[b])
            valid = ~ignore                          # 参与分类的 anchor（正 + 真负）
            cls_sum = cls_sum + sigmoid_focal_loss(cls[b][valid], cls_t[valid]).sum()
            if pos.any():                            # 有正样本才算回归
                # 【标准】beta=1/9 与 mmdet SmoothL1Loss 默认一致；【可改】0.1 对异常框更宽容，1.0 接近纯 L1
                reg_sum = reg_sum + F.smooth_l1_loss(reg[b][pos], reg_t[pos],
                                                     reduction="sum", beta=1 / 9)
                num_pos += int(pos.sum().item())
        num_pos = max(num_pos, 1)                    # 防止除 0
        total = (cls_sum + reg_sum) / num_pos        # RetinaNet 原文：用正样本数归一化
        stats = {"cls": cls_sum.item() / num_pos, "reg": reg_sum.item() / num_pos, "num_pos": num_pos}
        return total, stats