# 文件：utils/metric.py
# 核心作用：计算 VOC 口径的 AP / mAP@0.5（按类别、全点插值）
# 调用关系：trainer/eval.py 收集完所有预测后 -> voc_map()；不参与训练
# 对应参考：mmdet mmdet/evaluation/metrics/voc_metric.py；detectron2 detectron2/evaluation/pascal_voc_evaluation.py
# 输入：每张图的预测框（xyxy + score + label）与标注框
# 输出：20 类 AP、mAP
# 状态：占位文件，待实现


# 文件：utils/metric.py
# 核心作用：VOC 口径的 AP / mAP@0.5 计算
# 调用关系：trainer/eval.py 的 evaluate() 调 compute_map()
import numpy as np                                   # 数组运算
import torch                                         # 张量
from utils.box import box_iou                        # IoU 矩阵


def voc_ap(rec, prec):
    """功能：由 PR 序列算 AP（全点插值）
    输入：rec、prec —— numpy 数组，已按分数降序
    输出：float
    调用位置：compute_map()
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))       # 两端补点
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):            # 从右往左做"单调不增"包络
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]         # recall 发生变化的点
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))   # 面积求和


def compute_map(detections, gts, num_classes=20, iou_thr=0.5):
    """功能：对整个验证集算每类 AP 与 mAP
    输入：detections = list[dict(boxes(T,4), scores(T,), labels(T,))]（每张图的预测）
          gts        = list[dict(boxes(M,4), labels(M,))]（每张图的真值，顺序一致）
    输出：(mAP: float, aps: dict[类别id → AP])
    调用位置：trainer/eval.py 的 evaluate()
    """
    aps = {}                                         # 每类 AP
    for c in range(num_classes):                     # 逐类算
        scores_all, tp_all, npos = [], [], 0         # 该类所有图的预测分数 / TP 标记 / GT 总数
        for det, gt in zip(detections, gts):         # 逐图
            gmask = gt["labels"] == c                # 该类的 GT
            gboxes = gt["boxes"][gmask]
            npos += int(gboxes.shape[0])             # 累计该类 GT 数（分母）
            dmask = det["labels"] == c               # 该类的预测
            dboxes = det["boxes"][dmask]
            dscores = det["scores"][dmask]
            if dboxes.numel() == 0:                  # 该图没预测这个类 → 跳过
                continue
            order = dscores.argsort(descending=True) # 按分数降序
            dboxes, dscores = dboxes[order], dscores[order]
            # 与该图该类 GT 算 IoU（GT 为空时给 0 列，避免报错）
            ious = box_iou(dboxes, gboxes) if gboxes.numel() > 0 else torch.zeros((len(dboxes), 0))
            used = torch.zeros(len(gboxes), dtype=torch.bool)   # 标记该 GT 已被匹配
            for i in range(len(dboxes)):             # 逐预测框判 TP/FP
                if gboxes.numel() == 0:              # 没 GT → 必然 FP
                    tp = 0
                else:
                    j = int(ious[i].argmax())        # 最匹配的 GT
                    if ious[i, j] >= iou_thr and not used[j]:
                        used[j] = True               # 占住这个 GT
                        tp = 1
                    else:
                        tp = 0
                tp_all.append(tp)                    # 记录
                scores_all.append(float(dscores[i]))
        if npos == 0:                                # 该类在验证集没有 GT → 不参与 mAP
            continue
        order = np.argsort(-np.array(scores_all))    # 全局再按分数排序
        tp_arr = np.array(tp_all)[order].astype(np.float32)
        fp_arr = 1.0 - tp_arr
        tp_cum = np.cumsum(tp_arr)                   # 累计 TP
        fp_cum = np.cumsum(fp_arr)                   # 累计 FP
        rec = tp_cum / max(npos, 1)                  # Recall
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)   # Precision
        aps[c] = voc_ap(rec, prec)                   # 该类 AP
    mAP = float(np.mean(list(aps.values()))) if aps else 0.0
    return mAP, aps