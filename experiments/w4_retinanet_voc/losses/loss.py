# 文件：losses/loss.py
# 核心作用：检测的损失与正负样本定义：IoU 分配 + Focal Loss（分类）+ smooth L1（回归，仅正样本）
# 调用关系：trainer/train.py 训练循环 -> compute_loss(cls_scores, bbox_preds, anchors, gt_boxes, gt_labels)
# 对应参考：mmdet mmdet/models/task_modules/assigners/max_iou_assigner.py + models/losses/focal_loss.py；
#           detectron2 detectron2/modeling/matcher.py + detectron2/layers/sigmoid_focal_loss
# 计划实现：assign_targets()（pos_iou=0.5 / neg_iou=0.4）、sigmoid_focal_loss()（gamma=2.0、alpha=0.25）、
#           smooth_l1_loss()（beta=1/9）
# 状态：占位文件，待实现
