# 文件：models/anchor.py
# 核心作用：在 P3/P4/P5 上生成多尺度 anchor，并实现 anchor ↔ 预测框的编码 / 解码
# 调用关系：models/retinanet.py 前向之后 -> losses/loss.py 计算回归目标；trainer/eval.py 解码预测框
# 对应参考：mmdet mmdet/models/task_modules/prior_generators/anchor_generator.py + coders/delta_xywh_bbox_coder.py；
#           detectron2 detectron2/modeling/anchor_generator.py + box2box_transform.py
# 计划参数：strides=[8,16,32]，scales=[4*2^(0,1/3,2/3)]（相对 stride），ratios=[0.5,1,2] -> 每位置 9 个
# 计划数量（IMG_SIZE=320）：1600×9 + 400×9 + 100×9 = 18900 个/图
# 状态：占位文件，待实现
