# 文件：models/retinanet.py
# 核心作用：整机 RetinaNet = ResNet-18 backbone + FPN neck + RetinaHead；forward 逐层输出 cls / reg
# 调用关系：trainer/train.py、trainer/eval.py、tools/predict.py 实例化 RetinaNet 并调用 forward
# 对应参考：mmdet mmdet/models/detectors/retinanet.py（整机）+ dense_heads/retina_head.py（头）；
#           detectron2 detectron2/modeling/meta_arch/retinanet.py（整机与头在同一个文件）
# 本文件内计划包含：class RetinaHead（4 层 3×3 卷积 + cls/reg 两个 3×3 输出层，分类 bias 初始化 π=0.01）
#                   class RetinaNet（三段式组装，forward 返回 [cls_scores, bbox_preds]）
# 状态：占位文件，待实现
