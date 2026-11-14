# 文件：utils/metric.py
# 核心作用：计算 VOC 口径的 AP / mAP@0.5（按类别、全点插值）
# 调用关系：trainer/eval.py 收集完所有预测后 -> voc_map()；不参与训练
# 对应参考：mmdet mmdet/evaluation/metrics/voc_metric.py；detectron2 detectron2/evaluation/pascal_voc_evaluation.py
# 输入：每张图的预测框（xyxy + score + label）与标注框
# 输出：20 类 AP、mAP
# 状态：占位文件，待实现
