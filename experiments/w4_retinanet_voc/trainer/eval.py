# 文件：trainer/eval.py
# 核心作用：评估入口：加载 checkpoints/best.pth -> 在 val/test 上推理 -> 解码 + NMS -> 计算 VOC mAP@0.5
# 调用关系：python -m trainer.eval --split val
#           -> data/voc.py（数据）、models/retinanet.py（模型）、models/anchor.py（解码）、utils/metric.py（mAP）
# 对应参考：mmdet tools/test.py；detectron2 tools/train_net.py --eval-only（build_evaluator）
# 状态：占位文件，待实现
