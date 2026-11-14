# 文件：models/backbone.py
# 核心作用：ResNet-18 取 C3/C4/C5 三层特征 + FPN 融合，输出 P3/P4/P5（通道统一 256）
# 调用关系：models/retinanet.py 的 RetinaNet.__init__ 实例化 -> RetinaNet.forward 中先调用本文件
# 对应参考：mmdet mmdet/models/backbones/resnet.py + mmdet/models/necks/fpn.py；
#           detectron2 detectron2/modeling/backbone/resnet.py + fpn.py
# 预期输出形状（IMG_SIZE=320）：P3 (B,256,40,40)、P4 (B,256,20,20)、P5 (B,256,10,10)
# 状态：占位文件，待实现
