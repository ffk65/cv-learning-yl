# 文件：utils/box.py
# 核心作用：框运算工具：IoU、框裁剪到图像边界、xyxy ↔ xywh 转换、解码后过滤
# 调用关系：losses/loss.py（IoU 分配）与 trainer/eval.py、tools/predict.py（解码 / NMS）
# 对应参考：detectron2 detectron2/modeling/box2box_transform.py + structures/boxes.py
# 约定：全项目统一使用 xyxy（左上角原点、x 向右、y 向下）；文件内不 import 任何项目内模块
# 状态：占位文件，待实现
