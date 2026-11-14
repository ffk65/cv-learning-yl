# 文件：tools/predict.py
# 核心作用：单图推理：读图 -> 预处理 -> RetinaNet 前向 -> 解码 + NMS -> 在图上画框并保存
# 调用关系：python tools/predict.py --img <图片路径>
#           -> models/retinanet.py + models/anchor.py（解码）+ utils/box.py（NMS）
# 对应参考：mmdet mmdet/apis/inference.py；detectron2 demo/demo.py
# 状态：占位文件，待实现
