# 文件：data/voc.py
# 核心作用：把 VOC2007 的 (jpg + xml) 读成 (image[3,H,W], boxes[N,4] xyxy, labels[N])，并提供 collate_fn
# 调用关系：trainer/train.py、trainer/eval.py 的 build_loader() -> VOCDetection(...) + collate_fn
#          自检入口：python -m data.voc（打印形状 / 框 / 标签）
# 对应参考：mmdet mmdet/datasets/voc.py + xml_style.py；detectron2 detectron2/data/datasets/pascal_voc.py
# 状态：占位文件，待实现
