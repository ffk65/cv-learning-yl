# 文件：data/make_split.py
# 核心作用：把 VOC2007 的 ImageSets/Main/trainval.txt 按 8:2 切成 train.txt / val.txt，供 Dataset 读取
# 调用关系：独立脚本（python -m data.make_split）；产物 data/splits/*.txt 被 data/voc.py 的 VOCDetection.__init__ 读取
# 输入：ImageSets/Main/trainval.txt（5011 个图片 id）
# 输出：data/splits/train.txt（约 4009 行）、data/splits/val.txt（约 1002 行）
# 状态：占位文件，待实现
