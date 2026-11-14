# 文件：configs/config.py
# 核心作用：全项目唯一超参数来源（路径 / 输入尺寸 / batch / lr / seed / anchor 参数），对应 mmdet 的 configs/_base_
# 约定：所有路径用 PROJECT_ROOT 拼接，禁止写死绝对路径；换机器只改 DATA_ROOT 一行
# 调用位置（计划）：data/voc.py、trainer/train.py、trainer/eval.py、tools/predict.py 导入本文件常量
# 状态：占位文件，按任务书 9.3–9.6 填充
