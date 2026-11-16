# 文件：config/config.py
# 核心作用：全项目唯一超参数来源（路径 / 输入尺寸 / batch / lr / seed / anchor 参数），对应 mmdet 的 configs/_base_
# 约定：所有路径用 root 拼接，禁止写死绝对路径；换机器只改 data_root 一行
# 调用位置：data/voc.py、trainer/train.py、trainer/eval.py、tools/predict.py 导入本文件参数
# 命名约定：参数用小写（img_size / batch_size ...），不是常量；utils/misc.py 已按“小写命名”收集，勿改成全大写
# 调用关系：data/voc.py、models/*.py、losses/loss.py、trainer/*.py、tools/predict.py 全部从这里 import
from pathlib import Path

root = Path(__file__).resolve().parent.parent              # w4_retinanet_voc\ 的绝对路径
data_root = root / "data" / "VOCdevkit" / "VOC2007"        # VOC2007 根目录（步骤 1 下到项目内，随项目走，换机器不用改）
split_dir = root / "data" / "splits"                       # train.txt / val.txt（步骤 8）
ckpt_dir = root / "checkpoints"                            # 权重输出目录
run_dir = root / "runs"                                    # TensorBoard 日志目录

img_size = 448            # 输入边长：EXP-004 由 320 提到 448，专治小目标（pottedplant/bottle 只有十几个像素）
batch_size = 1            # 448 下 batch 必须降到 1：单张像素是 320 的 1.96 倍，显存占用≈原来 320+batch2
num_workers = 0           # Windows 先设 0，避免多进程句柄报错
epochs = 6                # 448 单轮计算量≈320 的 2 倍，6 轮≈原来 12 轮的总计算量，耗时基本持平
lr = 1e-4                 # AdamW 学习率（预训练 backbone 微调常用量级）
weight_decay = 1e-4       # 权重衰减，防过拟合
num_classes = 20          # VOC 20 类（不含背景，用 sigmoid 多标签）
fpn_channels = 256        # FPN 每层统一通道数

# --- anchor：全部对齐 mmdetection RetinaNet 的默认值，不自己拍脑袋 ---
anchor_strides = (8, 16, 32)      # P3/P4/P5 的下采样倍数，必须与 backbone 输出一致
anchor_ratios = (0.5, 1.0, 2.0)   # 高宽比：扁 / 方 / 高（mmdet 默认 ratios）
octave_base_scale = 4             # 每层基准边长 = stride × 4（mmdet 默认 octave_base_scale）
scales_per_octave = 3             # 每层 3 个尺度，乘 2^(i/3)（mmdet 默认 scales_per_octave）
bbox_std = (0.1, 0.1, 0.2, 0.2)   # 回归目标归一化 std（mmdet DeltaXYWHBBoxCoder 默认 target_stds）

pos_iou = 0.5             # IoU ≥ 0.5 → 正样本
neg_iou = 0.4             # IoU < 0.4 → 负样本；0.4~0.5 之间"忽略"
score_thr = 0.05          # 推理时先滤低分，减少 NMS 负担
nms_thr = 0.5             # NMS 的 IoU 阈值
topk_per_img = 100        # 每张图最终最多保留 100 个框
seed = 42                 # 随机种子（任务书强制要求固定）