# 文件：trainer/train.py
# 核心作用：训练入口：读 configs/config.py -> build_loader -> RetinaNet -> compute_loss -> 反向 -> 保存权重 -> TensorBoard
# 调用关系：python -m trainer.train（在 w4_retinanet_voc 目录下执行）
#           -> data/voc.py（数据）、models/retinanet.py（模型）、losses/loss.py（损失）
#           -> 产物 checkpoints/last.pth + runs/ 事件文件
# 对应参考：mmdet tools/train.py + mmengine Runner；detectron2 tools/train_net.py + DefaultTrainer
# 计划包含：build_loader()、train_one_epoch()、save_checkpoint()、main()
# 状态：占位文件，待实现



# 文件：trainer/train.py
# 核心作用：组装 数据/模型/损失/优化器 → 训练 → 写 TensorBoard → 存 checkpoint
# 运行：python -m trainer.train --epochs 2 --exp EXP-002
# 调用关系：data/voc.py（数据）→ models/retinanet.py（模型）→ losses/loss.py（损失）
import argparse                                      # 命令行参数
import random                                        # Python 随机种子
import time                                          # 时间戳（日志目录名）
import numpy as np                                   # NumPy 随机种子
import torch                                         # 张量
from torch.utils.data import DataLoader              # 数据加载器
from torch.utils.tensorboard import SummaryWriter    # TensorBoard
from config.config import (batch_size, ckpt_dir, epochs, lr, num_workers,
                            run_dir, seed, weight_decay)      # 超参
from data.voc import VOCDataset, collate_fn        # 数据集
from models.retinanet import RetinaNet, flatten_outputs   # 模型
from losses.loss import Loss                # 损失
from utils.misc import dump_run_config               # 实验 config 快照（12.5）


def set_seed(seed=seed):
    """功能：固定所有随机源，保证实验可复现（任务书硬要求）
    输入：seed（int）
    输出：无
    调用位置：main()
    """
    random.seed(seed)                                # Python
    np.random.seed(seed)                             # NumPy
    torch.manual_seed(seed)                          # CPU 上的 torch
    torch.cuda.manual_seed_all(seed)                 # 所有 GPU
    torch.backends.cudnn.deterministic = True        # cuDNN 确定性（略慢，但可复现）
    torch.backends.cudnn.benchmark = False


def build_loader(split, batch_size):
    """功能：构造 DataLoader
    输入：split ∈ {"train","val"}；batch_size
    输出：DataLoader
    调用位置：main()
    """
    ds = VOCDataset(split)                         # data/voc.py
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers, collate_fn=collate_fn,
                      drop_last=(split == "train"), pin_memory=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, writer):
    """功能：训练一个 epoch，每个epoch都是重复一轮所有图片
    输入：模型/数据/损失/优化器/AMP scaler/设备/轮次/日志器
    输出：本 epoch 平均 loss（float）
    调用位置：main() 的 for epoch 循环
    """
    model.train()                                    # 切训练模式（BN/Dropout 生效）
    # 注意：model.train() 会把 BN 全部切回训练态，所以这里要把 stem 的 BN 再冻结一次
    for m in model.backbone.stem.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()                                 # 只用 running_mean/var，不更新统计量
    total, n = 0.0, 0                                # 累计 loss 与步数
    for step, (images, boxes, labels) in enumerate(loader):          # 4009/batch_size步（轮次）
        images = torch.stack(images).to(device, non_blocking=True)   # list → (B,3,320,320)
        boxes = [b.to(device) for b in boxes]                        # 框搬到 GPU
        labels = [l.to(device) for l in labels]                      # 标签搬到 GPU

        optimizer.zero_grad(set_to_none=True)                        # 清梯度
        with torch.cuda.amp.autocast():                              # 混合精度，省显存
            cls_list, reg_list, anchors = model(images)              # 前向
            cls, reg = flatten_outputs(cls_list, reg_list)           # 展平
            loss, stats = criterion(cls, reg, anchors, boxes, labels)  # 算损失

        scaler.scale(loss).backward()                                # 反向（带缩放）
        scaler.unscale_(optimizer)                                   # 还原梯度以便裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)     # 梯度裁剪，防爆炸
        scaler.step(optimizer)                                       # 更新参数
        scaler.update()                                              # 更新缩放系数

        total += loss.item()                                         # 累计
        n += 1
        if step % 100 == 0:                                           # 每 100 步打印
            print(f" {step}/{len(loader)}: loss={loss.item():.4f}, cls={stats['cls']:.4f}, reg={stats['reg']:.4f}, pos={stats['num_pos']}")
            writer.add_scalar("train/loss_step", loss.item(), epoch * len(loader) + step)
    return total / max(n, 1)                                         # 平均


def main():
    """功能：训练入口（解析参数 → 建模型 → 循环训练 → 存权重）
    输入：命令行 --epochs/--batch/--resume
    输出：checkpoints/last.pth + runs/ 日志
    调用位置：`python -m trainer.train`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=epochs)         # 轮数
    parser.add_argument("--batch", type=int, default=batch_size)      # batch 大小
    parser.add_argument("--exp", type=str, default=time.strftime("EXP-%m%d-%H%M"))  # 实验编号（15.2）
    parser.add_argument("--resume", type=str, default="")             # 断点续训（可选）
    args = parser.parse_args()

    set_seed()                                                        # 固定种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir.mkdir(parents=True, exist_ok=True)                       # 建目录
    run_dir.mkdir(parents=True, exist_ok=True)

    model = RetinaNet(pretrained=True).to(device)                     # 模型上 GPU
    # 6GB 优化：冻结 stem 里的 BN（不更新统计量），省显存且小 batch 更稳
    for m in model.backbone.stem.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.requires_grad_(False)

    criterion = Loss()                                       # 损失
    params = [p for p in model.parameters() if p.requires_grad]        # 可训练参数
    # 【标准】mmdet RetinaNet 1x 用 MultiStepLR：第 8、11 轮学习率 ×0.1（configs/_base_/schedules/schedule_1x.py）
    # 【可改】想要平滑衰减就换 CosineAnnealingLR(optimizer, T_max=args.epochs)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()                              # AMP
    exp_run_dir = run_dir / args.exp                                      # runs/EXP-002/
    writer = SummaryWriter(str(exp_run_dir))                              # 日志写到实验目录
    dump_run_config(exp_run_dir, {"epochs": args.epochs, "batch": args.batch})  # 落盘 config.json（实验日志要用）

    if args.resume:                                                   # 续训
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])

    loader = build_loader("train", args.batch)                        # 训练集
    for epoch in range(args.epochs):   
        print(f" ============= 第 {epoch+1}/{args.epochs}轮训练开始 ============= ")                               # 逐轮
        avg = train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, writer)
        scheduler.step()                                              # 每轮结束走一步：第 8、11 轮自动 ×0.1
        writer.add_scalar("train/loss_epoch", avg, epoch)             # 记录
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pth")  # 存"最新一轮"权重（覆盖写，不是最优；要 best.pth 见步骤 17）
        print(f"[第 {epoch+1} 轮训练结束] avg_loss={avg:.4f} -> saved checkpoints/last.pth")
    writer.close()                                                    # 关闭日志


if __name__ == "__main__":
    main()
    # tensorboard --logdir runs