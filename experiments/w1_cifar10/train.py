"""
train.py —— 第 1 周 CIFAR-10 实验的训练脚本（只负责“训练 + 验证”，不碰测试集）

核心作用：
  1) 从 configs/*.yaml 读取全部超参（不再写死在代码里）；
  2) 用 45k 训练集训练，用 5k 验证集挑选 best 权重并早停；
  3) 产物全部写进 experiments/w1_cifar10/outputs/<run_name>/；
  4) 日志同时打到终端与 logs/train.log（不再需要 Tee-Object / 输出重定向）。

调用关系：
  train.py  →  import common（路径/数据/模型/配置/日志工具）
  train.py  →  common.build_datasets → common.stratified_split / common.TransformSubset
  train.py  →  common.build_loader、common.build_model、common.resolve_run_dir、common.setup_logger
  eval.py   →  读取本脚本产出的 outputs/<run>/ckpt/best.pth 与 outputs/<run>/config.yaml

运行方式（在 experiments/w1_cifar10 目录下）：
  python train.py --config resnet.yaml
  python train.py --config cnn.yaml
"""

# ---------- 标准库 ----------
import argparse                       # 解析命令行参数
import csv                            # 写 metrics.csv
import time                           # 统计每轮耗时
from pathlib import Path              # 路径处理

# ---------- 第三方库 ----------
import torch                          # 深度学习框架
import yaml                           # 把最终配置原样存一份到 run 目录
from torch import nn                  # 损失函数
from torch.utils.tensorboard import SummaryWriter   # TensorBoard 记录器

# ---------- 本项目公共库（同目录下的 common.py） ----------
from common import (                  # noqa: E402  （import 顺序说明：common 依赖第三方库，放在其后）
    CLASS_NAMES,                      # 类别名列表（10 类）
    OUTPUTS_DIR,                      # outputs 根目录
    build_datasets,                   # 构建 train/val/test 数据集
    build_loader,                     # DataLoader 包装
    build_model,                      # 模型工厂
    get_device,                       # 设备选择
    load_config,                      # 读取 YAML 配置
    resolve_data_root,                # data_root -> 绝对路径
    resolve_run_dir,                  # 创建并返回 run 输出目录
    save_json,                        # 写 summary.json
    set_seed,                         # 固定随机种子
    setup_logger,                     # 终端 + 文件日志
)


def train_one_epoch(model, loader, loss_fn, optimizer, device, logger,
                    epoch: int, log_every: int = 50, limit_batches: int = 0):
    """
    功能：在训练集上跑一个完整 epoch（前向 -> 反向 -> 更新参数）。
    输入参数：
        model        —— nn.Module，已在 device 上。
        loader       —— 训练集 DataLoader（shuffle=True）。
        loss_fn      —— 损失函数（交叉熵）。
        optimizer    —— 优化器。
        device       —— torch.device。
        logger       —— logging.Logger，用于打日志。
        epoch        —— int，当前轮次（从 1 开始），只用于日志显示。
        log_every    —— int，每多少个 batch 打一行进度日志。
        limit_batches—— int，>0 时只跑前若干个 batch（冒烟测试用），0 表示跑完整个 epoch。
    输出参数：
        (avg_loss, acc) —— float，按“样本数”加权的平均损失与训练准确率。
    调用位置：
        train.py 的 main() 训练循环中，每个 epoch 调用一次。
    """
    # 切换到训练模式：启用 Dropout / BatchNorm 的训练行为
    model.train()
    # 累计损失（用 sum 而不是 mean，最后除以样本数，避免最后一个 batch 尺寸不同导致偏差）
    running_loss = 0.0
    # 累计预测正确的样本数
    correct = 0
    # 累计已处理的样本数（作为准确率分母，也是损失的加权分母）
    seen = 0
    # 记录本次 epoch 实际跑过的 batch 数（冒烟测试时会小于 len(loader)）
    batches = 0

    # 遍历训练集
    for batch_idx, (images, labels) in enumerate(loader):
        # 冒烟测试：跑够 limit_batches 个 batch 就停
        if limit_batches and batch_idx >= limit_batches:
            break
        # 把数据搬到 GPU/CPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 清空上一步的梯度
        optimizer.zero_grad()
        # 前向：得到每个类别的 logits
        logits = model(images)
        # 计算损失
        loss = loss_fn(logits, labels)
        # 反向传播：计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        # 本 batch 的样本数（最后一个 batch 可能不足 batch_size）
        batch_size = labels.size(0)
        # 累加损失（loss.item() 是该 batch 的平均损失，需要乘样本数还原为总和）
        running_loss += loss.item() * batch_size
        # 统计预测正确数
        correct += (logits.argmax(dim=1) == labels).sum().item()
        # 统计已处理样本数
        seen += batch_size
        # batch 计数
        batches += 1

        # 每 log_every 个 batch 打一行进度（不是每个 batch 都刷屏）
        if log_every and batch_idx % log_every == 0:
            logger.info(
                "  epoch %d | batch %d/%d | loss %.4f | acc %.4f",
                epoch, batch_idx, len(loader), loss.item(), correct / max(seen, 1),
            )

    # 平均损失：总和 / 样本数
    avg_loss = running_loss / max(seen, 1)
    # 训练准确率：正确数 / 样本数
    acc = correct / max(seen, 1)
    # 返回 (平均损失, 准确率)，同时日志里补充实际跑了多少 batch
    logger.info("  epoch %d 训练完成：%d 个 batch，loss %.4f，acc %.4f", epoch, batches, avg_loss, acc)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, limit_batches: int = 0):
    """
    功能：在验证集（或测试集）上评估，返回平均损失与准确率；不计算梯度、不更新参数。
    输入参数：
        model        —— nn.Module，已在 device 上。
        loader       —— 评估集 DataLoader（shuffle=False）。
        loss_fn      —— 损失函数。
        device       —— torch.device。
        limit_batches—— int，>0 时只评估前若干个 batch（冒烟测试用）。
    输出参数：
        (avg_loss, acc) —— float。
    调用位置：
        train.py 的 main()：每个 epoch 用验证集调用；
        eval.py 中不使用本函数（eval 需要逐样本标签画混淆矩阵，故单独实现）。
    """
    # 切换到评估模式：关闭 Dropout，BatchNorm 用滑动统计量
    model.eval()
    # 累计损失
    running_loss = 0.0
    # 累计正确数
    correct = 0
    # 累计样本数
    seen = 0

    # 遍历评估集
    for batch_idx, (images, labels) in enumerate(loader):
        # 冒烟测试的 batch 上限
        if limit_batches and batch_idx >= limit_batches:
            break
        # 搬到设备
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # 前向
        logits = model(images)
        # 损失
        loss = loss_fn(logits, labels)
        # 累加损失与正确数
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch_size

    # 返回平均损失与准确率
    return running_loss / max(seen, 1), correct / max(seen, 1)


def build_optimizer(model, opt_cfg: dict):
    """
    功能：按配置创建优化器。
    输入参数：
        model   —— nn.Module，提供 parameters()。
        opt_cfg —— dict，含 name / lr / momentum / weight_decay。
    输出参数：
        torch.optim.Optimizer。
    调用位置：
        train.py 的 main()，在模型建好之后调用。
    """
    # 优化器名统一小写
    name = str(opt_cfg.get("name", "sgd")).lower()
    # 学习率、动量、权重衰减
    lr = float(opt_cfg["lr"])
    momentum = float(opt_cfg.get("momentum", 0.9))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    # SGD：与原实验一致的默认选择
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # AdamW：后续调参时可直接切换
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 未知名字直接报错
    raise ValueError(f"未知优化器: {name}，可选 sgd / adamw")


def build_scheduler(optimizer, sched_cfg: dict, epochs: int):
    """
    功能：按配置创建学习率调度器。
    输入参数：
        optimizer  —— 优化器。
        sched_cfg  —— dict，含 name。
        epochs     —— int，总轮数；余弦退火的 T_max 必须等于它（这是修复旧代码 T_max=50 / 跑 100 轮的关键）。
    输出参数：
        torch 调度器 或 None（name 为 none 时）。
    调用位置：
        train.py 的 main()；每个 epoch 结束后调用 scheduler.step()。
    """
    # 调度器名统一小写
    name = str(sched_cfg.get("name", "cosine")).lower()
    # 余弦退火：学习率按余弦曲线从 lr 降到 0，T_max 设为总轮数
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # 不调度
    if name == "none":
        return None
    # 未知名字直接报错
    raise ValueError(f"未知调度器: {name}，可选 cosine / none")


def save_summary(run_dir: Path, config: dict, best: dict, last_epoch: int,
                 last_val_acc: float, args, logger) -> None:
    """
    功能：训练结束后写 summary.json / summary.md，记录“这次实验到底做了什么、结果如何”。
    输入参数：
        run_dir      —— Path，run 输出目录。
        config       —— dict，本次使用的完整配置。
        best         —— dict，含 best_epoch / best_val_acc / best_val_loss。
        last_epoch   —— int，实际跑到第几轮（早停时可能小于 epochs）。
        last_val_acc —— float，最后一轮的验证准确率。
        args         —— argparse.Namespace，用于记录启动命令。
        logger       —— logging.Logger。
    输出参数：
        无（写两个文件）。
    调用位置：
        train.py 的 main() 结尾。
    """
    # 机器可读的 summary（后续可脚本批量对比多个 run）
    summary = {
        "run_name": run_dir.name,
        "model": config["model"]["name"],
        "epochs_planned": config["epochs"],
        "epochs_run": last_epoch,
        "best_epoch": best["best_epoch"],
        "best_val_acc": best["best_val_acc"],
        "best_val_loss": best["best_val_loss"],
        "last_val_acc": last_val_acc,
        "config": config,
        "command": f"python train.py --config {args.config}"
                   f"{' --run-name ' + args.run_name if args.run_name else ''}"
                   f"{' --limit-batches ' + str(args.limit_batches) if args.limit_batches else ''}"
                   f"{' --limit-epochs ' + str(args.limit_epochs) if args.limit_epochs else ''}",
    }
    # 写 JSON
    save_json(run_dir / "summary.json", summary)

    # 人读的 summary.md（这就是任务书要求的“结论记录”，必须入库）
    lines = [
        f"# 训练小结：{run_dir.name}",
        "",
        f"- 模型：`{config['model']['name']}`（预训练：{config['model'].get('pretrained', False)}）",
        f"- 计划轮数：{config['epochs']}，实际轮数：{last_epoch}",
        f"- 最佳验证准确率：**{best['best_val_acc']:.4f}**（第 {best['best_epoch']} 轮）",
        f"- 最后一轮验证准确率：{last_val_acc:.4f}",
        f"- 权重：`ckpt/best.pth`（只由验证集选出，测试集未参与选模）",
        "",
        "## 配置",
        "",
        "```yaml",
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False).strip(),
        "```",
        "",
        "## 产物",
        "",
        "| 路径 | 内容 |",
        "| --- | --- |",
        "| `ckpt/best.pth` | 验证集最优权重 + 元信息 |",
        "| `metrics/metrics.csv` | 每轮 train/val 指标 |",
        "| `tb/` | TensorBoard 事件文件 |",
        "| `logs/train.log` | 完整训练日志 |",
        "| `config.yaml` | 本次运行配置副本 |",
        "",
        "## 下一步",
        "",
        f"```powershell",
        f"python eval.py --run {run_dir.name}",
        f"```",
    ]
    # 写 Markdown
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 日志提示
    logger.info("已写出 summary：%s 与 %s", run_dir / "summary.json", run_dir / "summary.md")


def main():
    """
    功能：训练主流程（解析参数 -> 读配置 -> 建数据 -> 建模型 -> 训练循环 -> 写结果）。
    输入参数：
        无（全部来自命令行）。
    输出参数：
        无（结果落盘到 outputs/<run_name>/）。
    调用位置：
        本文件底部 if __name__ == "__main__": main()。
    """
    # ---------- 1. 命令行参数 ----------
    parser = argparse.ArgumentParser(description="CIFAR-10 训练（train 45k + val 5k，test 不参与）")
    # 配置文件：相对 configs/ 的文件名即可
    parser.add_argument("--config", type=str, default="resnet.yaml", help="配置文件名，如 cnn.yaml / resnet.yaml")
    # run 名称：决定 outputs/<run_name>/，不填则用配置里的 run_name
    parser.add_argument("--run-name", type=str, default=None, help="本次 run 名称（默认取配置里的 run_name）")
    # 是否覆盖同名 run
    parser.add_argument("--overwrite", action="store_true", help="同名 run 目录已存在时清空重跑")
    # 冒烟测试：只跑前 N 个 batch
    parser.add_argument("--limit-batches", type=int, default=0, help=">0 时每轮只跑前 N 个 batch（冒烟测试）")
    # 冒烟测试：只跑前 N 轮（0 = 按配置的 epochs 跑完）
    parser.add_argument("--limit-epochs", type=int, default=0, help=">0 时只跑前 N 轮（冒烟测试）")
    # 解析参数
    args = parser.parse_args()

    # ---------- 2. 读取配置 + 固定种子 ----------
    config = load_config(args.config)          # 读取 YAML（相对 configs/ 解析）
    set_seed(int(config["seed"]))              # 固定随机种子（必须在建数据、建模型之前）

    # ---------- 3. 建 run 目录 + 日志 ----------
    run_name = args.run_name or config.get("run_name") or config["model"]["name"]
    run_dir = resolve_run_dir(run_name, overwrite=args.overwrite)    # 建目录（含 ckpt/tb/logs/figures/metrics）
    logger = setup_logger(run_dir / "logs" / "train.log", name="cifar10_train")   # 终端 + 文件
    logger.info("配置：%s", args.config)
    logger.info("run 目录：%s", run_dir)
    logger.info("完整配置：\n%s", yaml.safe_dump(config, allow_unicode=True, sort_keys=False).strip())

    # 把本次配置原样复制进 run 目录：eval.py 靠它知道“该建哪个模型”
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    # ---------- 4. 设备 ----------
    device = get_device(logger)                # 传入 logger，设备信息才会写进 train.log

    # ---------- 5. 数据（45k train / 5k val，test 只建不用） ----------
    data_root = resolve_data_root(config)      # data/cifar10 -> 绝对路径
    logger.info("数据目录：%s", data_root)
    train_set, val_set, test_set = build_datasets(
        data_root=data_root,
        val_size=int(config["val_size"]),
        num_classes=int(config["num_classes"]),
        seed=int(config["seed"]),
        need_train=True,
    )
    # 打印三个集合的样本数，确认 45000 / 5000 / 10000
    logger.info("样本数：train=%d，val=%d，test=%d（test 只在 eval.py 用）",
                len(train_set), len(val_set), len(test_set))

    # DataLoader：训练打乱，验证固定顺序
    train_loader = build_loader(train_set, int(config["batch_size"]), shuffle=True,
                                num_workers=int(config["num_workers"]))
    val_loader = build_loader(val_set, int(config["batch_size"]), shuffle=False,
                              num_workers=int(config["num_workers"]))
    logger.info("train batches=%d，val batches=%d", len(train_loader), len(val_loader))

    # ---------- 6. 模型 / 损失 / 优化器 / 调度器 ----------
    model = build_model(config).to(device)                 # 建网络并搬到 GPU/CPU
    n_params = sum(p.numel() for p in model.parameters())  # 统计参数量，便于对比模型大小
    logger.info("模型：%s，参数量 %.2fM", config["model"]["name"], n_params / 1e6)

    loss_fn = nn.CrossEntropyLoss()                        # 分类任务标准损失
    optimizer = build_optimizer(model, config["optimizer"])           # 按配置建优化器
    epochs = int(config["epochs"])                        # 配置里的计划轮数
    # 冒烟测试时只跑前 N 轮；方便快速验证整条流水线（产物路径、日志、TensorBoard）
    total_epochs = args.limit_epochs if args.limit_epochs > 0 else epochs
    scheduler = build_scheduler(optimizer, config["scheduler"], total_epochs)   # T_max=实际轮数
    patience = int(config["early_stopping_patience"])      # 早停耐心值

    # TensorBoard：事件文件写到 outputs/<run>/tb
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # ---------- 7. 指标文件（csv 表头） ----------
    metrics_path = run_dir / "metrics" / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_sec"])

    # ---------- 8. 训练循环 ----------
    best = {"best_epoch": 0, "best_val_acc": 0.0, "best_val_loss": float("inf")}   # 记录最优
    wait = 0                                                                        # 连续未提升轮数
    last_epoch, last_val_acc = 0, 0.0                                               # 最后一轮结果

    for epoch in range(1, total_epochs + 1):
        # 记录本轮使用的学习率（scheduler.step 之后会变化，所以先存）
        lr_used = optimizer.param_groups[0]["lr"]
        # 计时开始
        t0 = time.perf_counter()
        logger.info("----- 第 [%d/%d] 轮开始（lr=%.6f）-----", epoch, total_epochs, lr_used)

        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, logger,
            epoch=epoch,
            log_every=int(config.get("log_every", 50)),
            limit_batches=args.limit_batches,
        )
        # 在验证集上评估（选模型的唯一依据）
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device,
                                     limit_batches=args.limit_batches)

        # 调度器步进（余弦退火，每个 epoch 一次）
        if scheduler is not None:
            scheduler.step()

        # 本轮耗时
        time_sec = time.perf_counter() - t0
        # 记录最后一轮结果
        last_epoch, last_val_acc = epoch, val_acc

        # 终端 + 文件日志：一行看完本轮关键指标
        logger.info(
            "第 %d 轮结束 | train_loss %.4f | train_acc %.4f | val_loss %.4f | val_acc %.4f | %.1fs",
            epoch, train_loss, train_acc, val_loss, val_acc, time_sec,
        )

        # TensorBoard 曲线（loss/train、loss/val、acc/train、acc/val、lr）
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("lr", lr_used, epoch)

        # 追加一行到 metrics.csv
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr_used, time_sec])

        # ---------- 9. 保存最佳权重（依据：验证集准确率） ----------
        if val_acc > best["best_val_acc"]:
            best = {"best_epoch": epoch, "best_val_acc": val_acc, "best_val_loss": val_loss}
            wait = 0                                   # 有提升，重置耐心计数
            # 权重文件只存“数字 + 元信息”，不存代码；eval.py 用 build_model 重建结构后灌入
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": config["model"]["name"],
                    "num_classes": int(config["num_classes"]),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": config,
                },
                run_dir / "ckpt" / "best.pth",
            )
            logger.info("  ↑ 新的最佳验证准确率 %.4f，已保存 ckpt/best.pth", val_acc)
        else:
            wait += 1                                  # 没提升，耐心计数 +1
            logger.info("  验证准确率未提升（%d/%d）", wait, patience)
            # 连续 patience 轮不提升 -> 早停
            if wait >= patience:
                logger.info("触发早停：验证集连续 %d 轮未提升", patience)
                break

    # ---------- 10. 收尾 ----------
    writer.close()                                  # 关闭 TensorBoard，确保事件写完
    logger.info("训练结束：最佳验证准确率 %.4f（第 %d 轮）", best["best_val_acc"], best["best_epoch"])
    # 写 summary.json / summary.md
    save_summary(run_dir, config, best, last_epoch, last_val_acc, args, logger)
    # 提示下一步（这就是 eval 要用的命令）
    logger.info("下一步：python eval.py --run %s", run_dir.name)


# 只有直接运行本文件时才执行 main()；被 import 时不会启动训练
if __name__ == "__main__":
    main()
