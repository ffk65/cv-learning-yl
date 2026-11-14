"""
eval.py —— 第 1 周 CIFAR-10 实验的评估脚本（测试集只在这里、只跑一次）

核心作用：
  1) 读取某个 run 的 config.yaml 与 ckpt/best.pth；
  2) 在 test（默认）或 val 上计算整体准确率、每类 precision/recall/f1；
  3) 画混淆矩阵、导出指标文件，全部写进该 run 的 figures/ 与 metrics/；
  4) 可选：对单张本地图片做一次推理（--img-path）。

调用关系：
  eval.py  →  import common（build_datasets / build_loader / build_model / load_checkpoint / ...）
  eval.py  →  common.build_datasets → common.stratified_split / common.TransformSubset
  eval.py  →  读取 train.py 产出的 outputs/<run>/config.yaml、outputs/<run>/ckpt/best.pth
  不再 import train.py（旧版 `from train import test_loader, device, select_model` 已删除，
  避免“评估脚本一 import 就下载数据、建 loader”的副作用）。

运行方式（在 experiments/w1_cifar10 目录下）：
  python eval.py --run cnn_baseline
  python eval.py --run resnet_baseline --img-path assets/plane.jpg
"""

# ---------- 标准库 ----------
import argparse                       # 命令行参数
import csv                            # 导出每类指标
from pathlib import Path              # 路径处理

# ---------- 第三方库 ----------
import matplotlib                     # 绘图（先导入并指定无界面后端，避免服务器/无 GUI 环境报错）
matplotlib.use("Agg")                 # Agg 后端：只写文件，不弹窗
import matplotlib.pyplot as plt       # 画混淆矩阵
import numpy as np                    # 数值计算
import torch                          # 框架
from PIL import Image                 # 读取本地图片
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch import nn                  # 损失函数
from torchvision import transforms    # 单张图片推理的预处理

# ---------- 本项目公共库 ----------
from common import (
    CLASS_NAMES,                      # 10 个类别名
    OUTPUTS_DIR,                      # outputs 根目录
    build_datasets,                   # 构建 val / test 数据集
    build_loader,                     # DataLoader
    build_model,                      # 按配置重建网络结构
    get_device,                       # 设备
    load_checkpoint,                  # 读取权重字典
    load_config,                      # 读取 run 目录里的 config.yaml
    resolve_data_root,                # data_root -> 绝对路径
    save_json,                        # 写 json 指标
    set_seed,                         # 固定种子（val 切分要可复现）
    setup_logger,                     # 终端 + 文件日志
)


def collect_predictions(model, loader, device, limit_batches: int = 0):
    """
    功能：遍历评估集，收集逐样本的真实标签与预测标签（画混淆矩阵必须逐样本，不能只算总体准确率）。
    输入参数：
        model        —— nn.Module，已加载权重并在 device 上。
        loader       —— 评估集 DataLoader。
        device       —— torch.device。
        limit_batches—— int，>0 时只评估前 N 个 batch（冒烟测试）。
    输出参数：
        (ys, ps) —— 两个 list[int]，分别是真实标签与预测标签。
    调用位置：
        eval.py 的 main() 中调用一次。
    """
    # 评估模式
    model.eval()
    # 真实 / 预测 标签列表
    ys, ps = [], []
    # 关闭梯度
    with torch.no_grad():
        # 逐 batch 推理
        for batch_idx, (images, labels) in enumerate(loader):
            # 冒烟测试上限
            if limit_batches and batch_idx >= limit_batches:
                break
            # 搬到设备
            images = images.to(device, non_blocking=True)
            # 前向 -> 取最大 logit 的下标作为预测类别
            preds = model(images).argmax(dim=1).cpu()
            # 累积标签（target 本身就是 CPU 张量，直接转 list）
            ys.extend(labels.tolist())
            ps.extend(preds.tolist())
    # 返回逐样本标签
    return ys, ps


def save_classification_report(run_dir: Path, split: str, ys, ps, logger) -> dict:
    """
    功能：计算每类 precision/recall/f1 与总体指标，导出 CSV 和 Markdown 表。
    输入参数：
        run_dir —— Path，run 输出目录。
        split   —— str，"test" 或 "val"，用于文件命名。
        ys      —— list[int]，真实标签。
        ps      —— list[int]，预测标签。
        logger  —— logging.Logger。
    输出参数：
        dict —— sklearn 的 classification_report(output_dict=True) 结果。
    调用位置：
        eval.py 的 main() 中，collect_predictions 之后调用。
    """
    # 计算分类报告（output_dict=True 便于写 CSV / JSON）
    report = classification_report(
        ys, ps,
        labels=list(range(len(CLASS_NAMES))),   # 固定 0~9，防止某类没出现时标签错位
        target_names=CLASS_NAMES,               # 显示类别名
        digits=4,                               # 保留 4 位小数
        zero_division=0,                        # 某类没有预测时按 0 处理，避免警告
        output_dict=True,
    )

    # 导出每类指标 CSV（按类别顺序）
    csv_path = run_dir / "metrics" / f"classification_report_{split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 表头
        w.writerow(["class", "precision", "recall", "f1-score", "support"])
        # 每一类一行
        for name in CLASS_NAMES:
            row = report[name]
            w.writerow([name, f"{row['precision']:.4f}", f"{row['recall']:.4f}",
                        f"{row['f1-score']:.4f}", int(row["support"])])
    # 日志提示
    logger.info("每类指标已保存：%s", csv_path)
    return report


def plot_confusion_matrix(run_dir: Path, split: str, ys, ps, logger) -> None:
    """
    功能：画并保存混淆矩阵（对角线越深 = 该类分得越准）。
    输入参数：
        run_dir —— Path，run 输出目录。
        split   —— str，"test" / "val"，用于文件名。
        ys      —— list[int]，真实标签。
        ps      —— list[int]，预测标签。
        logger  —— logging.Logger。
    输出参数：
        无（写 figures/confusion_matrix_<split>.png）。
    调用位置：
        eval.py 的 main() 中调用一次。
    """
    # 计算 10x10 混淆矩阵（行=真实，列=预测）
    cm = confusion_matrix(ys, ps, labels=list(range(len(CLASS_NAMES))))
    # 包成可绘制的对象
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    # 画图（x 轴标签旋转 45 度，避免重叠）
    disp.plot(xticks_rotation=45, cmap="Blues", values_format="d")
    # 紧凑布局
    plt.tight_layout()
    # 保存路径
    fig_path = run_dir / "figures" / f"confusion_matrix_{split}.png"
    # 300 dpi 保存
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    # 关闭图，释放内存
    plt.close()
    # 日志提示
    logger.info("混淆矩阵已保存：%s", fig_path)


def predict_single_image(model, image_path, device, logger) -> None:
    """
    功能：对单张本地图片做推理并打印 Top-3（可选功能，需要命令行传 --img-path）。
    输入参数：
        model      —— nn.Module，已加载权重。
        image_path —— str 或 Path，图片路径。
        device     —— torch.device。
        logger     —— logging.Logger。
    输出参数：
        无（只打印结果）。
    调用位置：
        eval.py 的 main() 中，当 args.img_path 不为空时调用。
    注意：
        预处理必须和训练时的 eval_transform 一致（Resize 32x32 + ToTensor + Normalize(0.5,0.5,0.5)），
        否则同一张图的预测会明显偏差。
    """
    # 路径存在性检查
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error("图片不存在：%s", image_path)
        return

    # 与训练一致的确定性预处理；外部图片尺寸任意，先缩到 32x32
    tf = transforms.Compose([
        transforms.Resize((32, 32)),                                  # 缩放到 CIFAR-10 输入尺寸
        transforms.ToTensor(),                                        # 转张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),       # 与训练同一归一化
    ])
    # 读取图片并强制 RGB（防止 PNG 透明通道导致 4 通道）
    image = Image.open(image_path).convert("RGB")
    # 预处理 -> 增加 batch 维 [3,32,32] -> [1,3,32,32] -> 搬到设备
    tensor = tf(image).unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(tensor)                       # [1,10]
        probs = torch.softmax(logits, dim=1)[0]      # 转概率
        top3_prob, top3_idx = probs.topk(3)          # Top-3

    # 打印前三名
    logger.info("单张图片预测：%s", image_path)
    for prob, idx in zip(top3_prob.tolist(), top3_idx.tolist()):
        logger.info("  %-10s %.4f", CLASS_NAMES[idx], prob)


def main():
    """
    功能：评估主流程（读配置 -> 建模型 -> 加载权重 -> 评估 -> 导出指标与图片）。
    输入参数：
        无（全部来自命令行）。
    输出参数：
        无（结果写进 outputs/<run>/）。
    调用位置：
        本文件底部 if __name__ == "__main__": main()。
    """
    # ---------- 1. 命令行参数 ----------
    parser = argparse.ArgumentParser(description="评估某个 run 的最佳权重（默认在 test 上只跑一次）")
    parser.add_argument("--run", type=str, required=True, help="run 名称，对应 outputs/<run>/")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"],
                        help="评估哪个集合：test（默认，最终指标）/ val（复核选模依据）")
    parser.add_argument("--ckpt", type=str, default=None, help="自定义权重路径（默认 outputs/<run>/ckpt/best.pth）")
    parser.add_argument("--limit-batches", type=int, default=0, help=">0 时只评估前 N 个 batch（冒烟测试）")
    parser.add_argument("--img-path", type=str, default=None, help="可选：单张图片推理路径")
    # 解析
    args = parser.parse_args()

    # ---------- 2. 定位 run 目录与配置 ----------
    run_dir = OUTPUTS_DIR / args.run
    config_path = run_dir / "config.yaml"
    # run 不存在或没有配置副本 -> 说明训练没跑过，直接报错
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在：{run_dir}\n先跑 train.py，例如：python train.py --config cnn.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"缺少 config.yaml：{config_path}（train.py 会自动写入，请重跑训练）")

    # 读取该 run 的配置（模型结构必须和训练时一致）
    config = load_config(config_path)

    # 日志：写进同一个 run 的 logs/eval_<split>.log，同时打到终端
    logger = setup_logger(run_dir / "logs" / f"eval_{args.split}.log", name="cifar10_eval")
    logger.info("评估 run=%s，split=%s", args.run, args.split)

    # 固定种子：保证 val 子集切分与训练时完全一致
    set_seed(int(config["seed"]))
    device = get_device(logger)                # 传 logger，设备信息写进 eval_<split>.log

    # ---------- 3. 重建模型并加载权重 ----------
    model = build_model(config).to(device)                       # 与训练同结构的空网络
    # 权重路径：默认 run 目录下的 best.pth
    ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "ckpt" / "best.pth"
    ckpt = load_checkpoint(ckpt_path, device)                    # 读取字典
    model.load_state_dict(ckpt["model_state"])                   # 灌入权重
    logger.info("已加载权重：%s（训练第 %s 轮，val_acc=%s）",
                ckpt_path, ckpt.get("epoch"), ckpt.get("val_acc"))

    # ---------- 4. 数据（只建需要的集合） ----------
    data_root = resolve_data_root(config)
    # split=val 时需要 train+val 复现切分；split=test 时只需要 test
    need_train = (args.split == "val")
    _, val_set, test_set = build_datasets(
        data_root=data_root,
        val_size=int(config["val_size"]),
        num_classes=int(config["num_classes"]),
        seed=int(config["seed"]),
        need_train=need_train,
    )
    # 选择评估集
    dataset = test_set if args.split == "test" else val_set
    logger.info("评估集：%s，样本数 %d", args.split, len(dataset))
    # 评估集不打乱，保证顺序固定、结果可复现
    loader = build_loader(dataset, int(config["batch_size"]), shuffle=False,
                          num_workers=int(config["num_workers"]))

    # ---------- 5. 逐样本预测 ----------
    ys, ps = collect_predictions(model, loader, device, limit_batches=args.limit_batches)
    # 整体准确率
    acc = float(np.mean(np.array(ys) == np.array(ps)))
    # 交叉熵损失（可选指标，用同一批标签计算）
    loss_fn = nn.CrossEntropyLoss()
    logger.info("Accuracy on %s: %.4f（样本 %d）", args.split, acc, len(ys))

    # ---------- 6. 每类指标 + 混淆矩阵 ----------
    report = save_classification_report(run_dir, args.split, ys, ps, logger)
    plot_confusion_matrix(run_dir, args.split, ys, ps, logger)

    # ---------- 7. 导出 JSON 汇总 ----------
    eval_summary = {
        "run_name": args.run,
        "split": args.split,
        "num_samples": len(ys),
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "checkpoint": str(ckpt_path),
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val_acc": ckpt.get("val_acc"),
        "limit_batches": args.limit_batches,
    }
    # 写 metrics/eval_<split>.json
    json_path = run_dir / "metrics" / f"eval_{args.split}.json"
    save_json(json_path, eval_summary)
    logger.info("评估汇总已保存：%s", json_path)

    # ---------- 8. 可选：单张图片推理 ----------
    if args.img_path:
        predict_single_image(model, args.img_path, device, logger)


# 只有直接运行时才执行
if __name__ == "__main__":
    main()
