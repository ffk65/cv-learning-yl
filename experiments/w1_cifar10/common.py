"""
common.py —— 第 1 周 CIFAR-10 实验的公共工具库（脚本本体不单独运行，只被 import）

核心作用：把 train.py 和 eval.py 必须共享的东西集中到一处，保证两边“同一份数据、同一个模型结构、同一套路径规则”：
  1) 路径常量：REPO_ROOT / EXPERIMENT_DIR / OUTPUTS_DIR / CONFIG_DIR
  2) 随机种子与设备选择
  3) 数据增强、45k/5k/10k 分层切分、DataLoader 构建
  4) 模型工厂（simple CNN / ResNet-18）
  5) YAML 配置读取、run 输出目录解析、日志初始化、权重读取、JSON 写出

调用关系（谁 import 我）：
  train.py  → build_datasets / build_loader / build_model / load_config / resolve_run_dir / setup_logger / set_seed / get_device / resolve_data_root / save_json
  eval.py   → build_datasets / build_loader / build_model / load_config / setup_logger / set_seed / get_device / resolve_data_root / load_checkpoint / save_json

注意：本文件里没有任何“训练”逻辑，import 它不会启动训练、也不会下载数据。
"""

# ---------- 标准库 ----------
import json                          # 写 metrics / summary 这类机器可读的结果文件
import logging                       # 训练日志：同时输出到终端与文件
import random                        # Python 自带随机数
import sys                           # 日志输出到标准输出
from pathlib import Path             # 统一用 Path 处理路径，避免 Windows 反斜杠问题

# ---------- 第三方库 ----------
import numpy as np                   # 分层切分时需要按类别操作索引
import torch                         # 深度学习框架
import yaml                          # 读取 configs/*.yaml
from torch import nn                 # 定义网络层
from torch.utils.data import DataLoader, Dataset   # 数据加载器、自定义子集
from torchvision import datasets, models, transforms   # CIFAR-10 数据集、ResNet、数据增强


# ==========================================================================
# 一、路径常量（全部由代码自动算出来，和你在哪个目录敲命令无关）
# ==========================================================================

# __file__ 是本文件 common.py 的路径；resolve() 得到绝对路径；
# parents[0] = experiments/w1_cifar10，parents[1] = experiments，parents[2] = 仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[2]

# 当前实验目录（experiments/w1_cifar10），输出、配置都放在它下面
EXPERIMENT_DIR = Path(__file__).resolve().parent

# 所有 run 的产出根目录：experiments/w1_cifar10/outputs
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"

# 配置文件目录：experiments/w1_cifar10/configs
CONFIG_DIR = EXPERIMENT_DIR / "configs"

# CIFAR-10 的 10 个类别名，顺序必须和 torchvision 的 label 编号一致
# 0:airplane 1:automobile 2:bird 3:cat 4:deer 5:dog 6:frog 7:horse 8:ship 9:truck
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ==========================================================================
# 二、基础工具：随机种子 / 设备
# ==========================================================================

def set_seed(seed: int) -> None:
    """
    功能：固定所有随机源，保证“同一份配置 + 同一个种子”跑出可复现的结果。
    输入参数：
        seed —— int，随机种子，来自 YAML 的 seed 字段（默认 42）。
    输出参数：
        无返回值，直接修改全局随机状态。
    调用位置：
        train.py 的 main() 开头；
        eval.py  的 main() 开头（保证 val 切分与训练时完全一致）。
    """
    random.seed(seed)                              # Python random 模块
    np.random.seed(seed)                           # NumPy
    torch.manual_seed(seed)                        # CPU 上的 torch
    torch.cuda.manual_seed_all(seed)               # 所有 GPU（没有 GPU 时该调用也安全）
    torch.backends.cudnn.deterministic = True      # 卷积结果确定化
    torch.backends.cudnn.benchmark = False         # 关闭自动选最快算法，换取可复现


def get_device(logger=None) -> torch.device:
    """
    功能：自动选择 GPU（有 CUDA 用 cuda，否则用 cpu）。
    输入参数：
        logger —— logging.Logger 或 None；传入本项目的 logger 才能把设备信息写进日志文件。
    输出参数：
        torch.device —— 可直接 .to(device) 使用。
    调用位置：
        train.py 的 main()（建模型、搬数据前）；
        eval.py  的 main()（加载权重前）。
    """
    # torch.cuda.is_available() 为 True 表示当前 Python 里的 torch 是 CUDA 版本且驱动可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 拼装要打印的设备信息（方便确认“到底用的是不是 GPU”）
    if device.type == "cuda":
        message = f"Device: cuda ({torch.cuda.get_device_name(0)})"
    else:
        message = "Device: cpu（如果本机有 N 卡，请确认用的是 cv 环境，而不是 base 的 CPU 版 torch）"
    # 有 logger 就写进 run 日志，没有则退回标准 logging
    (logger.info if logger else logging.info)(message)
    return device


# ==========================================================================
# 三、数据：增强、分层切分、子集包装、DataLoader
# ==========================================================================

def build_transforms():
    """
    功能：构建训练用（带增强）和评估用（不带增强）两套图像预处理。
    输入参数：
        无（CIFAR-10 的尺寸 32x32、均值方差都写死在这里，保证 train/val/test 一致）。
    输出参数：
        (train_transform, eval_transform) —— 两个 torchvision.transforms.Compose 对象。
    调用位置：
        common.build_datasets()（内部给 train 子集和 val/test 用）。
    说明：
        val 和 test 必须用 eval_transform（只 ToTensor + Normalize），
        否则验证集带随机裁剪/翻转，指标会抖动，选出的 best 权重不可靠。
    """
    # 训练集：随机裁剪（四周补 4 像素再裁回 32）+ 随机水平翻转，属于常规轻量增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                          # 位置扰动
        transforms.RandomHorizontalFlip(),                             # 镜像扰动
        transforms.ToTensor(),                                         # PIL -> [0,1] 的 Tensor，形状 [3,32,32]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),        # 归一化到 [-1,1]
    ])
    # 验证集 / 测试集：只做确定性变换，不加任何随机性
    eval_transform = transforms.Compose([
        transforms.ToTensor(),                                         # 只转张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),        # 只做同样的归一化
    ])
    # 返回两套变换
    return train_transform, eval_transform


class TransformSubset(Dataset):
    """
    功能：从同一个 CIFAR-10 数据集里按“索引列表”取出一个子集，
          并允许该子集使用自己独立的 transform（train 子集带增强，val 子集不带）。
    输入参数：
        base_dataset —— torchvision.datasets.CIFAR10，构造时必须 transform=None（返回 PIL 图像）。
        indices      —— list[int]，该子集包含的样本索引。
        transform    —— 该子集要施加的预处理（Compose 对象）。
    输出参数：
        __getitem__ 返回 (image_tensor, label)。
    调用位置：
        common.build_datasets() 内部，分别创建 train_subset 与 val_subset。
    为什么不用 torch.utils.data.Subset：
        官方 Subset 只能复用 base_dataset 的 transform，无法让 train/val 两个子集用不同增强。
    """

    def __init__(self, base_dataset: Dataset, indices, transform):
        # 保存原始数据集（不带 transform，取出来的是 PIL.Image）
        self.base_dataset = base_dataset
        # 转成 list 并保存，保证可被 DataLoader 的 worker 正常取用
        self.indices = list(indices)
        # 保存本子集专用的 transform
        self.transform = transform

    def __len__(self):
        # 子集长度 = 索引个数
        return len(self.indices)

    def __getitem__(self, position: int):
        # position 是“子集内的第几个”，需要映射回原始数据集的下标
        base_index = self.indices[position]
        # 从原始数据集取出 (PIL 图, int 标签)
        image, label = self.base_dataset[base_index]
        # 对图像施加本子集的 transform（训练子集是增强版，验证子集是确定性版）
        if self.transform is not None:
            image = self.transform(image)
        # 返回模型可直接使用的 (Tensor, label)
        return image, label


def stratified_split(targets, val_size: int, num_classes: int, seed: int):
    """
    功能：对 50k 训练集做“按类别等比例”的切分（分层抽样），得到 45k 训练 / 5k 验证。
    输入参数：
        targets     —— 长度为 50000 的类别标签序列（CIFAR10.targets）。
        val_size    —— int，验证集总张数（配置里默认 5000）。
        num_classes —— int，类别数（CIFAR-10 为 10）。
        seed        —— int，随机种子，必须与训练时一致，eval 才能复现同一个 val 子集。
    输出参数：
        (train_indices, val_indices) —— 两个 list[int]，互不重叠，并集为全部 50k。
    调用位置：
        common.build_datasets() 内部。
    设计说明：
        每个类别各拿出 val_size/num_classes 张（5000/10 = 500 张），
        这样验证集的类别分布与训练集一致，val 准确率不会因类别不平衡而虚高/虚低。
    """
    # 把标签转成 numpy 数组，便于做按类别筛选
    targets = np.asarray(targets)
    # 每个类别要抽出的验证样本数（整除，保证总数正好等于 val_size）
    per_class_val = val_size // num_classes
    # 独立随机数生成器：只受 seed 影响，不受 torch / numpy 全局状态影响
    rng = np.random.default_rng(seed)
    # 分别收集训练索引与验证索引
    train_indices, val_indices = [], []
    # 对每个类别单独处理
    for class_id in range(num_classes):
        # 找出属于当前类别的所有样本下标（升序）
        class_indices = np.flatnonzero(targets == class_id)
        # 打乱这份下标，保证抽取的验证样本不是固定前 500 张
        rng.shuffle(class_indices)
        # 前 per_class_val 张进验证集
        val_indices.extend(class_indices[:per_class_val].tolist())
        # 剩下的进训练集
        train_indices.extend(class_indices[per_class_val:].tolist())
    # 返回两个索引列表（顺序不重要，DataLoader 会自行 shuffle）
    return train_indices, val_indices


def build_datasets(data_root: Path, val_size: int, num_classes: int, seed: int,
                   need_train: bool = True):
    """
    功能：构建 train / val / test 三个数据集对象（train 与 val 来自同一个 50k 训练集）。
    输入参数：
        data_root   —— Path，CIFAR-10 数据目录（仓库内为 data/cifar10）。
        val_size    —— int，验证集张数。
        num_classes —— int，类别数。
        seed        —— int，切分种子。
        need_train  —— bool，是否需要 train/val（eval 只跑 test 时可传 False，省一次切分）。
    输出参数：
        (train_subset, val_subset, test_dataset)
        need_train=False 时前两项为 None。
    调用位置：
        train.py 的 main()（need_train=True）；
        eval.py  的 main()（split=test 时 need_train=False，split=val 时 True）。
    说明：
        download=True 只会在 data_root 缺失数据时触发一次下载（约 170MB）。
    """
    # 取两套预处理
    train_transform, eval_transform = build_transforms()

    # 测试集：10k 张，永远只用 eval_transform，且只在最终评估时使用
    test_dataset = datasets.CIFAR10(
        root=str(data_root),        # 数据目录（绝对路径，和 CWD 无关）
        train=False,                # False = test 分支
        download=True,              # 缺失时自动下载
        transform=eval_transform,   # 确定性预处理
    )

    # 如果调用方只需要 test，直接返回，避免无谓地构造 50k 训练集
    if not need_train:
        return None, None, test_dataset

    # 50k 训练集：先以 transform=None 构造，取出来是 PIL 图，再由子集各自施加变换
    base_train = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=True,
        transform=None,             # 关键：这里不加 transform，交给 TransformSubset 处理
    )

    # 分层切分 45k / 5k
    train_indices, val_indices = stratified_split(
        targets=base_train.targets,
        val_size=val_size,
        num_classes=num_classes,
        seed=seed,
    )

    # 训练子集：带数据增强
    train_subset = TransformSubset(base_train, train_indices, train_transform)
    # 验证子集：不带增强（与 test 完全同样的预处理）
    val_subset = TransformSubset(base_train, val_indices, eval_transform)

    # 返回三个数据集
    return train_subset, val_subset, test_dataset


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    """
    功能：把一个 Dataset 包装成 DataLoader。
    输入参数：
        dataset     —— 任意 Dataset（上面三个之一）。
        batch_size  —— int，批大小。
        shuffle     —— bool，是否每个 epoch 打乱（训练 True，验证/测试 False）。
        num_workers —— int，数据加载子进程数；Windows 下建议 0 或 2。
    输出参数：
        torch.utils.data.DataLoader。
    调用位置：
        train.py 的 main()（train/val/test 三个 loader）；
        eval.py  的 main()（test 或 val 一个 loader）。
    """
    # pin_memory 只在有 CUDA 时开启，可加速 CPU->GPU 拷贝
    use_pin_memory = torch.cuda.is_available()
    # 构造并返回 DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,                 # 训练打乱，评估固定顺序（保证指标可复现）
        num_workers=num_workers,         # 子进程数
        pin_memory=use_pin_memory,       # 锁页内存
        drop_last=False,                 # 不丢最后不足一个 batch 的样本
        # 有子进程时常驻它们，避免每个 epoch 重新 spawn + 重新拷贝数据集（Windows 上能省很多时间）
        persistent_workers=(num_workers > 0),
    )


# ==========================================================================
# 四、模型工厂
# ==========================================================================

class SimpleCNN(nn.Module):
    """
    功能：第 1 周使用的 3 层卷积小网络（结构与你原来的 simpleCNN 完全一致）。
    输入参数：
        无（通道数写死在结构里）。
    输出参数：
        forward 返回 [batch, 10] 的 logits。
    调用位置：
        common.build_model()（当配置 model.name == "cnn" 时实例化）。
    """

    def __init__(self):
        super().__init__()
        # 特征提取部分：3 次 (卷积 -> ReLU -> 2x2 最大池化)，空间尺寸 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        # 分类部分：展平 128x4x4 = 2048 维 -> 256 -> 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        # 先过卷积特征
        x = self.features(x)
        # 再过全连接分类
        return self.classifier(x)


def build_model(config: dict) -> nn.Module:
    """
    功能：按 YAML 配置创建网络（不加载预训练权重，除非配置显式开启）。
    输入参数：
        config —— dict，完整配置；读取 model.name / model.pretrained / num_classes。
    输出参数：
        nn.Module —— 未搬到设备的模型（调用方负责 .to(device)）。
    调用位置：
        train.py 的 main()（建待训练模型）；
        eval.py  的 main()（建同样结构的空网络，再灌权重）。
    为什么 eval 也要用这个函数：
        .pth 只存权重数字，不存结构；必须用同一个工厂函数重建结构才能 load_state_dict。
    """
    # 读取模型名，统一转小写避免大小写不一致
    name = str(config["model"]["name"]).lower()
    # 读取类别数
    num_classes = int(config.get("num_classes", 10))
    # 读取是否使用 ImageNet 预训练权重（默认 False，和原实验保持一致）
    pretrained = bool(config["model"].get("pretrained", False))

    # 分支 1：简单 CNN
    if name == "cnn":
        return SimpleCNN()

    # 分支 2：ResNet-18，可选用预训练权重
    if name == "resnet18":
        # weights="IMAGENET1K_V1" 会联网下载约 45MB；weights=None 表示随机初始化
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.resnet18(weights=weights)
        # 替换最后的全连接层，输出改成 CIFAR-10 的 10 类
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # 其他名字直接报错，避免静默用错网络
    raise ValueError(f"未知模型名: {name}，可选 cnn / resnet18")


# ==========================================================================
# 五、配置、run 目录、日志、权重、JSON
# ==========================================================================

def load_config(path) -> dict:
    """
    功能：读取 YAML 配置并补上默认值。
    输入参数：
        path —— str 或 Path；可以写绝对路径，也可以写相对 configs/ 的文件名（如 "resnet.yaml"）。
    输出参数：
        dict —— 已合并默认值的配置字典。
    调用位置：
        train.py 的 main()（命令行 --config）；
        eval.py  的 main()（读取 run 目录里保存的 config.yaml 副本）。
    路径解析规则（解决“我不想记 REPO_ROOT”）：
        1) 绝对路径 -> 直接用；
        2) 相对路径 -> 先按 configs/ 解析（"resnet.yaml" 等价于 configs/resnet.yaml）；
        3) 还找不到 -> 再按当前工作目录解析（兼容你 cd 到某处再运行）。
    """
    # 统一转成 Path
    candidate = Path(path)
    # 规则 1：绝对路径直接使用
    if candidate.is_absolute():
        config_path = candidate
    else:
        # 规则 2：优先当作 configs/ 下的文件名
        config_path = CONFIG_DIR / candidate
        # 规则 3：configs/ 下不存在时，退回按当前工作目录解析
        if not config_path.exists():
            config_path = Path.cwd() / candidate

    # 文件仍不存在 -> 明确报错，提示可用配置
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件: {path}\n可用配置: {sorted(p.name for p in CONFIG_DIR.glob('*.yaml'))}"
        )

    # 读取 YAML（encoding 指定 utf-8，允许中文注释）
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 基础默认值：配置里没写的键用这里的值，避免 KeyError
    defaults = {
        "seed": 42,                          # 随机种子
        "data_root": "data/cifar10",         # 相对仓库根的数据目录
        "num_classes": 10,                   # 类别数
        "val_size": 5000,                    # 验证集张数
        "batch_size": 128,                   # 批大小
        "num_workers": 0,                    # 数据加载子进程（Windows 从 0 开始最稳）
        "epochs": 100,                       # 训练轮数
        "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
        "scheduler": {"name": "cosine"},     # 余弦退火，T_max 自动等于 epochs
        "early_stopping_patience": 15,       # 验证集连续多少轮不提升就停
        "log_every": 50,                     # 每多少个 batch 打一行日志
        "model": {"name": "cnn", "pretrained": False},
    }
    # 浅层合并：以 defaults 为底，用配置文件里的值覆盖
    merged = {**defaults, **config}
    # model / optimizer / scheduler 是嵌套字典，需要各自再合并一次，防止只写一半字段
    for section in ("model", "optimizer", "scheduler"):
        merged[section] = {**defaults.get(section, {}), **(config.get(section) or {})}
    # 返回合并后的配置
    return merged


def resolve_data_root(config: dict) -> Path:
    """
    功能：把配置里的 data_root（相对仓库根的字符串）解析成绝对 Path。
    输入参数：
        config —— dict，含 data_root。
    输出参数：
        Path —— 绝对路径，如 D:/Research/Code/cv-learning/data/cifar10。
    调用位置：
        train.py 与 eval.py 的 main()，在 build_datasets 之前调用。
    """
    # 去掉首尾空白，避免 YAML 里多敲空格
    raw = str(config["data_root"]).strip()
    # 绝对路径直接用
    if Path(raw).is_absolute():
        return Path(raw)
    # 相对路径一律相对于仓库根，而不是当前工作目录
    return (REPO_ROOT / raw).resolve()


def resolve_run_dir(run_name: str, overwrite: bool = False) -> Path:
    """
    功能：确定并创建本次 run 的输出目录 outputs/<run_name>/ 及其子目录。
    输入参数：
        run_name  —— str，run 名称（如 "cnn_baseline"）。
        overwrite —— bool，目录已存在时是否允许清空重跑。
    输出参数：
        Path —— run 目录绝对路径。
    调用位置：
        train.py 的 main() 开头（先确定目录，再建日志）。
    目录结构（全部由本函数创建）：
        outputs/<run_name>/
            ckpt/       权重（不上传）
            tb/         TensorBoard（不上传）
            logs/       终端日志（不上传）
            figures/    图片（上传）
            metrics/    csv/json 指标（上传）
    """
    # 目标目录
    run_dir = OUTPUTS_DIR / run_name
    # 已存在且不允许覆盖 -> 直接报错，保护已有实验结果
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"run 目录已存在: {run_dir}\n"
            f"换一个名字：--run-name {run_name}_v2；或确认要覆盖：--overwrite"
        )
    # 允许覆盖时，先整个删掉旧目录（显式 --overwrite 才会走到这里）
    if run_dir.exists() and overwrite:
        import shutil                      # 延迟导入，只有覆盖时才需要
        shutil.rmtree(run_dir)             # 递归删除旧产物
    # 逐个创建子目录（mkdir 的 parents=True 允许中间目录不存在）
    for sub in ("ckpt", "tb", "logs", "figures", "metrics"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    # 返回 run 目录
    return run_dir


def setup_logger(log_file, name: str = "cifar10") -> logging.Logger:
    """
    功能：创建“同时写终端 + 写文件”的日志器（替代你原来的 Tee-Object / 输出重定向）。
    输入参数：
        log_file —— Path，日志文件路径（如 outputs/<run>/logs/train.log）。
        name     —— str，logger 名称，避免多次调用重复绑定 handler。
    输出参数：
        logging.Logger —— 直接用 logger.info(...) 即可。
    调用位置：
        train.py（logs/train.log）；eval.py（logs/eval_<split>.log）。
    说明：
        日志文件用 utf-8 编码，中文在 Windows 下不会乱码；
        logger.propagate=False 防止重复输出到 root logger。
    """
    # 取（或创建）指定名字的 logger
    logger = logging.getLogger(name)
    # 设为 DEBUG 以上全部记录（本项目主要用 INFO）
    logger.setLevel(logging.INFO)
    # 已经有 handler 时先清空，避免重复打印（例如同一进程内重复调用）
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # 格式：时间 | 级别 | 内容
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 终端 handler：输出到 stdout（配合你原来的窗口习惯）
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 文件 handler：追加写入，utf-8
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)     # 父目录不存在时先建
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # 不向 root logger 传递，避免重复
    logger.propagate = False
    return logger


def load_checkpoint(ckpt_path, device) -> dict:
    """
    功能：读取训练保存的权重文件（本项目的权重是“字典”，含 model_state/epoch/val_acc/config）。
    输入参数：
        ckpt_path —— str 或 Path，权重路径。
        device    —— torch.device，map_location 用，兼容 GPU 存、CPU 读。
    输出参数：
        dict —— 至少含 "model_state"，其余字段按训练时写入的内容。
    调用位置：
        eval.py 的 main()。
    """
    # 文件不存在时给出明确提示
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
    # weights_only=True：只反序列化张量与基础类型，更安全（torch >= 2.0 支持）
    return torch.load(ckpt_path, map_location=device, weights_only=True)


def save_json(path, obj) -> None:
    """
    功能：把 dict 写成 JSON（utf-8、带缩进、中文不转义）。
    输入参数：
        path —— str 或 Path，输出文件路径。
        obj  —— 可 JSON 序列化的对象。
    输出参数：
        无（只落盘）。
    调用位置：
        train.py 写 summary.json；eval.py 写 eval_<split>.json。
    """
    # 统一成 Path
    path = Path(path)
    # 父目录先建好
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写文件；ensure_ascii=False 保留中文
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
