# 重构变更对照表（供逐行核对）

> 本文档记录本次重构**具体改了哪里、为什么改**，方便你拿旧代码逐行对比。
> 旧代码来自 `CIFAR-10/train.py`、`CIFAR-10/eval.py`（已用 `git mv` 搬到 `experiments/w1_cifar10/`）。
> 所有行号指重构完成后的当前文件。

---

## 一、文件级变更

| 旧位置 | 新位置 | 动作 |
| --- | --- | --- |
| `CIFAR-10/train.py` | `experiments/w1_cifar10/train.py` | 搬移 + 重写（只保留训练/验证） |
| `CIFAR-10/eval.py` | `experiments/w1_cifar10/eval.py` | 搬移 + 重写（只跑 test，修 bug） |
| —（旧 train.py 里的公共代码） | `experiments/w1_cifar10/common.py` | **新增**：数据/模型/配置/日志公共库 |
| — | `experiments/w1_cifar10/configs/cnn.yaml` | **新增**：CNN 的全部超参 |
| — | `experiments/w1_cifar10/configs/resnet.yaml` | **新增**：ResNet-18 的全部超参 |
| `CIFAR-10/dataset/` | `data/cifar10/` | 搬移（`git mv` / `Move-Item`），数据集不再属于某个实验 |
| `modules/` | `experiments/w2_modules/` | 搬移（第二周模块独立成实验） |
| `notes/论文笔记/` | `notes/papers/` | 改名（中文路径改英文，避免 Git/工具兼容问题） |
| `CIFAR-10/checkpoints/`、`CIFAR-10/runs/`、`figures/`、`logs/` | `experiments/w1_cifar10/outputs/<run>/` | 产物统一随 run 走（旧目录已删除） |
| `CIFAR-10/reademe.txt`、`CIFAR-10/template.py`、`notes/3DWay.md`、`notes/R3DP.md` | — | 删除（空文件/无用模板/错字文件名） |
| — | `requirements.txt`、`docs/` 系列、各目录 `README.md` | **新增**：环境与约定说明 |

---

## 二、旧 `train.py` → 新代码（逐条对照）

### 1. 随机种子

| 旧代码 | 新代码 | 说明 |
| --- | --- | --- |
| L9–L14：模块级 `seed = 42`，import 时就执行 | [common.py](../experiments/w1_cifar10/common.py#L63-L79) `set_seed()`；在 [train.py L326](../experiments/w1_cifar10/train.py#L326) 调用 | 旧代码只要 import 就改全局随机状态（`eval.py` 也被牵连）；现在由调用方显式固定 |

### 2. 数据预处理

| 旧代码 | 新代码 | 说明 |
| --- | --- | --- |
| L17–L26：`train_transform` / `test_transform` | [common.py L109-L136](../experiments/w1_cifar10/common.py#L109-L136) `build_transforms()` | 内容**完全保留**（RandomCrop(32,4) + RandomHorizontalFlip + ToTensor + Normalize(0.5)），只是移到公共库 |

### 3. 数据集与 DataLoader

| 旧代码 | 新代码 | 说明 |
| --- | --- | --- |
| L29–L30：`datasets.CIFAR10(root='./dataset', ...)`，train 直接用 50k | [common.py L216-L273](../experiments/w1_cifar10/common.py#L216-L273) `build_datasets()` | ① 路径改为 `data/cifar10`（由 YAML 给，相对仓库根）② 50k 训练集**分层切 45k/5k** |
| 无验证集，每轮用 test 集评估 | [common.py L178-L214](../experiments/w1_cifar10/common.py#L178-L214) `stratified_split()` + [L138-L176](../experiments/w1_cifar10/common.py#L138-L176) `TransformSubset` | **核心修复**：验证集从训练集切分，且 val 不施加随机增强 |
| L33–L34：`DataLoader(batch_size=64)` 写死 | [common.py L275-L299](../experiments/w1_cifar10/common.py#L275-L299) `build_loader()` | batch_size / num_workers 来自 YAML；开启 `persistent_workers` 减少 Windows 下每轮重建子进程的开销 |

### 4. 设备

| 旧代码 | 新代码 | 说明 |
| --- | --- | --- |
| L38–L39：`torch.device(...)` + `print` | [common.py L82-L102](../experiments/w1_cifar10/common.py#L82-L102) `get_device(logger)` | 设备信息写进 `train.log`（原 `print` 只进终端） |

### 5. 模型定义

| 旧代码 | 新代码 | 说明 |
| --- | --- | --- |
| L45–L63：`class simpleCNN` | [common.py L308-L340](../experiments/w1_cifar10/common.py#L308-L340) `class SimpleCNN` | 结构**一模一样**，仅改名符合类名规范 |
| L66–L69：`get_resnet18(weights=None)` | [common.py L342-L381](../experiments/w1_cifar10/common.py#L342-L381) `build_model()` | 增加 `pretrained` 开关（YAML 里默认 false，与原实验一致） |
| L72–L78：`select_model(mod)`，`cnn` / `resnet` | 同上 `build_model(config)`，名字 `cnn` / `resnet18` | 旧 `--mod` 命令行改为 YAML 的 `model.name`，**命令变成 `--config`** |

### 6. 优化器 / 调度器

| 旧代码 | 新代码 | 说明 |
| --- | --- | --- |
| L98：`SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)` 写死 | [train.py L174-L199](../experiments/w1_cifar10/train.py#L174-L199) `build_optimizer()` | 超参移到 YAML；同时支持 `adamw` |
| L100：`CosineAnnealingLR(T_max=100)` 写死 | [train.py L201-L223](../experiments/w1_cifar10/train.py#L201-L223) `build_scheduler()` | **T_max 改为“实际总轮数”**，之后改 `epochs` 不会再与余弦周期错配 |

### 7. 训练循环

| 旧代码 | 新代码 | 说明 |
| --- | --- | --- |
| L106：`for epoch in range(100)` | [train.py L392](../experiments/w1_cifar10/train.py#L392) `range(1, total_epochs+1)` | 轮数来自 YAML；冒烟测试可用 `--limit-epochs` |
| L111–L120：batch 循环内联在 `main` 里 | [train.py L50-L126](../experiments/w1_cifar10/train.py#L50-L126) `train_one_epoch()` | 抽成函数；损失按样本数加权（旧代码用 `running_loss / len(loader)`，最后一个不满 batch 时有轻微偏差） |
| L125–L137：**每轮跑 test_loader** | [train.py L128-L172](../experiments/w1_cifar10/train.py#L128-L172) `evaluate()` 在 **val_loader** 上跑 | **核心修复：测试集不再参与训练/选模**，杜绝数据泄漏 |
| L145–L149：`if epoch_acc > best_acc` 保存权重 | [train.py L437-L458](../experiments/w1_cifar10/train.py#L437-L458)：`if val_acc > best_val_acc` | best 权重依据改成验证集 |
| L149：`checkpoints/{mod}_best_model.pth` | [train.py L441-L453](../experiments/w1_cifar10/train.py#L441-L453)：`outputs/<run>/ckpt/best.pth` | 权重带 run 归属；同目录同时存 `config.yaml` |
| L83 / L150–L154：`patience = 5` 全局变量 | [train.py L375](../experiments/w1_cifar10/train.py#L375) 读 YAML，默认 15 | 验证集噪声比 test 大，patience 5 容易误停 |
| L103–L104：`SummaryWriter(f'runs/cifar10_{mod}')`（相对 CWD） | [train.py L379](../experiments/w1_cifar10/train.py#L379) `SummaryWriter(log_dir=run_dir/"tb")` | 路径不再依赖“从哪个目录启动” |
| 无 | [train.py L381-L385](../experiments/w1_cifar10/train.py#L381-L385) `metrics/metrics.csv` | 每轮指标落盘，方便直接画图/对比 |
| 无 | [train.py L225-L297](../experiments/w1_cifar10/train.py#L225-L297) `save_summary()` | 自动写 `summary.md` / `summary.json`（任务书要求的“结论记录”） |
| L156–L157：`print` + `writer.close()` | 日志器 + `writer.close()` + summary | 终端输出同时写 `logs/train.log` |

### 8. 日志（替换你的 Tee-Object / 重定向）

| 旧做法 | 新做法 | 说明 |
| --- | --- | --- |
| `python train.py \| Tee-Object ..\logs\xxx.log` | 直接 `python train.py --config resnet.yaml` | 代码内置 `logging`，**同时**写终端和 `outputs/<run>/logs/train.log` |
| 窗口 A 重定向 + 窗口 B `Get-Content -Wait` | 窗口 B：`Get-Content outputs\resnet_baseline\logs\train.log -Wait` | 路径改为 run 目录下；训练命令本身保持干净 |

---

## 三、旧 `eval.py` → 新代码（逐条对照）

| 旧代码 | 新代码 | 说明 |
| --- | --- | --- |
| L11：`from train import test_loader, device, select_model` | [eval.py L41-L52](../experiments/w1_cifar10/eval.py#L41-L52)：只 `from common import ...` | **修 bug**：旧写法一 import 就下载数据、建 DataLoader、固定随机种子；且耦合两个脚本 |
| L16：`--mod` | [eval.py L220](../experiments/w1_cifar10/eval.py#L220)：`--run` | run 目录决定权重与配置，避免“模型名 + 权重文件”对不上 |
| L18：`--img_path` 被注释；L21 却用 `args.img_path` | [eval.py L225](../experiments/w1_cifar10/eval.py#L225)：`--img-path` 正式定义 | **修 bug**：旧代码只要运行必然 `AttributeError: 'Namespace' object has no attribute 'img_path'` |
| L23–L24：`torch.load(f'checkpoints/{mod}_best_model.pth')` | [eval.py L250-L257](../experiments/w1_cifar10/eval.py#L250-L257) | 默认读 `outputs/<run>/ckpt/best.pth`；权重现在是字典（含 epoch/val_acc/config），`--ckpt` 可指定任意路径 |
| L31–L36：逐 batch 收集标签 | [eval.py L55-L88](../experiments/w1_cifar10/eval.py#L55-L88) `collect_predictions()` | 逻辑保留，抽成函数；增加 `--limit-batches` 冒烟开关 |
| L53：`figures/{mod}_confusion_matrix2.png`（相对 CWD） | [eval.py L130-L158](../experiments/w1_cifar10/eval.py#L130-L158) → `outputs/<run>/figures/confusion_matrix_<split>.png` | 路径不依赖 CWD，且文件名去掉了多余的 `2` |
| 无 | [eval.py L90-L128](../experiments/w1_cifar10/eval.py#L90-L128) 每类 precision/recall/f1 → CSV | 新增：能看出“哪个类总被分错” |
| 无 | [eval.py L288-L305](../experiments/w1_cifar10/eval.py#L288-L305) `eval_<split>.json` | 新增：机器可读的评估汇总 |
| L59–L91：整段注释掉的单图推理 | [eval.py L162-L206](../experiments/w1_cifar10/eval.py#L162-L206) `predict_single_image()` | 功能**启用**并修好：Resize 32 + 同一 Normalize，打印 Top-3 |
| 无 | `--split val` | 新增：复核“选模依据”是否与训练时一致 |

---

## 四、行为变化速查（只想知道“结论变了什么”看这里）

1. **不再用测试集选模型**：验证集 = 从 50k 训练集分层切出的 5k；测试集 10k 只在 `eval.py` 用一次。
2. **权重命名带 run 归属**：`outputs/<run>/ckpt/best.pth`，同目录有 `config.yaml` 与 `summary.md`。
3. **路径不再依赖当前目录**：`REPO_ROOT` 由 `Path(__file__).resolve().parents[2]` 算出；数据、输出全部绝对定位。
4. **超参全部在 YAML**：改实验只改 `configs/*.yaml`，不动 `.py`。
5. **日志内置**：不再需要 `Tee-Object` 或 `1>>` 重定向；`Get-Content ... -Wait` 仍可用于实时跟踪。
6. **`eval.py` 不再 import `train.py`**：无副作用，可单独运行。
7. **`--img-path` 可用**：旧版本该参数缺失导致必崩。
8. 训练用 `val_loss/val_acc` 记录到 TensorBoard；旧版是 `test`。

---

## 五、本次没有改的东西（避免你误以为漏了）

- CNN 与 ResNet-18 的网络结构、数据增强参数、SGD 超参（lr=0.01 / momentum=0.9 / wd=5e-4）**全部保留**。
- 类别名顺序、Normalize 的 (0.5, 0.5, 0.5) **未改**。
- CIFAR-10 数据文件（`cifar-10-python.tar.gz` + `cifar-10-batches-py`）未动，只换了位置。
