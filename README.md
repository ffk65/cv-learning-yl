# cv-learning

> 山东大学 · 控制科学与工程学院 · 科研入门
> 作者：袁亮　远端仓库：<https://github.com/ffk65/cv-learning-yl.git>
> 最近更新：2026-11-03（目录重构 + 训练/评估流程修正）

按《CV 科研入门任务书》，用 PyTorch 复现 **CIFAR-10 图像分类**，走通「数据 → 网络定义 → 训练 → 验证 → 测试 → 可视化」全流程。

**重要变更**：旧版 README 里的准确率（CNN 0.8577 / ResNet-18 0.8555）是“用测试集选模型”得到的，属于数据泄漏，**已作废**。新代码改为 45k 训练 / 5k 验证 / 10k 测试，重跑后以 `experiments/w1_cifar10/outputs/<run>/summary.md` 为准。

---

## 一、目录结构

```text
cv-learning/
├── README.md                        ← 本文件（项目总入口）
├── .gitignore                       ← 数据/权重/TensorBoard/日志 不上传
├── requirements.txt                 ← Python 依赖清单
├── data/                            ← 【只读】数据集大盘
│   └── cifar10/                     ← CIFAR-10：tar.gz + 解压目录（约 340MB）【不上传】
├── experiments/                     ← 【实验】一次完整配置 = 一个子目录
│   ├── w1_cifar10/                  ← 第 1 周：CIFAR-10 分类
│   │   ├── common.py                ← 公共库：数据/模型/配置/日志（不单独运行）
│   │   ├── train.py                 ← 训练脚本：train 45k + val 5k
│   │   ├── eval.py                  ← 评估脚本：test 10k 只跑一次
│   │   ├── configs/
│   │   │   ├── cnn.yaml             ← 简单 CNN 的完整超参
│   │   │   └── resnet.yaml          ← ResNet-18 的完整超参
│   │   ├── assets/plane.jpg         ← 单图推理用的示例图片
│   │   ├── outputs/                 ← 每次 run 的产物目录
│   │   │   └── <run_name>/
│   │   │       ├── config.yaml      ← 本次配置副本（入库）
│   │   │       ├── summary.md/.json ← 本次结论（入库）
│   │   │       ├── ckpt/best.pth    ← 最佳权重【不上传】
│   │   │       ├── tb/              ← TensorBoard 事件【不上传】
│   │   │       ├── logs/train.log   ← 训练日志【不上传】
│   │   │       ├── figures/         ← 混淆矩阵等图片（入库）
│   │   │       └── metrics/         ← csv/json 指标（入库）
│   │   └── README.md                ← 第 1 周实验说明
│   ├── w2_modules/                  ← 第 2 周：模块 demo（自带 __main__ 自测）
│   │   ├── ResNet_blocks.py         ← BasicBlock / Bottleneck
│   │   ├── DeformConv.py            ← DeformConv v1 / v2
│   │   └── README.md
│   └── w4_retinanet_voc/            ← 第 4-5 周：简化版 RetinaNet（内部结构仿 mmdetection / detectron2）
│       ├── configs/config.py        ← 唯一超参源（对应框架的 configs/）
│       ├── data/                    ← make_split.py / voc.py / splits/
│       ├── models/                  ← backbone.py / anchor.py / retinanet.py
│       ├── losses/loss.py           ← 分配 + focal + smooth L1
│       ├── trainer/                 ← train.py / eval.py
│       ├── utils/                   ← box.py / metric.py
│       ├── tools/predict.py         ← 单图推理
│       ├── checkpoints/             ← 权重【不上传】
│       ├── runs/                    ← TensorBoard【不上传】
│       └── README.md                ← 设计思路 + 与两个参考框架的逐层对应表
├── docs/
│   ├── 环境配置文档.md              ← 环境类信息唯一事实源
│   └── refactor_changes.md          ← 本次重构的逐行变更对照（重点看这个）
└── notes/                           ← 【记录】笔记，不参与代码运行
    ├── README.md
    └── papers/                      ← 论文笔记（W2/W3）+ images
```

**四类东西分开**：`data/`（只读数据）｜`experiments/`（代码 + 配置 + 产物 + 结论）｜`docs/`（环境/约定）｜`notes/`（笔记）。

**仓库定位**：本仓库是**学习工作台**，不是框架。`experiments/` 下每个子目录 = 一个独立项目 = 未来可单独拆出去的仓库。项目内部结构参考 mmdetection / detectron2 的分层（configs / data / models / losses / trainer / utils / tools），但第三方框架本身放在仓库外：`D:\Research\Code\reference\mmdetection`。

---

## 二、文件清单与作用

| 文件 | 作用 |
| --- | --- |
| `experiments/w1_cifar10/common.py` | **公共库**。① 路径常量 `REPO_ROOT` / `OUTPUTS_DIR`；② `set_seed` / `get_device`；③ `build_transforms`、`TransformSubset`、`stratified_split`（45k/5k 分层切分）、`build_datasets`、`build_loader`；④ `SimpleCNN` / `build_model`；⑤ `load_config`（YAML）、`resolve_data_root`、`resolve_run_dir`、`setup_logger`、`load_checkpoint`、`save_json` |
| `experiments/w1_cifar10/train.py` | **训练入口**。读 YAML → 建 45k/5k → 训练 → 用验证集选 best + 早停 → 写 `tb/`、`metrics.csv`、`summary.*`、`logs/train.log` |
| `experiments/w1_cifar10/eval.py` | **评估入口**。读 `outputs/<run>/config.yaml` 与 `ckpt/best.pth` → 在 test（或 val）上评估 → 整体准确率 + 每类 P/R/F1 + 混淆矩阵 + JSON；支持 `--img-path` 单图推理 |
| `experiments/w1_cifar10/configs/cnn.yaml` | 简单 CNN 的全部超参（数据、模型、优化器、调度器、早停） |
| `experiments/w1_cifar10/configs/resnet.yaml` | ResNet-18 的全部超参（与 cnn 保持一致，只换模型） |
| `experiments/w2_modules/ResNet_blocks.py` | 第 2 周模块：`BasicBlock`、`Bottleneck`，底部 `__main__` 做形状自测 |
| `experiments/w2_modules/DeformConv.py` | 第 2 周模块：`DeformConvBlock`、`DeformConvV2Block`，底部 `__main__` 自测 |
| `experiments/w4_retinanet_voc/README.md` | W4-W5 复现项目：设计思路、与 mmdetection / detectron2 的逐层对应表、文件清单、调用关系、核对表（当前为骨架） |
| `experiments/w4_retinanet_voc/configs/config.py` | 复现项目唯一超参源（路径 / IMG_SIZE / batch / lr / seed / anchor 参数） |
| `experiments/w4_retinanet_voc/{data,models,losses,trainer,utils,tools}/` | 复现项目分层：数据 / 骨干+FPN+头 / 损失与分配 / 训练评估 / 框与 mAP 工具 / 单图推理（当前均为占位文件） |
| `docs/环境配置文档.md` | 硬件 / conda `cv` 环境 / PyTorch 版本 / CUDA 实测值 |
| `docs/refactor_changes.md` | 旧代码 → 新代码的逐行对照，说明每一处改动原因 |
| `requirements.txt` | pip 依赖清单（与 `cv` 环境实测版本一致） |
| `.gitignore` | 忽略 `data/`、`outputs/*/ckpt|tb|logs`、`*/checkpoints/`、`*/runs/`、`reference/`、缓存、编辑器配置 |

---

## 三、脚本调用关系（谁调用谁）

```text
python train.py --config resnet.yaml
  └─ import common
       ├─ load_config()         读 configs/resnet.yaml
       ├─ build_datasets()      45k train / 5k val / 10k test（test 只建不用）
       │    ├─ build_transforms()       训练增强 / 评估确定性
       │    ├─ stratified_split()       按类别各取 500 张做 val
       │    └─ TransformSubset           train 用增强，val 不用
       ├─ build_loader()        DataLoader（训练 shuffle，验证不 shuffle）
       ├─ build_model()         SimpleCNN 或 resnet18
       └─ resolve_run_dir()     outputs/<run>/（ckpt/tb/logs/figures/metrics）
    产出 → outputs/<run>/ckpt/best.pth、tb/、logs/train.log、metrics/metrics.csv、summary.*

python eval.py --run resnet_baseline
  └─ import common
       ├─ load_config()         读 outputs/<run>/config.yaml（训练时写的副本）
       ├─ build_model()         用同一结构重建空网络
       ├─ load_checkpoint()     读 ckpt/best.pth（字典：model_state + 元信息）
       ├─ build_datasets()      need_train=False（只建 test）
       └─ build_loader()
    产出 → outputs/<run>/figures/confusion_matrix_test.png、metrics/eval_test.json、logs/eval_test.log
```

**eval.py 不再 `import train`**：旧写法会触发数据下载、建 DataLoader 等副作用，现在两个脚本只共享 `common.py`。

---

## 四、函数调用链（对应文件与行号）

### train.py

```text
main()                         train.py:299
├─ load_config()               train.py:325      → common.py:383
├─ set_seed()                  train.py:326      → common.py:63
├─ resolve_run_dir()           train.py:330      → common.py:463
├─ setup_logger()              train.py:331      → common.py:500
├─ get_device()                train.py:342      → common.py:82
├─ resolve_data_root()         train.py:345      → common.py:444
├─ build_datasets()            train.py:347      → common.py:216
│   ├─ build_transforms()                          → common.py:109
│   ├─ datasets.CIFAR10(train=False)               （test）
│   ├─ stratified_split()                          → common.py:178
│   └─ TransformSubset ×2                          → common.py:138
├─ build_loader() ×2           train.py:359/361  → common.py:275
├─ build_model()               train.py:366      → common.py:342
│   └─ SimpleCNN                                   → common.py:308
├─ build_optimizer()           train.py:371      → train.py:174
├─ build_scheduler()           train.py:375      → train.py:201（T_max=实际轮数）
├─ SummaryWriter(log_dir=tb)   train.py:379
├─ 每轮：
│   ├─ train_one_epoch()       train.py:400      → train.py:50
│   ├─ evaluate()（val）       train.py:407      → train.py:128
│   └─ torch.save(best.pth)    train.py:441      条件：val_acc > best
└─ save_summary()              train.py:467      → train.py:225 → common.save_json:561
```

### eval.py

```text
main()                         eval.py:208
├─ load_config()               eval.py:239      → common.py:383（读 run 内 config.yaml）
├─ setup_logger()              eval.py:242      → common.py:500
├─ set_seed()                  eval.py:246      → common.py:63
├─ get_device()                eval.py:247      → common.py:82
├─ build_model()               eval.py:250      → common.py:342
├─ load_checkpoint()           eval.py:253      → common.py:542
├─ build_datasets()            eval.py:262      → common.py:216（need_train 由 --split 决定）
├─ build_loader()              eval.py:273      → common.py:275
├─ collect_predictions()       eval.py:277      → eval.py:55
├─ save_classification_report  eval.py:285      → eval.py:90
├─ plot_confusion_matrix()     eval.py:286      → eval.py:130
├─ save_json()                 eval.py:303      → common.py:561
└─ predict_single_image()      eval.py:308      → eval.py:162（仅 --img-path 时）
```

---

## 五、运行步骤

### 0. 环境（每次新开终端都要做）

```powershell
activate base
conda activate cv          # 必须用 cv 环境：base 里的 torch 是 2.5.1+cpu，没有 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.0.1+cu118 True
```

### 1. 进入实验目录

```powershell
cd D:\Research\Code\cv-learning\experiments\w1_cifar10
```

> 脚本里的路径**不依赖**当前目录（`REPO_ROOT` 自动计算），但进入实验目录后配置文件可以简写成 `cnn.yaml`。

### 2. 训练

```powershell
python -u train.py --config cnn.yaml        # 简单 CNN，输出到 outputs/cnn_baseline/
python -u train.py --config resnet.yaml     # ResNet-18，输出到 outputs/resnet_baseline/
```

**不用再写 `Tee-Object` 或 `1>>`**：代码内部已经把日志同时写到终端和 `outputs/<run>/logs/train.log`。

如果你还想保留“窗口 A 训练、窗口 B 跟踪”的习惯：

```powershell
# 窗口 A（训练，终端只显示日志，不刷屏之外的东西）
python -u train.py --config resnet.yaml

# 窗口 B（新开，实时跟踪同一个 run 的日志）
Get-Content outputs\resnet_baseline\logs\train.log -Wait -Tail 20
```

常见的可控参数：

| 参数 | 作用 |
| --- | --- |
| `--run-name resnet_run2` | 换一个 run 目录（同名目录已存在时必须换名或加 `--overwrite`） |
| `--overwrite` | 清空同名 run 重跑（**会删掉该 run 的旧产物**） |
| `--limit-epochs 2 --limit-batches 20` | 冒烟测试：只跑 2 轮、每轮 20 个 batch，几十秒验证整条流水线 |
| `--config cnn.yaml` | 选择配置；相对路径先按 `configs/` 解析 |

### 3. 看曲线

```powershell
tensorboard --logdir outputs
# 浏览器打开 http://localhost:6006；端口冲突加 --port 6007
```

### 4. 评估（测试集只跑一次）

```powershell
python -u eval.py --run cnn_baseline
python -u eval.py --run resnet_baseline
python -u eval.py --run cnn_baseline --img-path assets/plane.jpg   # 顺带单图推理
python -u eval.py --run cnn_baseline --split val                    # 复核选模依据
```

### 5. 第二周模块自测

```powershell
cd D:\Research\Code\cv-learning\experiments\w2_modules
python ResNet_blocks.py
python DeformConv.py
```

---

## 六、产物说明

| 路径 | 内容 | 是否入库 |
| --- | --- | --- |
| `outputs/<run>/config.yaml` | 本次运行的完整配置副本（eval 靠它重建模型） | ✅ |
| `outputs/<run>/summary.md` / `.json` | 最佳验证准确率、最佳轮次、配置、复现命令 | ✅ |
| `outputs/<run>/metrics/metrics.csv` | 每轮 train_loss / train_acc / val_loss / val_acc / lr / 耗时 | ✅ |
| `outputs/<run>/metrics/eval_test.json` | 测试集准确率、macro-F1、权重来源 | ✅ |
| `outputs/<run>/metrics/classification_report_test.csv` | 每类 precision / recall / f1 / support | ✅ |
| `outputs/<run>/figures/confusion_matrix_test.png` | 混淆矩阵（对角线越深越准） | ✅ |
| `outputs/<run>/ckpt/best.pth` | 验证集最优权重（含 epoch / val_acc / config） | ❌ |
| `outputs/<run>/tb/` | TensorBoard 事件文件 | ❌ |
| `outputs/<run>/logs/` | 训练/评估日志 | ❌ |

---

## 七、核心功能核对表

| 要求 | 实现位置 | 怎么验证 |
| --- | --- | --- |
| 数据三分，测试集不参与选模 | `common.py:178`（分层切分）、`train.py:400-408`（train/val）、`eval.py:262`（test） | 训练日志出现 `样本数：train=45000，val=5000，test=10000`；训练过程中没有任何 `test` 指标 |
| 验证集不加随机增强 | `common.py:216-273`（val 用 `eval_transform`） | 同一权重连续两次 `--split val` 准确率一致 |
| 早停基于验证集 | `train.py:437-461` | 日志出现 `验证准确率未提升（n/15）` |
| 余弦退火周期与轮数一致 | `train.py:201-223`、`train.py:375` | 改 YAML 的 `epochs` 后，学习率仍按整段余弦下降 |
| 超参可追溯 | `configs/*.yaml` + `outputs/<run>/config.yaml` | 每个 run 目录都有配置副本 |
| 产物随 run 走 | `common.py:463-498` `resolve_run_dir` | `outputs/<run>/` 下同时有 ckpt/tb/logs/figures/metrics |
| 路径不依赖当前目录 | `common.py:40-46`（`REPO_ROOT` / `OUTPUTS_DIR`） | 从任意目录运行结果一致 |
| 日志自动落盘 | `common.py:500-540`、`train.py:331` | `outputs/<run>/logs/train.log` 与终端内容一致 |
| 评估可复现、可复算 | `eval.py:277-303` | `metrics/eval_test.json` 记录样本数、权重路径、ckpt 轮次 |
| 单图推理可用 | `eval.py:162-206` | `--img-path assets/plane.jpg` 打印 Top-3 |

---

## 八、Git 说明

- 当前分支：`restructure`（重构前状态保存在 `main` 的 `8a9443a`）。
- 本次重构使用 `git mv`，文件历史可追溯（`git log --follow`）。
- 确认无误后再合并：`git switch main && git merge restructure`。
- 数据集、权重、TensorBoard、日志均被 `.gitignore` 排除，不会误传；`figures/`、`metrics/`、`summary.*` 是结论，会入库。
