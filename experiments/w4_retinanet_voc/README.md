# w4_retinanet_voc —— 简化版 RetinaNet（VOC2007 子集）

> 任务书第 4–5 周复现项目，选项 A：**ResNet-18 + FPN + Focal Loss + VOC subset**。
> 硬指标不是 mAP 40%，而是：**能跑通训练 / 能算出 mAP / 能讲清每个模块**。
> 当前状态：**设计与目录已定稿，代码按任务书副本“步骤 3–14”逐个填充**（步骤 1.1 已写本文件的设计思路）。

---

## 一、设计思路（任务书要求：先写 README 说清设计思路）

1. **整体**：单阶段检测器，直接在 3 个尺度的特征图上密集预测“类别 + 框偏移”，不需要 RPN 和第二阶段。
2. **backbone**：ResNet-18 取 C3/C4/C5（stride 8/16/32），负责从图像提多尺度特征。
3. **neck**：FPN 自顶向下 + 横向连接，把 C3/C4/C5 融合成 P3/P4/P5，每层统一 256 通道，让小目标也有强语义。
4. **head**：每个特征位置放 9 个 anchor（3 尺度 × 3 比例），输出 20 类 logits + 4 个框偏移。
5. **损失**：IoU ≥ 0.5 为正样本、< 0.4 为负样本，中间忽略；分类用 Focal Loss（γ=2, α=0.25），回归用 smooth L1。
6. **评估**：按 VOC 口径算每类 AP，再平均得 mAP@0.5；推理时先 sigmoid 过滤，再 NMS。

**与参考框架的关系**：只参考 mmdetection / detectron2 的**分层与默认超参**，不抄注册器、配置继承、分布式和 `img_metas` 包装。

| 层 | mmdetection / detectron2 | 本项目 |
| --- | --- | --- |
| 入口 | `tools/train.py` | `trainer/train.py`、`trainer/eval.py` |
| 配置 | `configs/` + 继承 | `config/config.py`（唯一参数源） |
| 数据 | `datasets/voc.py` | `data/voc.py`、`data/make_split.py` |
| 骨干 | `models/backbones/resnet.py` | `models/backbone.py`（`Backbone`） |
| 颈部 | `models/necks/fpn.py` | `models/neck.py`（`FPN`） |
| 头 | `models/dense_heads/retina_head.py` | `models/head.py`（`Head`） |
| 整机 | `models/detectors/retinanet.py` | `models/retinanet.py`（`RetinaNet`） |
| anchor | `anchor_generator.py` | `models/anchor.py` |
| 损失 | `losses/focal_loss.py` | `losses/loss.py`（`Loss`） |
| 评估 | `evaluation/metrics/` | `utils/metric.py` |

---

## 二、文件清单 + 每个文件作用

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `config/config.py` | 唯一参数源：`root`、`data_root`、`split_dir`、`img_size`、`batch_size`、`lr`、`seed`、anchor 参数 | 待写（步骤 3） |
| `config/__init__.py` | 包标记，使 `python -m trainer.train` 能 import `config.config` | 待建（步骤 2） |
| `data/make_split.py` | VOC2007 `trainval.txt` 按 8:2 切 `data/splits/train.txt` / `val.txt` | 待写（步骤 8.1） |
| `data/voc.py` | `VOCDataset`（jpg + xml → image/boxes/labels）+ `collate_fn` + 自检入口 | 待写（步骤 8.2） |
| `data/splits/` | 存放切分后的图片 id 列表（由 `make_split.py` 生成，入库） | 待生成 |
| `models/backbone.py` | `Backbone`：ResNet-18 取 C3/C4/C5 | 待写（步骤 4） |
| `models/neck.py` | `FPN`：C3/C4/C5 → P3/P4/P5（256 通道） | 待写（步骤 4） |
| `models/anchor.py` | 多尺度 anchor 生成（18900 个）+ `encode` / `decode` | 待写（步骤 5） |
| `models/head.py` | `Head`：每层输出 9×(20+4) 的 cls/reg | 待写（步骤 6） |
| `models/retinanet.py` | `RetinaNet` 整机 + `flatten_outputs` + 自检入口 | 待写（步骤 7） |
| `losses/loss.py` | `Loss`：`assign_targets`（IoU 0.5/0.4）+ `sigmoid_focal_loss` + smooth L1 | 待写（步骤 10） |
| `utils/box.py` | `box_iou`、`clip_boxes`、`xyxy↔xywh`、`encode` / `decode` | 待写（步骤 9） |
| `utils/metric.py` | VOC AP / mAP@0.5（全点插值） | 待写（步骤 13） |
| `utils/misc.py` | `get_device` / `dump_run_config` / `save_json` | 待写（步骤 12） |
| `trainer/train.py` | 训练入口：数据 → 模型 → loss → 反向 → ckpt → TensorBoard | 待写（步骤 12） |
| `trainer/eval.py` | 评估入口：加载 `checkpoints/last.pth` → 推理 → mAP | 待写（步骤 13） |
| `tools/overfit_one.py` | 单图过拟合验收（1 张图 loss 降到 <1.0） | 待写（步骤 11） |
| `tools/predict.py` | 单图推理：解码 + NMS + 画框保存 | 待写（步骤 14） |
| `checkpoints/` | 权重输出目录（`*.pth` 被 `.gitignore` 忽略） | 空目录 |
| `runs/` | TensorBoard 事件目录（被 `.gitignore` 忽略） | 空目录 |
| `docs/experiments/` | 实验日志 `EXP-001.md`、`EXP-002.md` | 待写（步骤 15） |
| `docs/figures/` | 曲线截图、预测图 | 待生成 |
| `docs/repro_retinanet.md` | 复现报告（6 节） | 待写（步骤 15） |

> **忽略规则**：仓库根 `.gitignore` 只忽略仓库根 `data/`（CIFAR-10 原始数据，`/data/` 已锚定）、本项目的 `checkpoints/`、`runs/`、`*.pth` 和 `__pycache__`；`data/*.py` 和 `data/splits/*.txt` 会正常入库，组会时能看到切分文件。

---

## 三、脚本调用关系

```text
config/config.py                     # 参数源，被所有模块 import，自己不 import 项目内模块
   ↑
data/make_split.py  ──生成──> data/splits/{train,val}.txt
   ↑                                   ↑
   └────────────── data/voc.py ────────┘        # VOCDataset + collate_fn
                        ↑
        ┌───────────────┴───────────────────────────┐
trainer/train.py      trainer/eval.py          tools/predict.py
   │ build_loader()      │ build_loader()          │ load_image()
   │ RetinaNet(...)      │ RetinaNet(...)          │ RetinaNet(...)
   │ Loss(...)           │ decode + NMS            │ decode + NMS
   └─ checkpoints/*.pth  └─ utils/metric.py→mAP   └─ 画框保存
      runs/（TensorBoard）
```

依赖方向单向：`config → utils → models → losses → data → trainer/tools`；`utils/` 不 import 任何项目内模块。

---

## 四、函数调用链 + 文件 + 代码行（实现后回填）

| 入口 | 调用链 | 文件:行 |
| --- | --- | --- |
| `python -m data.make_split` | `main()` → 读 `ImageSets/Main/trainval.txt` → 写 `splits/*.txt` | 待回填 |
| `python -m data.voc` | `__main__` → `VOCDataset.__getitem__` → `collate_fn` | 待回填 |
| `python -m models.retinanet` | `__main__` → `RetinaNet.forward` → `Backbone.forward` → `FPN.forward` → `Head.forward` → `generate_anchors` | 待回填 |
| `python -m tools.overfit_one` | `main()` → `VOCDataset` → `RetinaNet.forward` → `Loss.forward` → `backward` | 待回填 |
| `python -m trainer.train` | `main()` → `set_seed` → `build_loader` → `RetinaNet.forward` → `Loss.forward` → `dump_run_config` → `save` | 待回填 |
| `python -m trainer.eval` | `main()` → `load_state_dict` → `RetinaNet.forward` → `decode + NMS` → `compute_map` → `voc_ap` | 待回填 |
| `python -m tools.predict` | `main()` → `load_image` → `RetinaNet.forward` → `decode + NMS` → 画框 | 待回填 |

> 每实现一个文件，把该文件顶部的“功能 / 输入 / 输出 / 调用位置”注释补全，并回到本表填上真实行号。

---

## 五、运行步骤

```powershell
# 0. 环境（必须是 cv，base 里是 CPU 版 torch）
conda activate cv
cd D:\Research\Code\cv-learning\experiments\w4_retinanet_voc
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 期望 2.0.1+cu118 True

# 1. 数据（VOC2007 在 D:\Research\Data\VOCdevkit\VOC2007，约 870 MB；步骤 1）
python -m data.make_split        # 期望 train=4009, val=1002
python -m data.voc               # 期望 image (3,320,320) / boxes (N,4) / labels (N,)

# 2. 模型 shape 自检（不训练）
python -m models.retinanet       # 期望三层 cls/reg shape + anchors (18900,4)

# 3. 单图过拟合（链路自检）
python -m tools.overfit_one      # 期望 loss 从 ~4.x 降到 <1.0

# 4. 训练 + 曲线
python -m trainer.train --epochs 12 --exp EXP-002
tensorboard --logdir runs

# 5. 评估
python -m trainer.eval --ckpt checkpoints/last.pth --exp EXP-002

# 6. 单图推理
python -m tools.predict --img <图片路径> --ckpt checkpoints/last.pth
```

数据位置约定：`D:\Research\Data\VOCdevkit\VOC2007\`。若换位置，只改 `config/config.py` 的 `data_root` 一行。

---

## 六、核心功能核对表

| 要求（任务书 / 任务书副本步骤） | 实现位置 | 怎么验证 | 状态 |
| --- | --- | --- | --- |
| 先写 README 说清设计思路 | 本文件第一节 | 能对着讲清 backbone/neck/head/loss 四段 | ✅ 已完成 |
| 目录符合 `config/ models/ trainer/ utils/` | 本目录 | `tree /F` 对照 | 待步骤 2 |
| seed 固定 | `config/config.py` `seed` | 两次训练首轮 loss 一致 | 待步骤 3/12 |
| VOC 数据可读、框在图像范围内 | `data/voc.py` | `python -m data.voc` 打印 shape / boxes | 待步骤 8 |
| FPN 输出 P3/P4/P5 形状正确 | `models/neck.py` | 自检打印 `(256,40,40)/(256,20,20)/(256,10,10)` | 待步骤 4 |
| anchor 总数 = 18900 | `models/anchor.py` | 自检断言 1600×9 + 400×9 + 100×9 | 待步骤 5 |
| Focal Loss 数值随 `p_t` 单调下降 | `losses/loss.py` | 对照 7.6 笔记的曲线表 | 待步骤 10 |
| 正负样本按 IoU 0.5/0.4 分配 | `losses/loss.py` `assign_targets` | 打印正样本数 / 每图目标数 | 待步骤 10 |
| 单图能过拟合 | `tools/overfit_one.py` | loss 从 ~4.x 降到 <1.0 | 待步骤 11 |
| 训练曲线可看（loss / lr） | `trainer/train.py` + `runs/` | `tensorboard --logdir runs` | 待步骤 12 |
| mAP 能算出来 | `trainer/eval.py` + `utils/metric.py` | 打印每类 AP + mAP@0.5 | 待步骤 13 |
| 报告写“改了什么 / 差多少 / 为什么” | `docs/repro_retinanet.md` | 对照任务书副本步骤 15 | 待步骤 15 |

---

## 七、以后拆成独立 git 仓库

本目录按“**自包含**”设计：只依赖 `config/` 内参数，不 import `experiments/` 里的其他项目，数据路径可由 `config.py` 一处改成绝对路径。因此成熟后可以整体搬走：

```powershell
# 1. 复制出去（不要用 git subtree，学习仓历史不需要带过去）
Copy-Item -Recurse D:\Research\Code\cv-learning\experiments\w4_retinanet_voc D:\Research\Code\detect-repro

# 2. 独立成仓
cd D:\Research\Code\detect-repro
git init
git add .
git commit -m "feat: 简化版 RetinaNet（ResNet-18 + FPN + Focal Loss）复现"
```

搬走前需要补的两件事：① `config/config.py` 的 `data_root` 改成绝对路径或环境变量；② 新增独立的 `requirements.txt`。

**通用函数策略**：`set_seed` / `get_device` / 日志 / JSON 这类工具，在**每个项目自己的 `utils/` 内放一份**（本项目放 `utils/misc.py`），**不跨项目 import**。代价是少量重复，换来“整目录搬走即可运行”。
