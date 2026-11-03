# cv-learning

> 山东大学 · 控制科学与工程学院 · 科研入门
> 作者：袁亮　远端仓库：<https://github.com/ffk65/cv-learning-yl.git>
> 最近更新：2026-10-19

## 项目简介

按《CV 科研入门任务书》，用 PyTorch 复现 **CIFAR-10 图像分类**，走通「数据 → 网络定义 → 训练 → 评估 → 可视化」全流程。

**当前进度（第 1 周）**：简单 CNN 与 ResNet-18 两种模型均已训练、评估完成，50 轮跑满（未触发早停），测试集准确率：

| 模型 | 测试集准确率（最优） | 权重文件 |
| --- | --- | --- |
| 简单 CNN（3 层卷积） | **0.8577** | `CIFAR-10/checkpoints/cnn_best_model.pth` |
| ResNet-18（随机初始化） | **0.8555** | `CIFAR-10/checkpoints/resnet_best_model.pth` |

---

## 一、目录结构

```text
cv-learning/
├── README.md                      ← 本文件
├── .gitignore                     ← 告诉 Git 哪些本地大文件/缓存不上传
├── CIFAR-10/                      ← 第 1 周：CIFAR-10 图像分类实验
│   ├── train.py                   ← 训练脚本（数据 + 网络 + 训练循环）
│   ├── eval.py                    ← 评估脚本（准确率 + 混淆矩阵）
│   ├── dataset/                   ← CIFAR-10 原始数据，约 340MB          【不上传】
│   ├── runs/                      ← TensorBoard 事件日志（cifar10_cnn / cifar10_resnet）【不上传】
│   ├── checkpoints/               ← 训练权重 cnn_best_model.pth / resnet_best_model.pth 【不上传】
│   ├── __pycache__/               ← Python 字节码缓存                            【不上传】
│   └── .vscode/                   ← VS Code 个人配置                             【不上传】
├── figures/                       ← 实验产出（图片 + 曲线导出，全部上传）
│   ├── cnn_confusion_matrix.png   ← eval.py 产出（简单 CNN 混淆矩阵）
│   ├── resnet_confusion_matrix.png← eval.py 产出（ResNet-18 混淆矩阵）
│   ├── cnn_eval.jpg               ← 评估运行截图（终端输出，运行证据）
│   ├── resnet_eval.jpg            ← 评估运行截图（ResNet-18）
│   ├── accuracy_test.svg          ← TensorBoard 曲线导出（准确率）
│   ├── loss_train.svg             ← TensorBoard 曲线导出（训练损失）
│   ├── lr.svg                     ← TensorBoard 曲线导出（学习率）
│   └── run-cifar10_{cnn,resnet}-tag-*.csv   ← 上述曲线对应的原始数据点
├── logs/                          ← 终端输出留档目录（当前为空，备用）       【不上传】
└── docs/
    └── 环境配置文档.md            ← 第 1 周交付物：环境配置全过程记录
```

---

## 二、文件清单与作用

| 文件 / 目录 | 作用 |
| --- | --- |
| `CIFAR-10/train.py` | **训练主脚本**。三段合一：① 读取 CIFAR-10（Dataset / DataLoader，含随机裁剪 + 翻转的数据增强）② 定义网络（`simpleCNN` 与 `get_resnet18`，用命令行 `--mod` 选择）③ 训练循环（保存最佳权重、早停、余弦退火学习率、写 TensorBoard） |
| `CIFAR-10/eval.py` | **评估脚本**。读回最佳权重 → 在测试集上算准确率 → 打印混淆矩阵 → 保存混淆矩阵图 |
| `figures/` | 实验产出统一归档处。成品图带实验前缀（`cnn_` / `resnet_`），便于多实验横向对比 |
| `logs/` | 终端输出留档目录，**本地累积、不上传**（`.gitignore` 已排除）。当前为空，跑长任务时可用输出重定向留档 |
| `docs/环境配置文档.md` | 第 1 周交付物：硬件、Conda / PyTorch / 显卡驱动、Git 等环境配置全过程 |
| `.gitignore` | 排除 `dataset/ runs/ checkpoints/ logs/ __pycache__/ *.pth .vscode/` 等本地文件 |

---

## 三、文件之间的关系（谁产出什么、谁读什么）

```text
python train.py --mod cnn            （--mod 可选 cnn / resnet）
   ├─ 读入 → CIFAR-10/dataset/                         ← CIFAR-10 原始数据
   ├─ 产出 → CIFAR-10/checkpoints/{mod}_best_model.pth ← 最佳权重（只存权重数字，不含网络结构）
   ├─ 产出 → CIFAR-10/runs/cifar10_{mod}/              ← TensorBoard 事件日志
   └─ 屏幕输出 → 每轮 loss / 准确率 / 学习率（可重定向进 logs/）

python eval.py --mod cnn
   ├─ 读入 → CIFAR-10/checkpoints/{mod}_best_model.pth ← 把权重灌回网络
   ├─ 读入 → train.py                                 ← 只"借"数据、设备与建网络的函数，不会触发训练
   └─ 产出 → figures/{mod}_confusion_matrix.png       ← 混淆矩阵图
```

> **为什么 `eval.py` 要 `import train`**：两个脚本是**两次独立运行**。`train.py` 跑完进程结束，内存里的模型消失，只留下 `.pth` 文件；而 `.pth` **只存权重数字、不存网络结构**。所以 `eval.py` 必须用**同一个建网络的函数（`select_model()`）**先搭出结构一样的空网络，再把权重灌进去。
>
> `train.py` 里的训练代码都写在 `if __name__ == '__main__':` 里，所以 `eval.py` 导入它时只会执行顶层的「下载数据、建 DataLoader、定义网络」部分，**不会真的开始训练**。
>
> ⚠️ 因此两个脚本的 `--mod` **必须一致**（如训练用 `--mod cnn`，评估也必须 `--mod cnn`），否则会报 `size mismatch` / 找不到权重文件。

---

## 四、运行方法

**环境**：Python 3.10 + PyTorch 2.0.1+cu118（详见 [docs/环境配置文档.md](docs/环境配置文档.md)）。
**准备**：每条新命令行窗口先 `activate base` → `conda activate cv`（⚠️ 不要直接敲 `python`，会命中别的 Python）。

```powershell
cd D:\Research\Code\cv-learning\CIFAR-10     # 1. 进入脚本目录（脚本里的相对路径都基于这里）

python train.py --mod cnn                    # 2. 训练简单 CNN（默认即 cnn）
python train.py --mod resnet                 #    或训练 ResNet-18
python eval.py  --mod cnn                    # 3. 评估，产出 figures/cnn_confusion_matrix.png

tensorboard --logdir runs                    # 4. 看曲线，浏览器打开 http://localhost:6006
```

**可选：把终端输出落盘留档**（跑长任务时推荐，跑完就在 `logs/` 留一份完整记录）：

```powershell
python train.py --mod resnet 1>> ..\logs\cifar10_resnet_run1.log 2>&1
python eval.py  --mod resnet 1>> ..\logs\cifar10_resnet_eval1.log 2>&1

```

**1.输出同时存文件**
python -u train.py --mod resnet | Tee-Object ..\logs\cifar10_resnet_run1.log

**2.窗口A：训练，窗口干净，不刷屏**
python -u train.py --mod resnet 1>> ..\logs\cifar10_resnet_run1.log 2>&1
**窗口B：新开，单独跟踪**
Get-Content ..\logs\cifar10_resnet_run1.log -Wait





- `1>>` = 把**正常输出**追加写进文件；`2>&1` = 把**报错信息**也并进同一个文件。
- `tensorboard` 端口被占 → 加 `--port 6007`。

---

## 五、结果

| 产物 | 说明 |
| --- | --- |
| `figures/cnn_confusion_matrix.png`、`figures/resnet_confusion_matrix.png` | 两个模型的混淆矩阵，对角线越深代表分得越准 |
| `figures/accuracy_test.svg`、`loss_train.svg`、`lr.svg` | 测试准确率 / 训练损失 / 学习率三条曲线（TensorBoard 导出） |
| `figures/run-cifar10_*-tag-*.csv` | 上述曲线的原始数据点，便于自己画图或对比 |
| `figures/cnn_eval.jpg`、`resnet_eval.jpg` | 评估运行时的终端截图（准确率与混淆矩阵输出） |

> 两个模型准确率接近（0.86 上下），说明在当前超参（SGD lr=0.01、50 轮）下，ResNet-18 相对简单 CNN **没有明显优势**——这是后续调参 / 加预训练权重时可以继续观察的点。
