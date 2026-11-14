# 第 1 周：CIFAR-10 图像分类

> 详细的项目级说明（目录、调用关系、函数调用链、核对表）见仓库根 [README.md](../../README.md)。
> 本文件只写“这个实验目录里有什么、怎么跑”。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `common.py` | 公共库：路径、种子、设备、数据切分、模型工厂、YAML、日志、权重读写（**不单独运行**） |
| `train.py` | 训练入口：45k 训练 + 5k 验证，按验证集选 best、早停 |
| `eval.py` | 评估入口：10k 测试集只跑一次，输出准确率/每类指标/混淆矩阵；可选单图推理 |
| `configs/cnn.yaml` | 简单 CNN 的全部超参 |
| `configs/resnet.yaml` | ResNet-18 的全部超参 |
| `assets/plane.jpg` | 单图推理示例 |
| `outputs/<run>/` | 每次 run 的产物（权重/TensorBoard/日志不上传，figures/metrics/summary 上传） |

## 调用关系

```text
train.py ──> common.py（build_datasets → stratified_split / TransformSubset；build_loader；build_model）
train.py ──产出──> outputs/<run>/{config.yaml, ckpt/best.pth, tb/, logs/train.log, metrics/metrics.csv, summary.*}
eval.py  ──> common.py（load_config / build_model / load_checkpoint / build_datasets）
eval.py  ──读取──> outputs/<run>/{config.yaml, ckpt/best.pth}
eval.py  ──产出──> outputs/<run>/{figures/, metrics/, logs/eval_<split>.log}
```

## 运行步骤

```powershell
activate base && conda activate cv
cd D:\Research\Code\cv-learning\experiments\w1_cifar10

python -u train.py --config cnn.yaml        # 或 resnet.yaml
tensorboard --logdir outputs                # http://localhost:6006
python -u eval.py --run cnn_baseline        # 测试集只跑一次
python -u eval.py --run cnn_baseline --img-path assets/plane.jpg
```

快速冒烟（几十秒验证流水线是否正常）：

```powershell
python -u train.py --config cnn.yaml --run-name smoke --limit-epochs 2 --limit-batches 20 --overwrite
python -u eval.py --run smoke --limit-batches 4
```

## 数据协议

- 训练：50k 训练集中的 **45k**（带 RandomCrop + HorizontalFlip 增强）
- 验证：50k 训练集中的 **5k**（按类别各 500 张，不加增强，用来选 best / 早停）
- 测试：**10k** 官方测试集，只在 `eval.py` 使用一次
- 切分种子 `seed=42`，`eval.py --split val` 可复现同一份验证集

## 核对表

| 项 | 状态 / 验证方式 |
| --- | --- |
| train/val/test 互不重叠 | 日志打印 45000 / 5000 / 10000；`stratified_split` 按类别取索引 |
| 测试集不参与训练与选模 | 训练日志里只有 train/val 指标 |
| 配置可追溯 | 每个 run 目录有 `config.yaml` + `summary.md` |
| 权重选模依据 | `ckpt/best.pth` 内记录 `val_acc` / `epoch` |
| 单图推理预处理一致 | `eval.py` 里 Resize(32,32) + Normalize(0.5) |
