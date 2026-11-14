# 第 2 周：模块 demo

两个文件都是**独立可运行**的模块演示，底部带 `if __name__ == "__main__"` 形状自测，不依赖第 1 周的实验代码。

| 文件 | 作用 | 运行 |
| --- | --- | --- |
| `ResNet_blocks.py` | `BasicBlock`（两层 3x3 残差块）、`Bottleneck`（1x1-3x3-1x1 瓶颈块），演示 stride 与通道变化对输出形状的影响 | `python ResNet_blocks.py` |
| `DeformConv.py` | `DeformConvBlock`（可变形卷积 v1：学习 offset）、`DeformConvV2Block`（v2：额外学习 modulation mask） | `python DeformConv.py` |

```powershell
activate base && conda activate cv
cd D:\Research\Code\cv-learning\experiments\w2_modules
python ResNet_blocks.py
python DeformConv.py
```

预期：两个脚本都打印各配置下的输出张量形状，最后输出成功提示（如 `✅ 全部模块运行成功！`）。

调用关系：本目录文件**不 import** 其它实验目录；以后若某个模块在第二个实验里被复用，再提升到仓库根的 `shared/`（只提升被复用两次以上的代码）。
