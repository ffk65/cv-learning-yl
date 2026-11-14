import torch
import torch.nn as nn

# todo  ========================== 1. 基础残差块 BasicBlock (ResNet18/34) ==========================
class BasicBlock(nn.Module):
    """
    BasicBlock: 两个 3x3 卷积 + BN + ReLU，残差连接
    结构：
        conv1 -> bn1 -> relu -> conv2 -> bn2
        最后 out = x + 残差分支
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长，stride=2 用来下采样、缩小特征图尺寸
        downsample: 下采样模块，当通道变多 / 尺寸变小的时候，用来对齐 shortcut 的维度
    """
    expansion = 1  # BasicBlock 最后输出通道 = out_channels * 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # tod 第一个3×3卷积，stride可变，可能是1或2，padding=1保持尺寸不变（stride=1）或减半（stride=2）
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # BN层，标准化，加速训练
        self.relu = nn.ReLU(inplace=True)        # inplace=True 原地操作，省显存

        # 第二个3×3卷积，stride固定=1，不再缩小尺寸
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample  # shortcut分支的下采样模块

    def forward(self, x):
        # 保存原始输入，作为残差shortcut
        identity = x

        # 主分支卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样（通道/尺寸对不上），对identity做变换
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加：核心！H(x)=F(x)+x
        out += identity
        out = self.relu(out)
        return out


# todo ========================== 2. Bottleneck 瓶颈块 (ResNet50/101/152) ==========================
class Bottleneck(nn.Module):
    """
    Bottleneck：1×1降维 →3×3卷积 →1×1升维
    expansion=4：输出通道 = out_channels * 4
    好处：减少参数量，深层大网络用这个
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1x1卷积：降通道
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3卷积：空间特征提取
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1卷积：升通道 × expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# todo ========================== 3. 测试 Demo：逐个跑模块，打印张量shape ==========================
if __name__ == "__main__":
    # 构造模拟输入：batch_size=2, 通道=64, H=32, W=32
    # 含义：2张图，每张64个特征通道，高32宽32
    x = torch.randn(2, 64, 32, 32)
    print("=" * 50)
    print(f"输入张量 shape [B,C,H,W] : {x.shape}")
    print("=" * 50)


    # 残差块可能出现尺寸变化/通道变化,瓶颈块默认通道一定变化
    # todo ---------- 一、测试 BasicBlock ----------
    print("\n>>> 测试 BasicBlock")
    basic_block = BasicBlock(in_channels=64, out_channels=64, stride=1)
    out_basic = basic_block(x)
    print(f"BasicBlock 输出 shape: {out_basic.shape}")

    # 测试 stride=2（下采样版本，尺寸减半）
    print("\n>>> 测试 BasicBlock stride=2 (下采样)")
    # downsample：通道对齐 + 尺寸对齐
    downsample = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(64)
    )

    # 残差快可能出现尺寸变化/通道变化，stride=2--卷积主干尺寸减半，因此需要downsample对齐
    basic_block_down = BasicBlock(in_channels=64, out_channels=64, stride=2, downsample=downsample)
    out_basic_down = basic_block_down(x)
    print(f"BasicBlock stride=2 输出 shape: {out_basic_down.shape}")

    # todo ---------- 测试 Bottleneck ----------
    print("\n>>> 测试 Bottleneck")
    # 【修复关键】为Bottleneck添加downsample，对齐通道64→256
    downsample_bottle = nn.Sequential(
        nn.Conv2d(64, 64 * Bottleneck.expansion, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(64 * Bottleneck.expansion)
    )

                       # 瓶颈块默认通道一定变化，因此要1x1卷积对齐通道数
    bottleneck = Bottleneck(in_channels=64, out_channels=64, stride=1, downsample=downsample_bottle)
    out_bottle = bottleneck(x)
    print(f"Bottleneck 输出 shape: {out_bottle.shape}")
    # Bottleneck expansion=4，输出通道 64*4=256，可以看到C变成256


    print("\n✅ 全部模块运行成功！")




    