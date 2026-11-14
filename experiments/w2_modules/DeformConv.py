# ========================== 1. 导入必要的库 ==========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d, deform_conv2d


# ========================== 2. DeformConv V1 模块（你原来的，只有offset，无mask） ==========================
class DeformConvBlock(nn.Module):
    """
    可变形卷积V1模块（最关键：先预测offset偏移量，再送入DeformConv2d）
    原理：
        普通卷积：采样点固定网格
        可变形卷积V1：每个采样点额外学习 (dx, dy) 偏移，采样位置跟着物体形状走
    注意：
        DeformConv2d 输入有两个：特征图 + offset偏移图
        offset通道数 = 2 * kernel_size * kernel_size
        每个卷积核采样点，都有x偏移、y偏移两个值
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        # offset通道：每个采样点 x,y 两个偏移
        offset_channels = 2 * kernel_size * kernel_size

        # 分支1：预测偏移量 offset
        self.offset_conv = nn.Conv2d(
            in_channels, offset_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        # 分支2：可变形卷积V1
        self.deform_conv = DeformConv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 第一步：根据输入特征预测偏移 offset
        offset = self.offset_conv(x)
        # 第二步：把特征图+offset送入可变形卷积
        out = self.deform_conv(x, offset)
        out = self.bn(out)
        out = self.relu(out)
        return out


# ========================== 3. 【新增】DeformConv V2 模块（offset + modulation mask，论文原版v2） ==========================
class DeformConvV2Block(nn.Module):
    """
    Deformable Convolution v2：偏移offset + 调制mask
    和你单独的维度验证demo逻辑完全对齐，底层调用 deform_conv2d functional
    V1：只学采样点偏移
    V2：学采样点偏移 + 每个采样点的权重mask（0~1之间，动态抑制不重要采样点）
    通道规则：
        offset_channels = 2 * k * k  每个点dx、dy
        mask_channels   = 1 * k * k  每个点一个调制系数
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        offset_channels = 2 * kernel_size * kernel_size
        mask_channels = kernel_size * kernel_size

        # 卷积分支1：预测offset
        self.offset_conv = nn.Conv2d(
            in_channels, offset_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        # 卷积分支2：预测modulation mask
        self.mask_conv = nn.Conv2d(
            in_channels, mask_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )

        # 可变形卷积权重，注册为可训练参数，和普通Conv2d的weight一样
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = None

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 预测偏移
        offset = self.offset_conv(x)
        # mask经过sigmoid限制在(0,1)，是v2标准操作
        mask = torch.sigmoid(self.mask_conv(x))

        # 调用底层functional算子，支持mask输入，就是你demo里的调用方式
        out = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask
        )
        out = self.bn(out)
        out = self.relu(out)
        return out



# ========================== 4. 测试 Demo：逐个跑模块，打印张量shape ==========================
if __name__ == "__main__":
    # 构造模拟输入：batch_size=2, 通道=64, H=32, W=32
    # 含义：2张图，每张64个特征通道，高32宽32
    x = torch.randn(2, 64, 32, 32)

    # ---------- 测试 DeformConvBlock V1（你原来的） ----------
    print("\n>>> 测试 DeformConvBlock V1（只有offset，无mask）")
    deform_block = DeformConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    out_deform = deform_block(x)
    print(f"DeformConv V1 输出 shape: {out_deform.shape}")

    # ---------- 【新增】测试 DeformConvV2Block V2（offset + mask） ----------
    print("\n>>> 测试 DeformConvV2Block V2（offset + modulation mask）")
    deform_v2_block = DeformConvV2Block(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    out_deform_v2 = deform_v2_block(x)
    print(f"DeformConv V2 输出 shape: {out_deform_v2.shape}")
