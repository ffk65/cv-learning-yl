# 功能：新实验通用开头模板——固定所有随机源 + 选择计算设备 + 打印环境信息
# 输入参数：seed（随机种子，默认 42）
# 输出参数：无返回值；运行时打印设备与 torch 版本，用于确认环境没跑错
# 调用位置：把 set_seed() 复制进任何新实验脚本的 main() 第一行；get_device() 用于选 CPU/GPU
import os          # 用来设置 PYTHONHASHSEED 环境变量
import random      # Python 自带随机数（数据抽样、shuffle 会用到）
import numpy as np # numpy 随机数（数据增强、初始化可能用到）
import torch       # PyTorch 主库

def set_seed(seed: int = 42) -> None:
    """功能：把 Python / NumPy / PyTorch（含 GPU）的随机源全部固定，保证实验可复现
    输入参数：seed —— 随机种子，整型
    输出参数：无
    调用位置：每个实验脚本训练开始前调用一次（通常是 main() 第一行）"""
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定 Python 自带随机数种子
    random.seed(seed)                         # 固定 Python 自带随机数种子
    np.random.seed(seed)                      # 固定 numpy 随机数种子
    torch.manual_seed(seed)                   # 固定 CPU 上的随机数种子
    torch.cuda.manual_seed(seed)              # 固定当前 GPU 上的随机数种子
    torch.cuda.manual_seed_all(seed)          # 固定所有 GPU 上的随机数种子
    torch.backends.cudnn.deterministic = True # 保证卷积算法可复现
    torch.backends.cudnn.benchmark = False    # 保证卷积算法可复现


def get_device() -> torch.device:
    """功能：有 GPU 就用 GPU，没有就退回 CPU
    输入参数：无
    输出参数：torch.device 对象（'cuda' 或 'cpu'）
    调用位置：模型 .to(device)、数据 .to(device) 之前调用"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    set_seed(42)                                   # 先固定种子
    dev = get_device()                             # 再选设备
    print('device:', dev, '| torch:', torch.__version__)   # 打印，确认没跑到 CPU 上
    a = torch.randn(3, 3, device=dev)              # 生成一个随机张量作为"指纹"
    print('第一个随机数:', a[0, 0].item())           # 连跑两次，这个数应当完全相同