# 文件：utils/misc.py
# 核心作用：设备选择 + 实验 config 快照 + 指标 JSON 落盘
# 调用关系：trainer/train.py 的 main() 调 dump_run_config；trainer/eval.py 调 save_json
import json                                          # 写 JSON
import types                                         # 判断"模块/函数/类"，用来过滤 import 进来的东西
from pathlib import Path                             # 路径
import torch                                         # 判断 CUDA
import config.config as C                            # 读取全部参数（config.py 里是小写命名）


def get_device():
    """功能：返回可用设备
    输入：无
    输出：torch.device
    调用位置：trainer/train.py、trainer/eval.py、tools/predict.py
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dump_run_config(run_dir, extra=None):
    """功能：把 config.py 里的全部参数（小写命名） + 命令行覆盖项写成 config.json
    输入：run_dir（Path/str）、extra（dict，可选，如 epochs/batch）
    输出：无（写 run_dir/config.json）
    调用位置：trainer/train.py 的 main()
    注意：不能用 k.isupper() 过滤，因为本项目的参数名是小写（img_size 等），会全部漏掉
    """
    run_dir = Path(run_dir)                           # 统一成 Path
    run_dir.mkdir(parents=True, exist_ok=True)         # 确保目录存在
    cfg = {}                                           # 待落盘的 config
    for k, v in vars(C).items():                       # 遍历 config.py 的模块属性
        if k.startswith("_"):                          # 跳过 __name__ / __file__ 等内部属性
            continue
        if isinstance(v, (types.ModuleType, types.FunctionType, type)):  # 跳过 import 进来的模块/函数/类（如 Path）
            continue
        cfg[k] = str(v) if isinstance(v, Path) else v   # Path 不能直接进 JSON，转字符串
    if extra:                                          # 命令行覆盖项优先（如 --epochs / --batch）
        cfg.update(extra)
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def save_json(path, obj):
    """功能：把评估指标写到 JSON
    输入：path（Path/str）、obj（dict）
    输出：无
    调用位置：trainer/eval.py
    """
    path = Path(path)                                 # 统一成 Path
    path.parent.mkdir(parents=True, exist_ok=True)     # 确保目录存在
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")