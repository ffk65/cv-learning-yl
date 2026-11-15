# 文件：data/make_split.py
# 核心作用：把 VOC2007 的 ImageSets/Main/trainval.txt 切成 train.txt / val.txt，供 Dataset 读取----只做一次，切分后就不再需要
# 调用关系：独立脚本（python -m data.make_split）；产物 data/splits/*.txt 被 data/voc.py 的 VOCDetection.__init__ 读取
# 输入：ImageSets/Main/trainval.txt（5011 个图片 id）
# 输出：data/splits/train.txt（约 4009 行）、data/splits/val.txt（约 1002 行）
 
import random                                                  # 用固定种子做可复现的随机切分
from config.config import data_root, split_dir, seed           # 路径与种子常量（见 config/config.py）


def main():
    """功能：生成 8:2 的训练/验证图片 id 列表
    输入：无（路径来自 config/config.py 的 data_root、split_dir）
    输出：data/splits/train.txt、data/splits/val.txt（每行一个图片 id）
    调用位置：仅本脚本 __main__ 调用一次
    """
    # 读取官方 trainval 列表（5011 个 id，VOC2007 官方数值）
    raw = (data_root / "ImageSets" / "Main" / "trainval.txt").read_text()
    ids = [x.strip() for x in raw.split() if x.strip()]          # 清洗空行，得到 5011 个 id
    random.Random(seed).shuffle(ids)                               # 用 seed=42 打乱，保证每次切分结果完全一致
    n_val = max(1, int(len(ids) * 0.2))                           # 20% 做验证：int(5011×0.2)=1002
    split_dir.mkdir(parents=True, exist_ok=True)                  # 确保目录存在
    (split_dir / "val.txt").write_text("\n".join(ids[:n_val]))    # 随机后的前 1002 个 → val
    (split_dir / "train.txt").write_text("\n".join(ids[n_val:]))  # 剩下 4009 个 → train
    print(f"[make_split] train={len(ids) - n_val}, val={n_val}")  # 自检输出：应为 train=4009, val=1002


if __name__ == "__main__":
    main()                                                        # 仅直接运行时执行