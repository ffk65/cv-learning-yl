# 文件：data/voc.py
# 核心作用：数据集处理--图片，GT框坐标，类别
#          把 VOC2007 的每个图片(jpg + xml) 读成 (图像张量, 框坐标, 类别编号)并拼接所有图片输出image, boxes, labels，供 DataLoader 使用
# 运行自检：python -m data.voc
# 调用关系：trainer/train.py -> build_loader() -> VOCDataset(...) + collate_fn
#           trainer/eval.py  -> build_loader(split="val")
# 对应参考：mmdet mmdet/datasets/voc.py + xml_style.py；detectron2 detectron2/data/datasets/pascal_voc.py

import xml.etree.ElementTree as ET                                  # 解析 Annotations/*.xml
from PIL import Image                                               # 读 jpg + resize
import numpy as np                                                  # 像素归一化
import torch                                                        # 张量
from torch.utils.data import Dataset                                # 基类
from config.config import data_root, split_dir, img_size           # 路径与尺寸常量

# VOC2007 的 20 个类别（顺序必须固定，否则训练/评估标签错位）
voc_classes = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
class_to_idx = {name: i for i, name in enumerate(voc_classes)}      # aeroplane:0, bicycle:1, ..., tvmonitor:19

# ImageNet 归一化参数（ResNet 预训练权重就是按这个分布训的，必须保持一致）
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)            # 三通道均值
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)             # 三通道标准差


class VOCDataset(Dataset):
    """功能：VOC2007 检测数据集
    输入：split ∈ {"train","val"}；img_size 默认取 config.img_size=320
    输出：每项 = (image[3,H,W] float32, boxes[N,4] float32 xyxy, labels[N] int64)
    调用位置：trainer/train.py 的 build_loader()；trainer/eval.py 的 evaluate()
    """

    def __init__(self, split="train", img_size=img_size):
        # 读取train.txt id 列表 
        ids = (split_dir / f"{split}.txt").read_text().split()
        self.ids = [x.strip() for x in ids if x.strip()]            # 清洗
        self.img_dir = data_root / "JPEGImages"                     # 图片目录
        self.ann_dir = data_root / "Annotations"                    # 标注目录
        self.img_size = img_size                                    # 统一缩放到 320×320

    def __len__(self):
        """功能：返回样本数；输入：无；输出：int；调用位置：DataLoader 采样时自动调用"""
        return len(self.ids)    # 样本数 = 训练/验证集图片数，train=4009，val=1002

    def __getitem__(self, idx):
        """功能：读取第 idx+1 个样本并转成训练格式--单个样本
        输入：idx（int）
        输出：(image, boxes, labels) —— 图片，框坐标，标签0-19
        调用位置：DataLoader 迭代时自动调用
        """
        # 1. 读取第 idx 个图片并 resize + 归一化 + 标准化(像素值)，输出 image (3,H,W), (像素值 0-255  > 0-1  > 标准化)
        image_id = self.ids[idx]                                    # 例如 "000012"
        img = Image.open(self.img_dir / f"{image_id}.jpg").convert("RGB")  # 获取000012图片，统一 3 通道
        ow, oh = img.size                                           # 原图宽高（用于缩放框）
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)   # 直接拉伸到 320，像素 0-255
        arr = np.asarray(img, dtype=np.float32) / 255.0             # 像素值归一化到 [0,1], float32
        arr = (arr - mean) / std                                    # 按 ImageNet 标准化,
        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous() # （H, W, 3）→ （3, H, W）
        # 2. 读取第 idx 张图片的标注 XML，转换到固定尺寸320×320上标注，收集第 idx 张图片里所有框坐标、所有类别编号（0-19）
        boxes, labels = [], []                                      
        root = ET.parse(self.ann_dir / f"{image_id}.xml").getroot() # 解析 XML，拿到根节点
        for obj in root.iter("object"):                             # 遍历 XML 里的每个 object 节点
            name = obj.find("name").text.strip()                    # 获取该 object 的类别名
            if name not in class_to_idx:                            # 跳过 VOC 未使用的类
                continue
            b = obj.find("bndbox")                                  # 获取该 object 的边框节点 <bndbox>
            x1 = float(b.find("xmin").text)                         # 原图左上 x
            y1 = float(b.find("ymin").text)                         # 原图左上 y
            x2 = float(b.find("xmax").text)                         # 原图右下 x
            y2 = float(b.find("ymax").text)                         # 原图右下 y
            sx, sy = self.img_size / ow, self.img_size / oh         # 缩放比例
            x1, x2 = x1 * sx, x2 * sx                               # 原图的框坐标缩放到 320×320 图上
            y1, y2 = y1 * sy, y2 * sy
            if x2 - x1 < 2 or y2 - y1 < 2:                          # 缩得太小 → 丢弃（否则回归不稳）
                continue
            boxes.append([x1, y1, x2, y2])                          # xyxy
            labels.append(class_to_idx[name])                       # 0..19

        # ---------------- 循环结束，一张图所有物体标注处理完 ----------------
        if len(boxes) == 0:                                         # 极端情况：无目标
            boxes = torch.zeros((0, 4), dtype=torch.float32)        # 生成空框张量：shape [0,4]，代表0个框、每个框4个坐标
            labels = torch.zeros((0,), dtype=torch.int64)            
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)        # 转张量
            labels = torch.tensor(labels, dtype=torch.int64)
        return image, boxes, labels                                 # 交给 collate_fn
        # 举例一个图片两个物体：
        # boxes = [[6.4, 17.066, 128.0, 128.0], [10.0, 21.0, 130.0, 130.0]]
        # labels = [1, 5]
        # 转成张量后：boxes 数值完全没变，坐标全部保留，额外带上了 .shape、.dtype 这些张量元信息
        # boxes.shape = torch.Size([2, 4]), boxes.dtype = torch.float32
        # images：tensor(3,320,320), boxes：tensor(N,4)

def collate_fn(batch):
    """功能：把"每张图框数不同"的样本拼成 list（不能直接 stack, torch.stack要求所有张量shape完全一致: 图1 boxes.shape=[2,4]，图2 boxes.shape=[3,4]）
    输入：batch = [(image, boxes, labels), ...]
    输出：(images[list of 3,H,W], boxes[list of N,4], labels[list of N])
    调用位置：DataLoader(collate_fn=collate_fn)；trainer/train.py 的 build_loader()
    """

    """
        样本0：2个物体
        img0 = torch.randn(3,320,320)
        boxes0 = torch.tensor([[6.4,17.066,128,128], [10,21,130,130]]) # shape [2,4]
        labels0 = torch.tensor([1,5]) # shape [2]

        # 样本1：3个物体
        img1 = torch.randn(3,320,320)
        boxes1 = torch.tensor([[10,10,50,50],[60,60,90,90],[100,100,150,150]]) # shape [3,4]
        labels1 = torch.tensor([3,7,9]) # shape [3]

        # 输入：
        batch = [(img0, boxes0, labels0), (img1, boxes1, labels1)]
        images, boxes, labels = collate_fn(batch)

        # 输出：
        images = [tensor(3,320,320), tensor(3,320,320)]
        boxes  = [tensor([2,4]),     tensor([3,4])]
    """

    # 遍历batch中每一个样本b，b[0]就是样本里的image张量，全部放进images列表
    images = [b[0] for b in batch]                                  # 图像可以 stack，但放到训练循环里再 stack
    boxes = [b[1] for b in batch]                                   # 框数量不等 → 保持 list
    labels = [b[2] for b in batch]                                  # 标签同理
    return images, boxes, labels


if __name__ == "__main__":
    # 1. 自检：打印第 1 张图的形状、框数量、标签，验证数据管线
    ds = VOCDataset("val",250)                                        # 创建类实例，也就是调用 __init__，传入参数
    img, boxes, labels = ds[0]                                      # 取第一个样本，规则：x[y] → 去找 x.__getitem__(y)
    print("[voc] image:", tuple(img.shape), img.dtype)               # 期望 (3,320,320) float32
    print("[voc] boxes:", tuple(boxes.shape), boxes[:2])             # 期望 (N,4)
    print("[voc] labels:", tuple(labels.shape), labels[:2].tolist()) # 期望 (N,)
    # print("[voc] class_to_idx:", class_to_idx)               
    # 2. 自检：显示第 3548 张图，验证 train/val 切分是否正确
    ids = (split_dir / f"train.txt").read_text().split()
    ids = [x.strip() for x in ids if x.strip()]            # 清洗
    img_dir = data_root / "JPEGImages"                     # 图片目录
    image_id = ids[3547]                                    # 例如 "000012"
    pil_img = Image.open(img_dir / f"{image_id}.jpg").convert("RGB")
    pil_img.show() 