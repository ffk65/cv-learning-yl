# 文件：tools/predict.py
# 核心作用：单张图片推理 → 解码 + NMS → 按分数排序取前 N 个框 → 放大画布画框（标签清晰）→ 保存结果图
# 运行：python -m tools.predict --img data\VOCdevkit\VOC2007\JPEGImages\000001.jpg --thr 0.3
# 调用关系：models/retinanet.py（前向）、models/anchor.py（decode）、trainer/eval.py（postprocess 复用）
# 说明：eval 用 score_thr=0.05 是为了不漏检（mAP 需要召回）；看图要给 0.3 左右，否则满屏 0.05 的噪声框
import argparse                                      # 命令行参数
import numpy as np                                   # 像素处理
import torch                                         # 张量
from PIL import Image, ImageDraw, ImageFont          # 读写图 + 画框 + 字体
from config.config import img_size                   # 输入边长（画布放大倍数基于它）
from data.voc import voc_classes, mean, std          # 类别名 + 归一化参数
from models.retinanet import RetinaNet, flatten_outputs   # 模型
from trainer.eval import postprocess                 # 复用评估里的后处理（解码 + NMS）

# 20 类各一种颜色（VOC 类名顺序见 data/voc.py 的 voc_classes），避免全是红框分不清
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
           "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
           "#808000", "#ffd8b1", "#000075", "#808080", "#000000"]


def get_font(size):
    """功能：找一个可用的 TrueType 字体（Windows 自带），找不到就用默认位图字体
    输入：size（像素字号，整数）
    输出：ImageFont 对象
    调用位置：main()
    说明：标签看不清的根因是默认位图字体只有 ~11px，在 320 小图上糊成一团
    """
    for p in ("C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
              "C:/Windows/Fonts/simhei.ttf",      # 黑体
              "C:/Windows/Fonts/arial.ttf"):      # 英文后备
        try:
            return ImageFont.truetype(p, size)    # 找到就直接用
        except OSError:
            continue                              # 找不到换下一个
    return ImageFont.load_default()               # 全都没有才退回默认字体


def load_image(path):
    """功能：读图 → 归一化张量 + 缩放后的 PIL 图
    输入：path（str）
    输出：(tensor(1,3,320,320), pil_img(320,320))
    调用位置：main()
    """
    pil = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)  # 与训练一致
    arr = np.asarray(pil, dtype=np.float32) / 255.0              # 0~1
    arr = (arr - mean) / std                                     # 与训练一致的标准化
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
    return tensor, pil


def main():
    """功能：推理一张图并保存画框结果
    输入：--img 图片路径、--ckpt 权重、--out 输出路径、--thr 分数阈值、--topk 最多画几个框、--scale 画布放大倍数
    输出：带框图片（默认 runs/pred.jpg）
    调用位置：`python -m tools.predict`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)          # 输入图
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.pth")
    parser.add_argument("--out", type=str, default="runs/pred.jpg")
    parser.add_argument("--thr", type=float, default=0.3)          # 看图阈值：0.3 起步，框太乱就加到 0.4/0.5
    parser.add_argument("--topk", type=int, default=10)            # 图上最多画 10 个框（按分数排序）
    parser.add_argument("--scale", type=int, default=2)            # 画布放大 2 倍，标签才看得清
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有卡用卡
    model = RetinaNet(pretrained=False).to(device)                 # 不需要下载预训练权重
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])  # 读权重
    model.eval()                                                   # 推理模式（BN 用 running stats）

    tensor, pil = load_image(args.img)                             # 预处理
    with torch.no_grad():                                          # 不建计算图
        cls_list, reg_list, anchors = model(tensor.to(device))     # 前向
        cls, reg = flatten_outputs(cls_list, reg_list)             # 展平成 (1, 18900, 20) / (1, 18900, 4)
        det = postprocess(cls[0], reg[0], anchors, score_thr=args.thr)  # 解码 + 逐类 NMS

    # ---- 按分数从高到低只取前 topk 个，避免一张图糊 17 个框 ----
    scores = det["scores"]                                         # (T,)
    order = torch.argsort(scores, descending=True)[:args.topk]      # 前 topk 的下标
    boxes, scores, labels = det["boxes"][order], scores[order], det["labels"][order]

    canvas_size = img_size * args.scale                            # 放大后的画布边长（320→640）
    canvas = pil.resize((canvas_size, canvas_size), Image.BICUBIC)  # 放大图片（双三次插值更平滑）
    draw = ImageDraw.Draw(canvas)                                  # 在放大画布上画
    font = get_font(max(20, 12 * args.scale))                      # 字号跟着放大倍数走（640 画布用 24px，缩回 320 仍有 12px）
    line_w = max(2, args.scale)                                    # 线宽跟着放大倍数走

    for box, score, label in zip(boxes.cpu(), scores.cpu(), labels.cpu()):  # 逐个框
        x1, y1, x2, y2 = [float(v) * args.scale for v in box]      # 坐标同步放大
        x1, y1 = max(0.0, x1), max(0.0, y1)                        # 裁到画布内（解码可能越界）
        x2, y2 = min(float(canvas_size), x2), min(float(canvas_size), y2)
        color = PALETTE[int(label) % len(PALETTE)]                 # 每类固定颜色
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)   # 画框

        text = f"{voc_classes[int(label)]} {float(score):.2f}"      # 标签文字
        l, t, r, b = draw.textbbox((0, 0), text, font=font)         # 量文字宽高
        ty = y1 + 1 if y1 + 1 + (b - t) < canvas_size else y1 - (b - t) - 1  # 标签写在框内左上角，贴底就写框上方
        draw.rectangle([x1, ty, x1 + (r - l) + 6, ty + (b - t) + 2], fill="black")  # 黑底，保证任何图上都看得清
        draw.text((x1 + 3, ty), text, font=font, fill="white")      # 白字压在黑底上

    canvas.save(args.out)                                          # 保存
    print(f"[predict] thr={args.thr} topk={args.topk} -> 画了 {len(boxes)} 个框，已保存 {args.out}")
    for box, score, label in zip(boxes.cpu(), scores.cpu(), labels.cpu()):  # 终端也列一遍，方便写报告
        print(f"  {voc_classes[int(label)]:12s} {float(score):.3f}  "
              f"[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")


if __name__ == "__main__":
    main()
