# 文件：trainer/eval.py
# 核心作用：评估入口：加载 checkpoints/best.pth -> 在 val/test 上推理 -> 解码 + NMS -> 计算 VOC mAP@0.5
# 调用关系：python -m trainer.eval --split val
#           -> data/voc.py（数据）、models/retinanet.py（模型）、models/anchor.py（解码）、utils/metric.py（mAP）
# 对应参考：mmdet tools/test.py；detectron2 tools/train_net.py --eval-only（build_evaluator）
# 状态：占位文件，待实现



# 文件：trainer/eval.py
# 核心作用：加载权重 → 在 val 上逐图推理（decode + NMS）→ 算 mAP@0.5
# 运行：python -m trainer.eval --ckpt checkpoints/last.pth --exp EXP-002
# 调用关系：models/retinanet.py（前向）、models/anchor.py（decode）、utils/metric.py（mAP）
import argparse                                      # 命令行参数
import torch                                         # 张量
from torchvision.ops import nms                      # 官方 NMS（C++ 实现，快）
from config.config import (img_size, num_classes, nms_thr, score_thr,
                            topk_per_img, run_dir)   # 常量
from data.voc import VOCDataset, voc_classes        # 数据集 + 类别名
from models.retinanet import RetinaNet, flatten_outputs   # 模型
from models.anchor import decode                     # 解码
from utils.box import clip_boxes                     # 裁剪
from utils.metric import compute_map                 # mAP
from utils.misc import save_json                     # 指标落盘（12.5）


@torch.no_grad()                                     # 评估不需要梯度，省显存
def postprocess(cls, reg, anchors, score_thr=score_thr, nms_thr=nms_thr, topk=topk_per_img):
    """功能：把单张图的原始输出变成最终检测框
    输入：cls (N,C) logits、reg (N,4) 偏移、anchors (N,4)
    输出：dict(boxes(T,4), scores(T,), labels(T,))
    调用位置：evaluate()
    """
    scores = torch.sigmoid(cls)                      # (N,C) 概率
    boxes = decode(anchors, reg)                     # (N,4) 先全部解码，得到输出框坐标，每个框4个坐标值，4个偏移
    out_b, out_s, out_l = [], [], []                 # 收集每类结果
    for c in range(scores.shape[1]):                 # 逐类
        s = scores[:, c]                             # 该类分数
        keep = s > score_thr                         # 先滤低分
        if int(keep.sum()) == 0:                     # 该类没有候选
            continue
        b, s2 = boxes[keep], s[keep]
        if s2.numel() > 1000:                        # 太多就先取 top1000
            top = torch.topk(s2, 1000).indices
            b, s2 = b[top], s2[top]
        idx = nms(b, s2, nms_thr)                    # NMS 去重
        out_b.append(b[idx][:topk])                  # 最多保留 topk
        out_s.append(s2[idx][:topk])
        out_l.append(torch.full((len(idx[:topk]),), c, dtype=torch.long, device=cls.device))
    if not out_b:                                    # 全图为空
        z = torch.zeros((0, 4), device=cls.device)
        return {"boxes": z, "scores": z.new_zeros((0,)), "labels": z.new_zeros((0,), dtype=torch.long)}
    return {"boxes": torch.cat(out_b), "scores": torch.cat(out_s), "labels": torch.cat(out_l)}


def save_reports(exp, ckpt_path, mAP, aps, split="val"):
    """功能：把评估结果自动落盘成 3 个文件，不用手抄终端
          ① runs/<exp>/eval_log.txt   —— 终端原样日志（原始记录，可 diff）
          ② runs/<exp>/results.md     —— Markdown 表（直接贴进组会/报告）
          ③ runs/<exp>/metrics.json   —— 机器可读（已有，见 evaluate）
    输入：exp 实验编号、ckpt_path 权重路径、mAP 总指标、aps 每类 AP（dict）、split 数据划分
    输出：无（写 3 个文件）
    调用位置：evaluate() 最后一步
    """
    out_dir = run_dir / exp                                    # runs/EXP-002/
    out_dir.mkdir(parents=True, exist_ok=True)                 # 目录不存在就建

    # ---------- ① 原始日志：一行一条，和终端打印完全一致 ----------
    log = [f"[eval] split={split}  ckpt={ckpt_path}",
           f"[eval] mAP@0.5 = {mAP * 100:.2f}%"]
    for c, ap in sorted(aps.items()):                          # 按类别 id 顺序（与终端一致）
        log.append(f"  {voc_classes[c]:12s} AP={ap * 100:5.1f}")
    (out_dir / "eval_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")  # 原始记录落盘

    # ---------- ② Markdown：总指标 + 20 类（按 AP 从高到低，方便看短板） ----------
    ranked = sorted(aps.items(), key=lambda kv: -kv[1])        # [(类id, AP), ...] 降序
    md = [f"# {exp} 评估结果（{split}）", "",
          f"- 权重：`{ckpt_path}`",
          f"- 指标口径：VOC mAP@0.5（与 mmdet `mAP@0.5` 一致）",
          f"- **mAP@0.5 = {mAP * 100:.2f}%**", "",
          "| 类别 | AP(%) | 类别 | AP(%) |", "| --- | --- | --- | --- |"]
    half = (len(ranked) + 1) // 2                              # 20 类排 2 列，10 行
    for i in range(half):                                      # 左列 i，右列 i+half
        left = ranked[i]                                       # (类id, AP)
        right = ranked[i + half] if i + half < len(ranked) else (None, None)
        lname = voc_classes[left[0]]
        lval = f"{left[1] * 100:.1f}"
        rname = voc_classes[right[0]] if right[0] is not None else ""
        rval = f"{right[1] * 100:.1f}" if right[0] is not None else ""
        md.append(f"| {lname} | {lval} | {rname} | {rval} |")
    md += ["", "## 复现这条结果", "", "```powershell",
           f"python -m trainer.eval --ckpt {ckpt_path} --exp {exp}", "```"]
    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")  # 可直接贴的表格


def evaluate(ckpt_path, exp="EXP-002"):
    """功能：遍历验证集 → 推理 → 算 mAP
    输入：ckpt 路径、exp（实验编号，用于写 runs/EXP-xxx/metrics.json）
    输出：(mAP, aps)
    调用位置：`python -m trainer.eval` 的 main()
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaNet(pretrained=False).to(device)            # 不需要下载预训练
    ckpt = torch.load(ckpt_path, map_location=device)          # 读权重
    model.load_state_dict(ckpt["model"])
    model.eval()                                               # 评估模式

    ds = VOCDataset("val")                                   # 验证集
    dets, gts = [], []                                         # 预测 / 真值
    for i in range(len(ds)):                                   # 逐图（一张一张，省显存）
        img, boxes, labels = ds[i]                             # 处理一张图
        with torch.no_grad():                                  # 推理不建计算图，省显存
            cls_list, reg_list, anchors = model(img.unsqueeze(0).to(device))  # cls_list张量 = [(B, A*C, 40, 40), (B, A*C, 20, 20), (B, A*C, 10, 10)]3个大数组
            cls, reg = flatten_outputs(cls_list, reg_list)     # cls = (1, 18900, 20) 3层P的所有框
            det = postprocess(cls[0], reg[0], anchors)          # 解码 + NMS。 取第0个批次就是这一张图
        det["boxes"] = clip_boxes(det["boxes"], img_size, img_size)
        dets.append({k: v.cpu() for k, v in det.items()})       # 存到 CPU
        gts.append({"boxes": boxes, "labels": labels})
        if i % 100 == 0:
            print(f"[eval] {i}/{len(ds)}")

    mAP, aps = compute_map(dets, gts, num_classes)             # 算 mAP（utils/metric.py）
    print(f"[eval] mAP@0.5 = {mAP * 100:.2f}%")
    for c, ap in sorted(aps.items()):                          # 逐类打印
        print(f"  {voc_classes[c]:12s} AP={ap * 100:5.1f}")
    save_json(run_dir / exp / "metrics.json",                  # 实验日志的结果原文
              {"mAP": mAP, "ap": {voc_classes[c]: a for c, a in aps.items()}})
    save_reports(exp, ckpt_path, mAP, aps)                     # 自动写 eval_log.txt + results.md（不用手抄）
    print(f"[eval] 已写出 {run_dir / exp}/metrics.json、eval_log.txt、results.md")  # 告诉用户去哪看
    return mAP, aps


def main():
    """功能：解析参数并调用 evaluate；输入：--ckpt/--exp；输出：控制台 mAP + metrics.json；调用位置：__main__"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.pth")
    parser.add_argument("--exp", type=str, default="EXP-002")   # 与训练时的实验编号一致
    args = parser.parse_args()
    evaluate(args.ckpt, args.exp)


if __name__ == "__main__":
    main()