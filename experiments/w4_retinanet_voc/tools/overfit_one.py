# 文件：tools/overfit_one.py（临时脚本，验收完可删）
# 核心作用：用 1 张图过拟合，验证"数据→模型→损失→反传"这条链没有 bug
# 运行：python -m tools.overfit_one
# 调用关系：独立脚本；内部调用 data/voc.py、models/retinanet.py、losses/loss.py
import torch
from data.voc import VOCDataset
from models.retinanet import RetinaNet, flatten_outputs
from losses.loss import Loss
from config.config import lr

def main():
    """功能：单图过拟合；输入：无；输出：控制台 loss 曲线；调用位置：仅 __main__"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img, boxes, labels = VOCDataset("val")[0]                  # 固定一张图，导入图片和标准数据用于训练
    img = img.unsqueeze(0).to(device)                           # (1,3,320,320)
    boxes, labels = [boxes.to(device)], [labels.to(device)]     # 包成 batch=1 的 list

    model = RetinaNet(pretrained=True).to(device)               # 有预训练收敛更快
    criterion = Loss()                                 # 损失
    opt = torch.optim.AdamW(model.parameters(), lr=lr)          # 优化器

    model.train()
    for step in range(101):                                     # 100 步足够看到下降
        opt.zero_grad(set_to_none=True)
        cls_list, reg_list, anchors = model(img)
        cls, reg = flatten_outputs(cls_list, reg_list)          # 前向算出img分类和回归偏移
        loss, stats = criterion(cls, reg, anchors, boxes, labels)  # 将前向算出和标准的算loss
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step {step:3d} loss={loss.item():.4f} 分类损失cls={stats['cls']:.4f} "
                  f"回归损失reg={stats['reg']:.4f} 正样本anchor：pos={stats['num_pos']}")
    print("[overfit] 若 loss 明显下降（如 4.x → <1.0），说明链路正确")

if __name__ == "__main__":
    main()