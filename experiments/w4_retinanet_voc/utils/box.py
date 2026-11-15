# 文件：utils/box.py
# 核心作用：框运算工具：IoU、框裁剪到图像边界、xyxy ↔ xywh 转换、解码后过滤
# 调用关系：losses/loss.py（IoU 分配）与 trainer/eval.py、tools/predict.py（解码 / NMS）
# 对应参考：detectron2 detectron2/modeling/box2box_transform.py + structures/boxes.py
# 约定：全项目统一使用 xyxy（左上角原点、x 向右、y 向下）；文件内不 import 任何项目内模块
# 状态：占位文件，待实现



# 核心作用：框的 IoU、裁剪等基础运算, 提供给训练和推理阶段使用
# 调用关系：losses/loss.py 的 assign_targets()、utils/metric.py、trainer/eval.py
import torch                                        


def box_iou(a, b):
    """功能：两组框两两算 IoU
    输入：a (N,4)、b (M,4)，均为 xyxy
    输出：(N,M) IoU 矩阵
    调用位置：losses/loss.py；utils/metric.py
    """
    # 计算a,b组每个框的面积，shape [N], N个框--N个数据
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)   # a 的面积
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)   # b 的面积

    # a组和b组每个框都算交集，并集
    """
        广播lt：
        a:
        tensor([[ 10,  20, 100, 200],
                [ 30,  40, 120, 220]])
        b:
        tensor([[  5,  15,  80, 180],
                [ 25,  35,  90, 190],
                [ 50,  60, 150, 250]])
        1. 先取两组每个框的前两个数据（左上角坐标）
        2. a: [N,4] --> [N,1,2]增加维度; b: [M,4] --> [1,M,2];
           a:2个数组，每个数组4个值 --> 2个大数组，每个数组里面1个小数组; b: 3个数组，每个数组4个值 --> 1个大数组，每个数组里面3个小数组 
        3. 然后广播lt:  [N,M,2]--N个大数组，每个数组里面M个小数组，每个小数组里面2个值
           也就是每个a框和每个b框的左上角坐标，取两者中更大的值，得到交集的左上角坐标，NxM个
            lt = tensor([[[10, 20],                  [N,M,2]
                        [25, 35], 
                        [50, 60]],

                        [[30, 40],
                        [30, 40],
                        [50, 60]]])
            第 0 行：a0 分别和 b0,b1,b2
            第 1 行：a1 分别和 b0,b1,b2
        4. wh = tensor([[[70, 160],
                    [65, 155],
                    [50, 140]],

                    [[50, 140],
                    [60, 150],
                    [70, 160]]])
        5. area_a =  tensor([16200, 17100])        # a0面积，a1面积
           area_b = tensor([13650, 13300, 14000])   # b0,b1,b2面积
           inter = tensor([[11200, 10075,  7000],
                          [ 7000,  9000, 11200]])

        6. 第 0 行 
                a0 & b0     (16200 + 13650 - 11200 = 18650\)
                a0 & b1     (16200 + 13300 - 10075 = 19425\)
                a0 & b2     (16200 + 14000 - 7000 = 23200\)
           union = tensor([[18650., 19425., 23200.],
                          [23750., 21400., 19900.]])
    """

    # 交集左上角：取两个框左上角x、y中更大的值，shape [N,M,2]
    lt = torch.max(a[:, None, :2], b[None, :, :2])        # NxM个坐标--NxM个小数组
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])        # 交集的右下角
    # 交集宽高：rb - lt；如果rb<lt（无重叠），差值为负，clamp到0，交集面积=0
    # 和上面lt的输出格式一样，每个框的宽和高
    wh = (rb - lt).clamp(min=0)                           
    # 交集面积 = 宽 * 高，shape [N,M]
    inter = wh[..., 0] * wh[..., 1]                       
    # 并集面积 = A面积 + B面积 - 交集面积
    # area_a[:, None] → [N,1]；area_b[None, :] → [1,M]，广播后 [N,M]
    union = area_a[:, None] + area_b[None, :] - inter    
    # IoU = 交集 / 并集；clamp防止并集为0除零爆炸
    return inter / union.clamp(min=1e-6)                


def clip_boxes(boxes, width, height):
    """功能：把框裁到图像范围内
    输入：boxes (N,4) xyxy；width/height 图像宽高
    输出：(N,4)
    调用位置：trainer/eval.py 的 postprocess()
    """
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)    # x1/x2
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)   # y1/y2
    return boxes