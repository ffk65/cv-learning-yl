# 评估模型，计算测试集上的准确率，并绘制混淆矩阵
import os                                                          
import argparse
import torch, numpy as np,matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from PIL import Image  # ==========【新增：PIL用于读取本地图片】==========
from torchvision import transforms # ==========【新增：图片预处理】==========

# 从 train.py 导入必要的模块和函数
from train import test_loader, device, select_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--mod', type=str, default='cnn', choices=['cnn', 'resnet'], help='Mod type: cnn or resnet')
    # # ==========【新增命令行参数：外部图片路径，可选参数，不填则只跑测试集评估】==========
    # parser.add_argument('--img_path', type=str, default=None, help="本地待预测图片路径，例如：'test_img/my_plane.jpg'")
    args = parser.parse_args()
    mod = args.mod
    img_path = args.img_path # ==========【新增：读取图片路径参数】==========

    model = select_model(mod).to(device)
    model.load_state_dict(torch.load(f'checkpoints/{mod}_best_model.pth', weights_only=True)) # 加载训练好的最佳模型参数
    model.eval()

    # CIFAR-10 类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  
    
    # 真实标签和预测标签
    ys, ps = [], []      
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            pred = model(data).argmax(dim=1).cpu()           # 对每个样本，在所有类别得分中选出最大值对应的类别索引（预测类别）
            ys += target.tolist(); ps += pred.tolist()

    # 打印前10个真实标签和预测标签
    print(ys[:10], ps[:10])  
    # 逐样本比较预测和真实标签是否相等,得到布尔数组并计算准确率
    acc = np.mean(np.array(ys) == np.array(ps))               
    print(f'Accuracy on test set: {acc:.4f}')


    # 计算混淆矩阵
    cm = confusion_matrix(ys, ps)         
    print("Confusion Matrix:\n", cm)
    # 创建混淆矩阵显示对象
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)      
    disp.plot(xticks_rotation=45)
    plt.tight_layout()

    fig_path = f'../figures/{mod}_confusion_matrix2.png'  # 保存混淆矩阵图像的路径
    # 保存混淆矩阵图像
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')                      
    plt.close()


    #     # ==========【新增代码开始：单张外部图片推理模块】==========
    # # 图片预处理：必须和train.py训练集预处理保持完全一致！CIFAR10输入32×32
    # transform = transforms.Compose([
    #     transforms.Resize((32,32)),  # 缩放至32×32，和训练输入尺寸匹配
    #     transforms.ToTensor(),
    #     # CIFAR10官方均值方差，和训练时normalize参数一致
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    # ])

    # # 判断是否传入图片路径，如果传入则执行图片识别
    # if img_path is not None:
    #     # 校验图片文件是否存在
    #     if not os.path.exists(img_path):
    #         print(f"错误：图片文件不存在！路径：{img_path}")
    #     else:
    #         # 读取图片，转为RGB（防止png透明通道报错）
    #         img = Image.open(img_path).convert("RGB")
    #         # 预处理图片
    #         img_tensor = transform(img).unsqueeze(0) # unsqueeze增加batch维度：[C,H,W] -> [1,C,H,W]
    #         img_tensor = img_tensor.to(device)

    #         # 推理，关闭梯度计算
    #         with torch.no_grad():
    #             output = model(img_tensor)
    #             pred_idx = output.argmax(dim=1).item() # 获取预测类别索引
    #             pred_class = class_names[pred_idx]     # 映射为类别名字

    #         print("\n===== 外部图片预测结果 =====")
    #         print(f'第轮训练结束,Test Accuracy: ')
    #         print(f"path：{img_path}")
    #         print(f"idx：{pred_idx}")
    #         print(f"class：{pred_class}")
    # # ==========【新增代码结束】==========