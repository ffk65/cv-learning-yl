# 评估模型，计算测试集上的准确率，并绘制混淆矩阵
import os                                                          
import argparse
import torch, numpy as np,matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 从 train.py 导入必要的模块和函数
from train import test_loader, device, select_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--mod', type=str, default='cnn', choices=['cnn', 'resnet'], help='Mod type: cnn or resnet')
    args = parser.parse_args()
    mod = args.mod

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

    fig_path = f'../figures/{mod}_confusion_matrix.png'  # 保存混淆矩阵图像的路径
    # 保存混淆矩阵图像
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')                      
    plt.close()