import torch
import random
import argparse
import numpy as np
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- 设置随机种子以确保可重复性 ---
seed = 42  # 随机种子
torch.manual_seed(seed) # 设置 PyTorch 随机种子
np.random.seed(seed)    # 设置 NumPy 随机种子
random.seed(seed)       # 设置 Python 随机种子
torch.backends.cudnn.deterministic = True  # 确保卷积运算的确定性

# --- 图像预处理 ：张量 + 标准化 ---
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- 下载数据集 ---
train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=test_transform)

# --- 数据加载器 ---
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# --- 设备配置 ---
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# --- 是否使用 ResNet18 模型 ---
USE_RESNET = False  

# --- 定义卷积神经网络模型 ---
class simpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1 ), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 定义 ResNet18 模型 ---
def get_resnet18(num_classes=10):
    model = models.resnet18(weights=None)  # 不加载预训练权重
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改全连接层输出类别数
    return model

# --- 选择模型：接受参数 mod，'resnet' 或 'cnn' ---
def select_model(mod):
    if mod == 'resnet':
        return get_resnet18(num_classes=10)
    elif mod == 'cnn':
        return simpleCNN()
    else:
        raise ValueError("Invalid model type. Choose 'resnet' or 'cnn'.")




best_acc, patience, wait = 0.0, 5, 0    # Early stopping 参数
# --- 优化模型参数 ---
if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10')
    parser.add_argument('--mod', type=str, default='cnn', choices=['cnn', 'resnet'], help='Mod type: cnn or resnet')
    args = parser.parse_args()
    mod = args.mod

    model = select_model(mod).to(device)
    # --- 损失函数 ---
    loss_fn = nn.CrossEntropyLoss()
    # --- 优化器 ---
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # ---LR调度器 ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # --- TensorBoard 记录 ---
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/cifar10_{mod}')  # 日志目录

    for epoch in range(50):
        print(f'-----第[{epoch+1}/50]轮训练开始-----')
        running_loss = 0.0
        #--- 训练模型 ---
        model.train()
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred  = model(data)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch % 100 == 0:
                print(f'Step [{batch}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        scheduler.step()

        # --- 测试模型 ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                predicted = pred.argmax(dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        epoch_acc = correct / total
        print(f'第{epoch+1}轮训练结束,Train Accuracy: {epoch_acc:.4f},Train Loss: {epoch_loss:.4f}')

        # --- TensorBoard 记录 (tensorboard --logdir runs启动)---
        writer.add_scalar('loss/train', epoch_loss, epoch) # 记录每轮平均训练损失
        writer.add_scalar('accuracy/test', epoch_acc, epoch)     # 记录测试准确率
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)  # 记录学习率

        # --- 保存最佳模型 ---
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            wait = 0
            save_path = f'checkpoints/{mod}_best_model.pth'
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1 
            if wait >= patience:
                print('Early stopping.')
                break

    print(f'Best model saved with accuracy: {best_acc:.4f}')
    writer.close()


