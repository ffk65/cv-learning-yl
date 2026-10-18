import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- 图像预处理 ：张量 + 标准化 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- 下载数据集 ---
train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

# --- 数据加载器 ---
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# --- 定义模型 ---
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class simpleCNN(nn.Module):
    def __int__(self):
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

model = simpleCNN().to(device)     
# --- 损失函数 ---
loss_fn = nn.CrossEntropyLoss()
# --- 优化器 ---
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
# ---LR调度器 ---
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_acc, patience, wait = 0.0, 5, 0    # Early stopping 参数
# --- 优化模型参数 ---
if __name__ == '__main__':
    for epoch in range(20):
        print(f'Epoch [{epoch+1}/20]')
        #--- 训练模型 ---
        model.train()
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred  = model(data)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if batch % 100 == 0:
            print(f'Step [{batch}/{len(train_loader)}], Loss: {loss.item():.4f}')

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
        acc = correct / total
        print(f'第{epoch+1}轮训练结束, Accuracy: {acc:.4f}')

        # --- 保存最佳模型 ---
        if acc > best_acc:
            best_acc = acc
            wait = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping.')
                break
    print(f'Best model saved with accuracy: {best_acc:.4f}')
