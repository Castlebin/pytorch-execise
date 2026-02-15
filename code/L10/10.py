# 10. 卷积神经网络 (CNN)（基础篇）

# %% 导包
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dl_d2l.util import colab_util
from dl_d2l.util import device_util
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据集存放目录
base_data_dir = colab_util.get_base_data_dir()
datasets_dir = os.path.join(base_data_dir, 'ML', 'Datasets')
os.makedirs(datasets_dir, exist_ok=True)
print(f'datasets dir: {datasets_dir}')

# 获取可用设备 (优先使用 GPU/MPS，否则使用 CPU)
device = device_util.get_available_device()
print(f"Using device: {device}")

# %% 1. 加载数据集
# batch 大小
batch_size = 64
# ToTensor: 将图片转换为 PyTorch 张量 (0-1 范围) [归一化]
# Normalize: 标准化 (均值 0.1307, 标准差 0.3081 是 MNIST 数据集的统计值)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root=datasets_dir, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root=datasets_dir, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 看一下数据集大小和形状
"""
训练数据集大小: 60000，样本形状：torch.Size([1, 28, 28])
测试数据集大小: 10000，样本形状：torch.Size([1, 28, 28])
"""
print(f"训练数据集大小: {len(train_dataset)}，样本形状：{train_dataset[0][0].shape}")
print(f"测试数据集大小: {len(test_dataset)}，样本形状：{test_dataset[0][0].shape}")


# %% 2. 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 1: 输入 1 通道, 输出 10 通道, 卷积核 5*5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 卷积层 2: 输入 10 通道, 输出 20 通道, 卷积核 5*5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # MaxPooling
        self.pooling = nn.MaxPool2d(2)
        # 全连接层
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        h = F.relu(self.pooling(self.conv1(x)))
        h = F.relu(self.pooling(self.conv2(h)))
        h = torch.flatten(h, 1)  # 展平
        output = self.fc(h)  # 全连接
        return output


# 模型实例
model = Net()
# 将模型放到可用设备上 (尽量使用 GPU/MPS)
model = model.to(device)

# %% 3. 定义损失函数和优化器
## 使用交叉熵损失函数 (CrossEntropyLoss <==> LogSoftmax + NLLLoss)
loss_fn = torch.nn.CrossEntropyLoss()
## 随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# %% 4. 定义训练函数和测试函数
## 训练和测试的代码和之前都一样，只不过这里会尝试去使用 GPU 来加速训练过程（使用可用设备 GPU、MPS、DirectML）
def train(epoch):
    # 训练模式
    model.train()

    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data

        # 将数据放到可用设备上
        inputs, target = inputs.to(device), target.to(device)

        # 获得模型预测结果(64, 10)
        outputs = model(inputs)

        # 交叉熵代价函数 outputs(64,10), target（64）
        loss = loss_fn(outputs, target)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    # 评估模式
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            # 将数据放到可用设备上
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            # 取最大的值作为预测分类
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第 0 个维度，行是第 1 个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %.4f %% ' % (100 * correct / total))


# %% 5. 训练
if __name__ == '__main__':
    for epoch in range(10):
        start_time = time.time()
        train(epoch)  # 训练
        print(f'epoch {epoch + 1}, train cost time: {time.time() - start_time} s , device: {device}')

        test()  # 测试
