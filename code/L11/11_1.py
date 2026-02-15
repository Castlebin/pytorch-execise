# 11. 卷积神经网络(CNN)（高级篇）
## part 1. GoogLeNet 介绍
"""
说明：Inception Module (主要思想是解决 1。使用多个卷积分支，将计算结果 concat 起来)
1、卷积核超参数选择困难，自动找到卷积的最佳组合。
2、1x1 卷积核，不同通道的信息融合。使用1x1卷积核虽然参数量增加了，但是能够显著的降低计算量(operations)
3、Inception Module 由 4 个分支组成，要分清哪些是在 Init 里定义，哪些是在 forward 里调用。4 个分支在 dim=1(channels) 上进行 concatenate。24+16+24+24 = 88
"""

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
## 首先，定义 Inception Module (使用多个卷积分支)
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 4 个分支
        ## 分支 1：Average Pooling + 1*1 Conv(24)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
        ## 分支 2：1*1 Conv(16)
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        ## 分支 3：1*1 Conv(16) + 5*5 Conv(24)
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        ## 分支 4：1*1 Conv(16) + 3*3 Conv(24) + 3*3 Conv(24)
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        # 分支 1
        branch_pool_o = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool_o = self.branch_pool(branch_pool_o)
        # 分支 2
        branch1x1_o = self.branch1x1(x)
        # 分支 3
        branch5x5_o = self.branch5x5_1(x)
        branch5x5_o = self.branch5x5_2(branch5x5_o)
        # 分支 4
        branch3x3_o = self.branch3x3_1(x)
        branch3x3_o = self.branch3x3_2(branch3x3_o)
        branch3x3_o = self.branch3x3_3(branch3x3_o)

        # concatenate at dim 1 (将 4 个分支的结果 concat 起来)
        outputs = [branch_pool_o, branch1x1_o, branch5x5_o, branch3x3_o]
        return torch.concat(outputs, 1)


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # 88 = 24x3 + 16

        self.incep_1 = InceptionA(in_channels=10)  # 与 conv1 中的10对应
        self.incep_2 = InceptionA(in_channels=20)  # 与 conv2 中的20对应

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        h = F.relu(self.mp(self.conv1(x)))
        h = self.incep_1(h)
        h = F.relu(self.mp(self.conv2(h)))
        h = self.incep_2(h)
        h = h.view(in_size, -1)
        output = self.fc(h)

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
