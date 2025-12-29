# 使用 Pytorch 中的 MINST 数据集进行图像分类 （手写数字识别）

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# %% 第一步：环境准备与数据加载

# 1. 设置设备 (优先使用 GPU/MPS，否则使用 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 定义数据转换
# ToTensor: 将图片转换为 PyTorch 张量 (0-1 范围)
# Normalize: 标准化 (均值 0.1307, 标准差 0.3081 是 MNIST 数据集的统计值)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 3. 下载并加载数据集
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)

# 4. 创建 DataLoader
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


# %% 第二步：使用 Matplotlib 可视化原始数据
# 在训练之前，查看一下数据长什么样是一个好习惯。
def visualize_data(loader):
    # 获取一个 batch 的数据
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # 创建画布
    figure = plt.figure(figsize=(10, 4))
    cols, rows = 5, 2

    for i in range(1, cols * rows + 1):
        # 随机选择图片索引
        sample_idx = torch.randint(len(images), size=(1,)).item()
        img, label = images[sample_idx], labels[sample_idx]

        figure.add_subplot(rows, cols, i)
        plt.title(f"Label: {label.item()}")
        plt.axis("off")
        # img 是 (1, 28, 28)，squeeze 后变成 (28, 28) 用于绘图
        plt.imshow(img.squeeze(), cmap="gray")

    plt.suptitle("MNIST Training Data Samples")
    plt.show()


# 调用函数显示图片
visualize_data(train_loader)


# %% 第三步：构建卷积神经网络 (CNN)
# 我们将构建一个简单的 CNN，包含两个卷积层和两个全连接层。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 1: 输入 1通道, 输出 32通道, 卷积核 3x3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 卷积层 2: 输入 32通道, 输出 64通道, 卷积核 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout 防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层 1:
        # 图片经过两次卷积(无padding, 尺寸变小)和一次最大池化后，
        # 尺寸变为 12x12 (具体推导: 28->26->24->12)，64通道
        self.fc1 = nn.Linear(9216, 128)
        # 全连接层 2: 输出 10 个类别 (0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 池化层
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # 展平
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) # 输出概率对数
        return output

# 实例化模型并移至设备
model = Net().to(device)
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)


# %% 第四步：定义训练和测试函数
train_losses = [] # 记录每个epoch的平均loss
test_accuracies = [] # 记录每个epoch的测试准确率

def train(model, device, train_loader, optimizer, epoch):
    model.train()  # 切换到训练模式

    epoch_loss = 0  # 累加 loss
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. 梯度清零（清理上一次循环留下的梯度）
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)
        output = model(data)  # 前向传播
        loss = F.nll_loss(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 计算该 epoch 的平均 loss 并记录
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}')


def test(model, device, test_loader):
    model.eval()  # 切换到评估模式

    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # 获取概率最大的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracies.append(accuracy)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')


# %% 第五步：开始训练
# 为了演示，我们只训练 3 个 Epoch（轮次），这通常足以在 MNIST 上达到 98% 以上的准确率。
# epochs = 5 # 建议至少训练 5-10 轮以看到明显曲线
epochs = 10
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 可视化训练过程中的 Loss 和 Accuracy 变化
def plot_training_history(train_losses, test_accuracies):
    plt.figure(figsize=(12, 5))

    # 子图 1: 训练 Loss 变化
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', color='blue', label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 子图 2: 测试 Accuracy 变化
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_accuracies)+1), test_accuracies, marker='s', color='orange', label='Test Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# 调用绘图函数
plot_training_history(train_losses, test_accuracies)


# %% 第六步：预测结果可视化 (Matplotlib)
# 最后，我们从测试集中取出一部分数据，让模型进行预测，并将预测结果与真实标签画在一起。
def visualize_predictions(model, device, test_loader):
    model.eval()

    # 获取一批测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # 进行预测
    output = model(images)
    preds = output.argmax(dim=1, keepdim=True)

    # 移回 CPU 以便绘图
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    # 绘图
    figure = plt.figure(figsize=(12, 6))
    cols, rows = 6, 3

    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)

        img = images[i - 1].squeeze()
        pred_label = preds[i - 1].item()
        true_label = labels[i - 1].item()

        plt.axis("off")
        plt.imshow(img, cmap="gray")

        # 如果预测正确用绿色，错误用红色
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=10)

    plt.tight_layout()
    plt.suptitle("Model Predictions vs Ground Truth", y=1.05)
    plt.show()


# 运行可视化
visualize_predictions(model, device, test_loader)

'''
总结与关键点
数据预处理 (Transforms): 这里的 Normalize 很重要，它让数据分布在 0 附近，能加速模型收敛。

模型结构 (Model): 使用了 Conv2d (卷积) 和 MaxPool2d (池化) 来提取图像特征，这是处理图像的标准方法。

模式切换: 训练时用 model.train()（启用 Dropout/BatchNorm），预测时务必用 model.eval()。

可视化: 结合 matplotlib 可以直观地看到模型在哪里犯了错（如果有红色的标题）。
'''

