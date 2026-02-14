# 9. 多分类问题 (示例：识别 手写数字输入 (MNIST 数据集))

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from dl_d2l.util import colab_util
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

base_data_dir = colab_util.get_base_data_dir()
datasets_dir = os.path.join(base_data_dir, 'ML', 'Datasets')
os.makedirs(datasets_dir, exist_ok=True)
print(f'datasets dir: {datasets_dir}')

# prepare dataset
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root=datasets_dir, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root=datasets_dir, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # -1 其实就是自动获取 mini_batch  (-1 表示自动计算该维度大小)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不做激活，不进行非线性变换


model = Net()

# construct loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update
def train(epoch):
    # 训练模式
    model.train()

    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data

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
            outputs = model(images)
            # 取最大的值作为预测分类
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第 0 个维度，行是第 1 个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
