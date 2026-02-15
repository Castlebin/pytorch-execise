# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

"""
batch=64；
数据转换函数--变张量
下载数据，取出训练数据，dataset.MNIST函数
数据打断设置batch,DataLoader
下载数据，取出测试数据，dataset.MNIST函数
数据不打断 设置batch
"""
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        """
        搭建连接层
        Conv2d(输入通道，输出通道，卷积核是 几乘几的)
        MaxPool2d（2）   池化层   除以二；x y 维度上
        Linear线性变换
        """
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        """
        #Flatten data from (n, 1, 28, 28) to (n, 784)
        获取x的第一个参数，返回给batch，即获取batch的大小
        先通过 卷积核（conv1） 进行卷积变换 - 进行 池化层（pooling）  - 再 正则化（relu）
        通过（conv2）卷积核 进行卷积变换 - 进行 池化层（pooling） - 再 正则化（relu）
        将做变换好的数据，拉长，变成一个 一维的数据。 - view（batch， -1） 一次处理batch个数据
        进行线性变换 - Linear 函数
        :param x: 输入数据
        :return: 输出数据
        """
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


"""
实例化Net类
判断在GPU还是cpu上跑，这里是在gpu上
将模型放在gpu上（to.(device)）
代价函数
优化器，初始化模型参数（权重，偏置），学习率0.01， 为了防止局部最优，在进行迭代时，再加一个函数，这里设置为0.5，和学习率一个等级
"""
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        """
        train_loader 传出 batch的索引，即第几次数据， data当前数据
        将数据的特征和标签传给 inputs，和 target
        将数据传入gpu上
        优化器中的梯度设置为零
        将数据放入模型中
        criterion计算损失值
        .backward - 反向传播
        优化器函数进行更新权重
        将损失值进行累加 传给 running_loss函数
        输出第300次及300的倍数次的测试数据  --  第几次迭代， 当前的第几组数据，即第几个batch，当前损失和
        每300次，损失值做一个归结
        """
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        """
        torch.no_grad()不用计算梯度
        对测试数据进行遍历
        将数据的 特征和标签 分别传给 images和labels
        数据传到gpu上
        带入模型进行预测 得到预测结果 - outputs
        这里是，我们取出概率最大的那个数作为输出
        取出标签的第一列， 进行累加，即计算总的数据集长度
        判断正确的数量，进行累加
        输出正确率
        """
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()