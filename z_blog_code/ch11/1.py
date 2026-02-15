# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
"""
构建数据处理函数 - 取出数据 - 进行数据处理（训练，测试）
搭建晓得模块的模型
再搭建整个模型 - 传入小模型，共同构建新的模型
实例化模型 - 选择损失函数，优化器
传入训练集对模型进行训练，传入测试集进行测试
写主函数
"""
import torch

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

"""
设置batch
调用数据转换函数，transform函数
取出训练集，达到训练数据 - train_dataset
将训练集用DataLoader函数进行处理， shuffle数据打乱，batch个数据合成一个数据
测试集同样操作，只是没有打乱数据的操作
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


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        """
        Inception类---继承Module类，
        重新写__init__ forward  方法
        搭建
        卷积层：branch1*1 = Conv2d（输入通达数，输出通道数，卷积核1*1）
        卷积层：branch5*5_1 = Conv2d(输入通道数，输出通道数，卷积核1*1)
        卷积层：branch5*5_2 = Conv2d(输入通道数，输出通道数，卷积核5*5,步长padding=2)
        卷积层：branch3*3_1 = Conv2d(输入通道数，输出通道数，卷积核1*1)
        卷积层：branch3*3_2 = Conv2d(输入通道数，输出通道数，卷积核3*3，步长padding=1)
        卷积层：branch3*3_3 = Conv2d(输入通道数，输出通道数，卷积核3*3，步长padding=1)
        卷积层：branch_pool = Conv2d(输入通道数，输出通道数，卷积核3*3，步长padding=1)
        :param in_channels: 输入通道数，即输出数据的通道数（厚度）
        """
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        """
        1、将数据用branch1*1进行处理--branch1*1
        2、将数据用branch5*5_1进行处理，再将处理过的数据放进branch5*5_2卷积层进行训练--branch5*5_2
        3、将数据用branch3x3_1进行处理，再将处理过的数据放进branch3x3_2卷积层进行训练得到新的数据，再讲这些数据放入branch3x3_3层得到__branch3x3
        4、将数据用avg_pool2d进行卷积操作，再将得到的数据放进branch_pool卷积进行训练得到--branch_pool
        将得到的四组数据进行整合--outputs
        # b,c,w,h  c对应的是dim=1，通道数
        :param x:输入数据
        :return:训练后的通道数
        """
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]

        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self):
        """
        Net继承Module类
        重写__init__ 和 forward 方法
        搭建
        卷积层:Conv2d(输入通道，输出通道，卷积核大小)--conv1
        卷积层:Conv2d(输入通道，输出通道，卷积核大小)--conv2
        调用Inception方法--incep1
        调用Inception方法--incep2
        池化层 MaxPool2d函数--mp
        搭建线性层Linear（输入数据维度，输出维度）--fc
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        """
        取出x的size的第一位，获取batch的大小，这里可以根据模型实例化后，看传入的参数是什么，这里取得是参数的size 的 第一个数据，这里是一个数据有多少数据组成的
        x - 卷积层conv1 - 进行池化mp - 正则化relu（调用的F中的relu函数）- x
        将得到的x - 放入incep1中得到新的 - x
        x - 卷积层conv2 - 进行池化mp - 正则化relu（调用的F中的relu函数）- x
        将得到的x - 放入incep2中得到新的 - x
        将x用.view函数将二维数据转化成一维数据
        记性线性变换
        :param x:输入数据
        :return:返回结果
        """
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)

        return x


"""
实例化类 - model
判断是在什么上运行gpu or cpu
将模型放在GPU上
调用损失函数--criterion
悬着SGD优化器，模型参数初始化（权重，偏置，.parameters）， 学习率0.01，momentum = 0.5，更好的进行求最小值，减少局部最优
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
        取出训练数据 - 第几个数据， 输入数据
        将数据取出特征和标签 - inputs, target
        将数据放入gpu上
        优化器的梯度设置为0
        将数据的特征放入模型，进行训练
        计算损失函数 - loss
        反向传播， - .backward
        优化器函数进行梯度迭代，更新权重和偏置
        损失累加 - running_loss
        每300个数据进行输出一次（第几次迭代，第几个数据，损失是多少）
        并且损失running_loss置零
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
        不用计算梯度
        遍历测试集
        将数据的 特征和标签 传给 - images labels
        将数据放在GPU上
        将特征放入模型进行训练 - outputs
        取出结果的最大值， max
        计算标签的长度，即计算测试了多少数量 通过累加 返回给 - total
        计算预测正确的函数进行累加 - correct
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