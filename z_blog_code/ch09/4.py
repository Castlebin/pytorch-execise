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
一次性组合64个数据为一个数据
对数据进行下载；
shuffle（数据是否打乱）
对数据进行打乱---在进行batch分割---train_dataset--train_loader
对数据直接进行分割 --- test_dataset--test_loader
"""
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class Net(torch.nn.Module):
    """
    继承Module类，重写__init__和forward方法‘
    继承Net类
    书写神经网络层:(线性层)
    forward函数书写每一层的连接状态，进行的什么变化
    """

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


"""
类的实例化--model
判断使用gpu还是cpu
模型传到gpu上（.to(device)）
调用损失函数--criterion
选用优化器SGD 模型的数值初始化（权值，偏置，梯度函数等等），学习率设置为0.01， 
一般，神经网络在更新权值时，采用如下公式:
                                                   w = w - learning_rate * dw
    引入momentum后，采用如下公式：
                                                   v = mu * v - learning_rate * dw
                                                   w = w + v
    其中，v初始化为0，mu是设定的一个超变量，最常见的设定值是0.9。可以这样理解上式：如果上次的momentum()与这次的
"""
model = Net()
# model = model.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    """
    损失为0.0
    :param epoch: 迭代次数
    :return: 训练模型
    """
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        """
        取一个batch数据的索引值和相应的数据
        data = 特征数据 + 标签
        将数据上传到gpu上
        优化器将梯度设置为0
        将特征放入模型
        利用criterion函数计算损失函数
        利用.backward函数进行反向传播
        利用优化器进行更新权值和偏置值，更新的效果
        将损失函数的高精度进行叠加，传给running_loss变量
        判断对所有数据进行的第几组batch，如果是300组则输出....
        并且running损失为0
        """

        inputs, target = data
        # inputs.cuda()
        # target.cuda()
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    """
    :return:预测结果
    定义两个变量
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            """
            取出测试数据
            将特征传给-images；标签传给-labels
            将数据放在gpu上
            将特征数据放入模型
            torch.max的返回值有两个，第一个是每一行的最大值是多少，第二个是每一行最大值的下标(索引)是多少。
            # dim = 1 列是第0个维度，行是第1个维度
               ---返回标签---
            如果预测值和标签一样，则加一
            total = 数据量
            # 张量之间的比较运算
            正确数据量除以总数据量  乘以  100 得到百分比
            """
            images, labels = data
            # images.cuda()
            # labels.cuda()
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