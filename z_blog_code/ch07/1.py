# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import numpy as np
import matplotlib.pyplot as py
import torch

"""
读取文件，以逗号未分割点
取所有行，和第一列到倒数第二列
取所有行，和最后一列
"""
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    """
    super继承Module库
    线性变换；
    8维指的是8个特征-6维是6个特征；8个权重-6个权重
    这是线性网络
    输入数据为8维-输出6维
    输入数据为6维-输出4维
    输入数据为4维-输出1维
    调用sigmoid函数，对数据判断
    """

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        :param x:输入数据
        :return:预测值
        """
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


"""
模型实例化
调用BCELoss函数--给criterion函数
使用SGD优化器，模型参数初始化（parameters()），学习率为0.01
"""
model = Model()
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    """
    遍历100次
    将特征值传入模型，返回预测值（张量）
    传入预测和真实值，通过criterion函数，返回损失函数
    输出  当前次数  和  损失函数的高精度
    优化器中梯度值 参数归零
    反向传播
    通过优化器函数，optimizer.step函数进行更新权值w和偏置b，以及梯度值
    """
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()