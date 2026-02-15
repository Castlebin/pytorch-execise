# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    """
    继承nn.Module中的逻辑回归函数类
    linear(1, 1)--特征和标签都是一维数据
    """

    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        先通过linear函数进行求得预测值
        然后通过调用F（torch.nn.functional）包中的sigmoid函数对预测结果达到分类效果
        :param x:特征
        :return: 分类标准（0， 1）用到的是sigmoid函数
        """
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


"""
模型实例化
调用函数使用BCELoss求损失，并传给criterion
对模型参数w和b进行初始化；学习率设置为0.01
"""
model = LogisticRegressionModel()
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    """
    遍历1000次
    将数据存入模型返回预测值y_pred
    使用函数criterion，并传入预测值和对应的真实值；求得损失函数
    输出 当前迭代次数 和 损失值的高精度数据

    优化器的梯度赋值为0  即损失函数的导数
    通过backward函数进行反向传播
    使用optimizer.step()函数进行更新权值和偏置
    """
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

"""
取0-10中200个点 赋值给x
将数组x变成张量  -- x_t
将张量x_t放入模型，得到预测张量y_t
利用y_t.data函数，取出预测值，变成数组形式 返回给y
画图传入参数x，y  （两个数据都是数组类型，并且长度是一样的）
画一条平行于x轴的红线。即[0, 10], [0.5, 0.5], c='r'
输出图像plot.show()
"""
x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()