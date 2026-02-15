# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    """
    模型继承：linear模型
    linear（1， 1）是特征和输出的维度为一维
    forward：构建模型中的各个层的关系---也是输出怎么到输出的，用到哪些层；---可以调用__init__里的函数
    __init__：通多调用nn.Module中函数进行搭建每一层所用到的函数，函数的参数
    forward：将__init__中的搭建好的函数连通起来，告诉我们怎么连通，叙述整个流程
    """

    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # w, b

    def forward(self, x):
        y_pred = self.linear(x)

        return y_pred


"""
模型的实例化
调用损失函数--criterion
选择优化器，学习率设置为0.01；同时其他的值通过model.parameters()函数进行初始化。
里面自己有各种参数，权重 偏置
"""
model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    """
    迭代1000次；
    模型传入特征值--返回预测值y_pred
    往损失函数-criterion-中传入预测值和真实值--返回损失值loss
    输出 当前迭代次数 + 当前损失值
    通过zero_grad函数将梯度值（即损失函数的导数）设置为0
    .backward进行反向传播
    optimizer.step()进行权值w和偏置b的迭代
    """
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

"""
输出最后的权值w
输出最后的偏置b
设置测试集数据
将测试集放入模型中进行预测
通过.data 输出预测数据
"""
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)