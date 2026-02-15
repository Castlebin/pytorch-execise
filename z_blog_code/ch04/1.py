# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True

"""
因为torch.Tensor函数，所以我们的w是一个张量；
下面我们计算时，返回的也是张量；这样就可以有一其他功能，便可直接函数取出我们想要的值；
item（）是从原来的精度上更加精确；
最外的两个print分别是，模型跑之前，和w值改变之后再跑的预测结果。
"""


def forward(x):
    return x * w


def loss(x, y):
    """
    :param x: 输入特征
    :param y: 真实值
    :return: 损失值即误差
    """
    y_pred = forward(x)

    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        """
        调用loss函数，求损失函数
        l.backward()--反向传播函数，自动会计算并且w的值也会做相应的更新，并且最终会释放掉数据
        这个函数作用就是用于做反向循环--这个时候w的值会有两个：w.data--w的值、w.grad--梯度值即对误差求导的值；
        更新w.data
        w.grad清零，即释放，用于下次计算
        """
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item(), w.data)
        w.data = w.data - 0.01 * w.grad.data
        print(w.data)
        w.grad.data.zero_()
        print(w.data, w.grad.item())

    print('progress:', epoch, l.item())

print('predict (after taining)', 4, forward(4).item())