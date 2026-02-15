# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):
    """
    :param xs: 特征
    :param ys: 对应的真实值
    :return: 误差；返回的均方误差
    """
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2

    return cost / len(xs)


def gradient(xs, ys):
    """
    求导（误差求导）
    :param xs: 特征
    :param ys: 对应的真实值
    :return: 返回的是均方误差的导数
    """
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)

    return grad / len(xs)


print('Predict (before traning)', 4, forward(4))

for epoch in range(100):
    """
    遍历100次
    调用cost函数计算损失值即误差值
    调用gradient函数求得求导值
    更新w值
    输出第几次迭代，对应的更新的权值，损失值
    """
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w = ', w, 'loss = ', cost_val)

print('Predict (after training)', 4, forward(4))