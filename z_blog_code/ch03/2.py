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


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


"""
#第一次更新w前，使用w=1.0对特征为4时进行预测，返回预测值
#第二次，更新w后，使用更新后的w，对特征为4时预测，返回预测值
"""
print('Predict (before traning)', 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        """
        求梯度
        更新w
        输出特征和对应的真实标签；
        计算损失
        输出w和loss
        """
        grad = gradient(x, y)
        w -= 0.01 * grad
        print('\tgrad:', x, y, grad)
        l = loss(x, y)
        print('progress:', epoch, 'w = ', w, 'loss = ', l)

print('Predict (after training)', 4, forward(4))