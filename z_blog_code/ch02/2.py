# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    """
    返回的是预测值
    :param x: 输入
    :return: 经过网络的输出
    """
    return x * w + b


def loss(x, y):
    """
    返回的是预测值与真实值得平方差
    :param x: 特征
    :param y: 预测值
    :return: 误差
    """
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 创建列表用于存放w值；损失值；偏执值
# w_list = []
mes_list = []
# b_list = []
W = np.arange(0.0, 4.1, 0.01)
B = np.arange(0.0, 4.1, 0.01)
[w, b] = np.meshgrid(W, B)

"""
#zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
"""
l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val
    # print(x_val, y_val, loss_val)
    # b_list.append(b)
# print('MSE:', l_sum / 3)
# w_list.append(w)
# mes_list.append(l_sum / 3)


# 画图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(w, b, l_sum / 3)

plt.show()