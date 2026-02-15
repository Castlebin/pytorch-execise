# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import numpy as np
import matplotlib.pyplot as plt

"""
数据：特征、标签；设置权值：（0,4,0.01）-特征值和不同的w进行变换，
根据损失函数，输出。
"""

# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    """
    特征经过网络进行变换得到预测
    特征乘权重；
    :param x: 输入
    :return: x * w；
    """
    return x * w


def loss(x, y):
    """
    求得预测与真实的误差
    :param x: 特征
    :param y: 真实值
    :return: 损失
    """
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 创建两个空列表；分别用于存放每一次权重值 和 每一次循环的 误差
w_list = []
mes_list = []

for w in np.arange(0.0, 4.1, 0.01):
    """
    进行了0.0-4.1距离；间隔0.01；次循环；进行循环
    损失和，初始化
    取特征集和对应的标签传给x_val,y_val，
    调用forward函数返回y_pred_val预测值；
    调用loss函数返回loss_val损失值
    求损失的和
    输出
    """
    # print('w = ', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE = ', l_sum / 3)
    w_list.append(w)
    mes_list.append(l_sum / 3)

# 画图：每一次的w为x轴；对应的误差为y轴
plt.plot(w_list, mes_list)
plt.ylabel('Loss')
plt.plot('w')
plt.show()

