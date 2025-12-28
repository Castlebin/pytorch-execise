"""
假设学生花 x 个小时学习，最后考试时能获得 y 分。

假设 y = wx  。即 y 和 x 是线性关系
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_cn import matplotlib_chinese

matplotlib_chinese.enable_matplotlib_chinese()


# 训练数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# w 的初始值
# requires_grad=True 表示需要计算 w 的梯度
w = torch.tensor([0.0], requires_grad=True)


# 预测函数
def predict(x, w):
    return w * x

# loss 函数
def loss_fn(y_pred, y):
    return (y_pred - y) ** 2

# 存储 w 和 mse 的值
w_list, mse_list = [], []


# 训练过程，执行 50 个 epoch
epochs = 50
for epoch in range(epochs):
    loss_sum = 0
    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]

        # 预测值
        y_pred = predict(x, w)

        # 计算平方误差
        loss = loss_fn(y_pred, y)

        # 反向传播，计算梯度
        loss.backward()

        # 更新 w 的值，学习率 lr=0.01
        with torch.no_grad():
            w -= 0.01 * w.grad

        # 清零梯度。否则下次反向传播时梯度会累加
        w.grad.zero_()

        # 平方误差的和
        loss_sum += loss.item()

    # 计算均方误差
    mse = loss_sum / len(x_data)

    w_list.append(w.item())
    mse_list.append(mse)

    print("progress:", epoch, "w=", w.item(), "mse=", mse)

# 绘制 w 和 Loss 的关系
plt.plot(w_list, mse_list)
plt.xlabel("w")
plt.ylabel("loss")
plt.title("w 和 loss 的关系")
plt.show()

# 找到最小 mse 对应的 w
min_mse = min(mse_list)
min_index = mse_list.index(min_mse)
best_w = w_list[min_index]
print("最小 mse=", min_mse, "对应的 w=", best_w)   # 最小 mse= 0.0 对应的 w= 2.0

# 绘制 epoch 和 Loss 的关系
plt.plot(range(len(mse_list)), mse_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("epoch 和 loss 的关系")
plt.show()
