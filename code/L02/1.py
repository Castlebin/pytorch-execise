"""
假设学生花 x 个小时学习，最后考试时能获得 y 分。

假设 y = wx  。即 y 和 x 是线性关系
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib_cn import matplotlib_chinese

matplotlib_chinese.enable_matplotlib_chinese()


# 训练数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 存储 w 和 mse 的值
w_list, mse_list = [], []
# 暴力穷举来猜测 w 的值
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    loss_sum = 0
    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]

        # 预测值
        y_pred = w * x

        # 计算平方误差
        loss = (y_pred - y) ** 2

        # 平方误差的和
        loss_sum += loss
        print("\t", "x=", x, "y=", y, "y_pred=", y_pred, "loss=", loss)

    # 计算均方误差
    mse = loss_sum / len(x_data)
    print("\t", "mse=", mse)

    w_list.append(w)
    mse_list.append(mse)


# 绘制 w 和 mse 的关系
plt.plot(w_list, mse_list)
plt.xlabel("w")
plt.ylabel("mse")
plt.title("w 和 mse 的关系")
plt.show()


# 找到最小 mse 对应的 w
min_mse = min(mse_list)
min_index = mse_list.index(min_mse)
best_w = w_list[min_index]
print("最小 mse=", min_mse, "对应的 w=", best_w)   # 最小 mse= 0.0 对应的 w= 2.0

