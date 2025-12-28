# 使用 Pytorch 进行线性回归

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_cn import matplotlib_chinese

matplotlib_chinese.enable_matplotlib_chinese()

# 0. 训练数据集
x_data = torch.tensor([
    [1.0],
    [2.0],
    [3.0]
])
# 0 代表不通过、1 代表通过
y_data = torch.tensor([
    [0.0],
    [0.0],
    [1.0]
])


# 1. 定义模型 （使用逻辑回归）
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # y = wx + b

    def forward(self, x):
        y_linear = self.linear(x)
        y_pred = torch.sigmoid(y_linear)  # 使用 sigmoid 激活函数
        return y_pred

model = LogisticRegressionModel()

# 2. 定义损失函数和优化器
loss_fn = torch.nn.BCELoss(reduction='sum')   # 二分类交叉熵损失函数，适用于逻辑回归（分类场景）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 3. 训练模型
epochs = 1000

# 保存 loss 值用于绘图
loss_list = []
for epoch in range(epochs):
    y_pred = model(x_data)
    loss = loss_fn(y_pred, y_data)
    print(f"epoch: {epoch+1}, loss={loss.item()}")

    loss_list.append(loss.item()) # 记录 loss 值，用于后续绘图

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 清零梯度
    optimizer.zero_grad()


# 4. 查看模型参数
[w, b] = model.parameters()
print(f"训练结束后，模型参数 w={w.item()}, b={b.item()}")

# 5. 使用模型进行预测
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print(f"预测输入 x=4.0 时，模型输出 y={y_test.item()}")


# 6. 绘制 loss 曲线
plt.plot(np.arange(1, epochs + 1), loss_list)
plt.xlabel("迭代次数 Epoch")
plt.ylabel("损失 Loss")
plt.title("训练过程中的损失变化曲线")
plt.show()


# 7. 绘制分类边界
x_plot = torch.linspace(0, 5, 100).reshape(-1, 1)
y_plot = model(x_plot).detach().numpy()
plt.plot(x_plot.numpy(), y_plot, label="分类边界")
plt.scatter(x_data.numpy(), y_data.numpy(), color='red', label="训练数据")
plt.xlabel("输入 x")
plt.ylabel("输出 y")
plt.title("逻辑回归分类边界")
plt.legend()
plt.show()


# 8. 绘制通过概率曲线，更直观地展示逻辑回归的分类效果（绘制 0.5 作为分界线）
x = np.linspace(0, 10, 200)
x_t = torch.tensor(x).view((200, 1)).float()
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], 'r--')  # 绘制 y=0.5 的分界线
plt.xlabel('Hours')
plt.ylabel('Pass Probability')
plt.title('Logistic Regression: Pass Probability vs. Hours')
plt.grid()
plt.show()


