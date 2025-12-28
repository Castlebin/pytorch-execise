import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_cn import matplotlib_chinese

matplotlib_chinese.enable_matplotlib_chinese()

# 0. 训练数据集
data = np.loadtxt("diabetes.csv.gz", delimiter=",", dtype=float)
## 数据集拆分为输入和输出。前几列是输入，最后一列是输出
x_data = torch.tensor(data[:, :-1]).float() # 前几列是输入（特征）
y_data = torch.tensor(data[:, -1:]).float() # 最后一列是输出（标签），这里 y 只有 0 和 1 两种取值。代表是否患有糖尿病

# 只使用前 80% 的数据进行训练
train_size = int(0.8 * len(x_data))

# 训练数据集
x_train_data = x_data[:train_size]
y_train_data = y_data[:train_size]

# 测试数据集
x_test_data = x_data[train_size:]
y_test_data = y_data[train_size:]

# 查看数据形状
# x_data shape: torch.Size([759, 8]), y_data shape: torch.Size([759, 1])
print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")

# 查看前 3 条数据
print("x 前 3 条数据：", x_data[:3])
print("y 前 3 条数据：", y_data[:3])

'''
x 有 8 个特征
'''


# 1. 定义模型
class MultipleDimensionInputModel(torch.nn.Module):
    def __init__(self):
        super(MultipleDimensionInputModel, self).__init__()
        # 定义一个多层模型，初始输入维度是 8 （必须跟输入 x 的维度一致）
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate_fn = torch.nn.Sigmoid() # 激活函数。这里公用一个，在下面的 forward 中多次使用

    def forward(self, x):
        y_pred = self.linear1(x)            # 线性层 1
        y_pred = self.activate_fn(y_pred)   # 激活函数

        y_pred = self.linear2(y_pred)       # 线性层 2
        y_pred = self.activate_fn(y_pred)   # 激活函数

        y_pred = self.linear3(y_pred)       # 线性层 3
        y_pred = self.activate_fn(y_pred)   # 激活函数

        return y_pred

model = MultipleDimensionInputModel()

# 2. 定义损失函数和优化器
loss_fn = torch.nn.BCELoss(reduction='mean')   # 二分类交叉熵损失函数，适用于逻辑回归（分类场景）
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# 3. 训练模型 # 使用 训练数据集 进行训练
epochs = 10000

# 保存 loss 值用于绘图
loss_list = []
# 准确率
accuracy_list = []
for epoch in range(epochs):

    # ！！ 注意，我们这里使用的是整个数据集进行训练（批量梯度下降法），而不是小批量数据
    # 正常情况下，数据集会被拆分为多个小批量进行训练（小批量梯度下降法），以提高训练效果。即 Mini-batch
    # 这里为了简化代码，直接使用整个数据集进行训练。后续将介绍如何使用 DataLoader 来实现小批量训练。
    # 正常情况下，我们都会使用 DataLoader 来加载数据集，并设置 batch_size 参数来指定小批量的大小。
    # 而不是直接使用整个数据集进行训练。
    y_pred = model(x_train_data) # 使用 训练数据集 进行训练

    loss = loss_fn(y_pred, y_train_data) # 计算损失

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 清零梯度
    optimizer.zero_grad()

    # 计算准确率
    y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
    acc = torch.eq(y_pred_label, y_train_data).sum().item() / y_train_data.size(0)
    accuracy_list.append(acc)

    # 每 1/10 个 epochs 输出一次，避免刷屏
    if (epoch+1) % (epochs // 10) == 0:
        print(f"epoch: {epoch+1}, loss={loss.item()}, accuracy={acc}")

    loss_list.append(loss.item()) # 记录 loss 值，用于后续绘图


# 绘制 loss 曲线
plt.plot(np.arange(1, epochs + 1), loss_list)
plt.xlabel("迭代次数 Epoch")
plt.ylabel("损失 Loss")
plt.title("训练过程中的损失变化曲线")
plt.show()


# 绘制 accuracy 曲线
plt.plot(np.arange(1, epochs + 1), accuracy_list)
plt.xlabel("迭代次数 Epoch")
plt.ylabel("准确率 Accuracy")
plt.title("训练过程中的准确率变化曲线")
plt.show()


# 4. 使用模型进行预测 # 使用 测试数据集 进行测试
y_test_pred = model(x_test_data)
y_test_pred_label = torch.where(y_test_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
test_acc = torch.eq(y_test_pred_label, y_test_data).sum().item() / y_test_data.size(0)
print(f"\n测试数据集上的准确率={test_acc}")

