# 使用自定义的 Dataset 和 DataLoader 加载数据集、进行小批量训练

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_cn import matplotlib_chinese

matplotlib_chinese.enable_matplotlib_chinese()


# 使用自定义的 Dataset 和 DataLoader 加载数据集、进行小批量训练
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

## Dataset 定义。需要继承 torch.utils.data.Dataset，并实现 __init__、 __len__ 和 __getitem__ 方法
class DiabetesDataset(Dataset):
    def __init__(self, file_path='diabetes.csv.gz'):
        # 读取数据
        data = np.loadtxt(file_path, delimiter=",", dtype=float)
        self.x_data = torch.tensor(data[:, :-1]).float() # 前几列是输入（特征）
        self.y_data = torch.tensor(data[:, -1:]).float() # 最后一列是输出（标签）

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# dataset
dataset = DiabetesDataset('diabetes.csv.gz')

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# 只使用前 80% 的数据进行训练，后 20% 作为测试集
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# dataloader - 用于小批量训练
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 查看数据形状
# x_data shape: torch.Size([759, 8]), y_data shape: torch.Size([759, 1])
print(f"x_data shape: {dataset.x_data.shape}, y_data shape: {dataset.y_data.shape}")

print(f"训练数据集大小: {len(train_dataset)}")
print(f"测试数据集大小: {len(test_dataset)}")

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)


# 3. 训练模型 # 使用 训练数据集 进行训练
epochs = 1000

# 保存 loss 值用于绘图
loss_list = []
# 准确率
acc_list = []
for epoch in range(epochs):
    model.train()
    # ！！使用 DataLoader 来加载数据集，并设置 batch_size 参数来指定小批量的大小。
    for batch_idx, (x_train_data, y_train_data) in enumerate(train_dataloader):
        y_pred = model(x_train_data) # 使用 训练数据集 进行训练

        loss = loss_fn(y_pred, y_train_data) # 计算损失

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 清零梯度
        optimizer.zero_grad()

        # 只记录每个 epoch 的最后一个小批量的 loss 和 acc
        if batch_idx == len(train_dataloader) - 1:
            loss_list.append(loss.item())  # 记录 loss 值，用于后续绘图
            # 计算准确率
            predicted = (y_pred >= 0.5).float() # 假设阈值为 0.5 。0.5 及以上为 1，以下为 0
            acc = (predicted.eq(y_train_data).sum().item()) / y_train_data.size(0)
            acc_list.append(acc)

    # 每 1/10 个 epochs 输出一次，避免刷屏
    if (epoch+1) % (epochs // 10) == 0:
        print(f"epoch: {epoch+1}, loss={loss.item()}")


# 绘制 loss 曲线
plt.plot(range(len(loss_list)), loss_list)
plt.xlabel("迭代次数 Epoch")
plt.ylabel("损失 Loss")
plt.title("训练过程中的损失变化曲线")
plt.show()


# 绘制 准确率 曲线
plt.plot(range(len(acc_list)), acc_list)
plt.xlabel("迭代次数 Epoch")
plt.ylabel("准确率 Accuracy")
plt.title("训练过程中的准确率变化曲线")
plt.show()


# 计算在测试集上的准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x_test_data, y_test_data in test_dataloader:
        y_test_pred = model(x_test_data)
        predicted = (y_test_pred >= 0.5).float()
        total += y_test_data.size(0)
        correct += (predicted.eq(y_test_data).sum().item())

test_acc = correct / total
print(f"\n测试数据集上的准确率={test_acc}")


'''
可以看到，模型现在收敛得并不是很好
'''


