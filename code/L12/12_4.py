"""
Example 12-2: Using RNN Module 
Train a model to learn:
• “hello” -> “ohlol”
"""
# %% 1. 准备数据
batch_size = 1
seq_len = 5
input_size = 4
hidden_size = 4
num_layers = 1

# 字典
idx2char = ['e', 'h', 'l', 'o']

x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

# 将 x_data 转化为 One-Hot 编码
x_one_hot = [one_hot_lookup[x] for x in x_data]

import torch
from torch import nn

# 输入数据和标签 (change data)
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)  # 跟前面例子不一样了
labels = torch.LongTensor(y_data)  # 跟前面例子不一样了


# %% 2. 定义模型，这里直接使用 RNN
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 直接使用 RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)

    def forward(self, input):
        # 初始化 hidden 为全 0 
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


net = Model(input_size, hidden_size, batch_size, num_layers)

# %% 3. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数 (有 SoftMax，常用于多分类)
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)  # Adam 优化器

# %% 4. 模型训练
epoches = 15
for epoch in range(epoches):
    # 每一轮训练开始，梯度清零
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = loss_fn(outputs, labels)

    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()

    predict_chars = [idx2char[x] for x in idx]
    print(f'Epoch [{epoch + 1}/{epoches}] loss = {loss.item()}. predict: {"".join(predict_chars)}')
