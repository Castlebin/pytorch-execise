"""
Example 12-1: Using RNNCell
Train a model to learn:
• “hello” -> “ohlol”
"""
# %% 1. 准备数据
input_size = 4
hidden_size = 4
batch_size = 1

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

# 输入数据和标签
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)


# %% 2. 定义模型，这里使用 RNNCell (不是直接用 RNN)
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnncell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden


net = Model(input_size, hidden_size, batch_size)

# %% 3. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数 (有 SoftMax，常用语多分类)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)  # Adam 优化器


# %% 4. 模型训练
epoches = 15
for epoch in range(epoches):
    loss = 0
    
    # 每一轮训练开始，梯度清零
    optimizer.zero_grad()
    
    # 隐藏层初始化为全 0
    hidden = torch.zeros(batch_size, hidden_size)

    predict_chars = []
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)   # 循环训练。将上一次计算得到的 hidden 作为下一次输入的组成部分
        
        loss += loss_fn(hidden, label)
        _, idx = hidden.max(dim=1)
        predict_char = idx2char[idx.item()]
        
        predict_chars.append(predict_char)
        
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{epoches}] loss = {loss.item()}. predict: {"".join(predict_chars)}')
    
