"""
Example 12-3 Using embedding and linear layer
Train a model to learn:
• “hello” -> “ohlol”
"""
# %% 1. 准备数据
num_class = 4
batch_size = 1
seq_len = 5
input_size = 4
hidden_size = 8
num_layers = 2
embedding_size = 10

# 字典
idx2char = ['e', 'h', 'l', 'o']

x_data = [[1, 0, 2, 2, 3]]  # hello   # 注意数据形状, batch_first
y_data = [3, 1, 2, 3, 2]  # ohlol


import torch
from torch import nn

# 输入数据和标签 (change data)
inputs = torch.LongTensor(x_data)   # 直接用原始数据（后面会接 Embedding ） 
labels = torch.LongTensor(y_data)   # 直接用原始数据（后面会接 Embedding ） 


# %% 2. 定义模型，这里使用 Embedding + RNN 
class Model(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, num_class):
        super(Model, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_class = num_class

        # Embedding 层
        self.emb = nn.Embedding(self.input_size, self.embedding_size)
        # RNN 层
        self.rnn = nn.RNN(input_size=self.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True) # 注意！
        
        # 还要一个全连接层 fc
        self.fc = torch.nn.Linear(self.hidden_size, self.num_class)

    def forward(self, x):
        # 初始化 hidden 为全 0 .  x.size(0) 就是 batch_size
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # 先 Embedding
        y = self.emb(x)
        y, _ = self.rnn(y, hidden)
        y = self.fc(y)
        
        return y.view(-1, self.num_class)


net = Model(input_size, embedding_size, hidden_size, num_layers, num_class)

# %% 3. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数 (有 SoftMax，常用语多分类)
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
