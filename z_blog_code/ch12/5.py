# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码

import torch

num_class = 4
batch_size = 1
seq_len = 5
input_size = 4
hidden_size = 8
num_layers = 2
embedding_size = 10

"""
创建字典，创建由特征索引构成的二维向量（1,5），和对应的标签对应字母索引所构成的（5）
将x_data y_data转换为张量  （1，5）（5）
"""
idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]]
y_data = [3, 1, 2, 3, 2]
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self):
        """
        Model继承Module类
        super超类
        重写__init__,forward方法
        调用Embedding函数，（input_size,embedding_size）--将input_size变成embedding_size(4-10)每一个数据4维变成10维
        调用RNN函数（输入维度，记忆层维度，层数）batch_first=true则喂数据时为：input（batch_size，seq_len,hidden_size）
        batch_first=False则喂数据时为：input(seq_len,batch_size,hidden_size)
        Linear(hidden_size,类别数量)
        """
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        """
        创建记忆体（全是0. 维度（层数，input_size(0) - 即batch_size），hidden_size）
        先对输入数据进行Embedding，嵌入层，将input_size  --  embedding_size；得到高维数据---(batch_size,seq_len,embedding_size)
        将得到的数据进行rnn模型进行训练（x（batch_size,seq_len,embedding_size），hidden（num_layers,batch,hidden_size））
        返回out最后一层的输出，和 hidden最后时刻的记忆体的参数
        out:(batch,seq_len,embedding)
        _ :(num_layer,batch_size, hidden_size)
        通过线性函数Linear。维度是num_class
        .view()展示的是将（out即x）的每一个值都组合起来，变成（seq_len，num_class）batch_size
        :param x: 输入数据input
        """
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)

        return x.view(-1, num_class)


"""
实例化类--net
调用损失函数--criterion
调用Adam优化器，初始化net模型中的参数（权重，偏置），学习率0.05
"""
net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    """
    优化器的梯度设置为0
    将input放入模型，返回out的数据（是矩阵（seq_len，num_class））
    计算损失函数
    反向传播
    优化器迭代，更新参数（权重，偏置，梯度）
    通过max返回每一组数据的最大值对应的索引
    查找出字典对应索引的字母，输出
    输出当前迭代次数和损失值
    """
    # loss = 0
    optimizer.zero_grad()
    outputs = net(inputs)
    print(outputs.shape)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()

    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))