# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch

batch_size = 1
seq_len = 5
input_size = 4
hidden_size = 4
num_layers = 1

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

"""
inputs(seq_len 5,batch 1,input_size 4)
"""
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        """
        继承Module类
        super超类
        重写__init__,forward方法
        调用RNN函数（输入数据维度，记忆层数据，层数）
        :param input_size:
        :param hidden_size:
        :param batch_size:
        :param num_layers:
        """
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)

    def forward(self, input):
        """
        创建记忆层的维度（hidden：层数，batch，hidden维度）
        out：最后一层的输出值（即预测值）--(seq_len,batch,input_size)
        _ :最后时刻的hidden的值 -- (num_layers,batch,hidden_size)
        :param input: 输入数据
        :return:seq_len,hidden_size(每个时间的数据都会有个out，我们将每一个out组合起来，
        out中的一个数据和hidden是一样的维度，因为out_1 = g(W_oh * hidden))相当于我们自己定义了out的维度
        """
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size)
        out, _ = self.rnn(input, hidden)

        return out.view(-1, self.hidden_size)


"""
实例化类--net  参数：输入数据，记忆层数据，batch，层数
调用损失函数
选择Adam优化器，模型初始化（W_ih,W_hh,B_ih,B_hh）学习率为0.05
"""
net = Model(input_size, hidden_size, batch_size, num_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    """
    优化器函数的梯度设置为0
    将数据放入net模型中
    计算损失函数
    反向传播
    优化器更新权值，偏置，梯度等
    期初每一个数据的最大值的索引，
    一次输出索引值对应的字母
    输出 迭代次数和损失函数
    """
    # loss = 0
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))