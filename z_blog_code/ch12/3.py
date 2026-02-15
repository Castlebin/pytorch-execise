# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch

batch_size = 1
# seq_len = 3
input_size = 4
hidden_size = 4
# num_layers = 1


"""
创建字典
设置输入数据中，每一个x的索引位置构成一个列表
同理。构建一个列表y
构建独热向量（one-hot）
利用for循环，搭建出每一个x对应的向量
-----
将搭建好的数据通过.view方法
构建张量
inputs（seq_len，batch,输入数据）（5，1，4）
label（seq_len,batch）
"""
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        """
        Model继承Module类
        重写__init__;forward;init_hidden方法
        init（输入数据维度，记忆体维度，batch）
        super超类
        调用torch..nn.RNNCell(输入数据的维度，记忆体的维度)函数
        :param input_size:（batch，输入数据）
        :param hidden_size:（batch，hidden数据）
        :param batch_size:（batch）
        """
        super(Model, self).__init__()
        # self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        """
        方法，直接使用RNNCell函数，进行训练
        :param input: (batch,inputs)
        :param hidden: (batch,hidden)
        :return: (out（batch，hidden）)
        """
        hidden = self.rnncell(input, hidden)

        return hidden

    def init_hidden(self):
        """
        使用torch中的zeros函数，生成维度为（batch，hidden），数值为（0.）的张量
        :return:
        """

        return torch.zeros(self.batch_size, self.hidden_size)


"""
实例化Model类 -- net
使用CrossEntropyLoss函数计算损失 -- criterion
选择Adam优化器，初始化模型参数（W_hh,W_ih,B_ih,B_hh）,学习率为0.1
"""
net = Model(input_size, hidden_size, batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(15):
    """
    遍历15次
    损失值为更新为0
    梯度值为0
    取出训练集的特征和对应的标签 - 一次取一组，for循环来取
    将特征集填入模型，传出hidden-即输出值
    通过criterion函数计算损失值进行累加
    通过max函数取出，hidden中最大值，返回其索引位置 -- idx
    输出当前预测的结果

    反向传播
    优化器更新参数（权值，偏置，梯度）
    每次循环完，输出当前的循环的损失值
    """
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print('Predicted string: ', end='')
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')

    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss = %.4f' % (epoch + 1, loss.item()))