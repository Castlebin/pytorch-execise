# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

"""
单个RNN
即循环神经网络的单个部分
多个部分组成的是RNN
"""
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# (seq, batch, features)
"""
生成标准正态分布的（均值为零，方差为一，的高斯白噪声）的随机数；
.randn生成（3，1，4）  --  张量
生成为全为0.的数，
.zeros生成（1，2）  --  张量
"""
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)
print(dataset)

for idx, input in enumerate(dataset):
    """
    idx：第几次遍历
    input：取出当前次数的数据，数据维度是（1，4）
    cell（输入数据（batch，特征的向量维度），记忆体维度（batch，hidden的维度））
    """
    print('=' * 20, idx, '=' * 20)
    hidden = cell(input, hidden)
    print('outputs size: ', hidden.shape)
    print(hidden)