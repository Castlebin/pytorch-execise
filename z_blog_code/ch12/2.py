# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

"""
循环神经网络函数RNN
（输入数据（数据的时间维度x的个数，batch，单个数据的维度），记忆体数据（层数（也即第几层的hidden），hidden的维度），层数）
"""
cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers)

# (seqLen, batchSize, inputSize)
inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

"""
输出数据：
out：（seq_len，batch，hidden数据维度）--最上边的那一行，横着的，也就是输出
hidden：（层数，batch，hidden数据维度）
"""
out, hidden = cell(inputs, hidden)
print('Output size: ', out.shape)
print('Output: ', out)
print('Hidden size: ', hidden.shape)
print('Hidden: ', hidden)