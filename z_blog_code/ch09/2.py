# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch

y = torch.LongTensor([0])
z = torch.Tensor([[0.2, 0.1, -0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z, y)
print(loss)

"""
Torch.nn.CrossEntropyLoss()
交叉验证熵求解
torch直接封装好了，我们直接用，但是我们需要把数据改为张量（tensor）
封装的包括：softmax 函数：（各个数据的：e^x / sum(e^x)）； 然后进行log变换，同时进行 loss函数(-ylogy_^)的求解
把后面的步骤都写出来了
所以我们的数据只需计算到最后一层，不需要进行softmax计算就好，
"""