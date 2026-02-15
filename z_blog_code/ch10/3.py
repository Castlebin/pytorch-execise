# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch

"""
数据转换：batch，channel，width，height
卷积核2*2，MaxPool2d函数
搭建模型
"""
input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6,
         ]
input = torch.Tensor(input).view(1, 1, 4, 4)

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = maxpooling_layer(input)
print(output)