# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

"""
通道---可以想想成厚度
输入数据的设置：
batch_size = 一次集合多少数据
输入通道数
数据平面的维度，宽和高
构建卷积层：
利用卷积核进行卷积操作：nn.Conv2d ：
输入通道，输出通道 卷积核的大小，就卷积核的平面层；
"""
input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)

conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)