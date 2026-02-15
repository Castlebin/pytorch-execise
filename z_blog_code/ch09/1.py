# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import numpy as np

y = np.array([1, 0, 0])
z = np.array([0.2, 0.1, -0.1])
y_pred = np.exp(z) / np.exp(z).sum()
loss = (-y * np.log(y_pred)).sum()
print(loss)
"""
这个是实现：我们给出真实的标签，和我们最后的输出层的数据；然后我们通过后softmax函数进行变换
softmax函数：e的x次方除以e的x次方的和，这个时候我们就可以保证每一个值是可以实现总和等于1的
同时我们就可以计算损失函数 -[ylog(y_^) + (1 - y)log(1 - y_^)]
我们可以直接写成 -ylog(y_^)
"""