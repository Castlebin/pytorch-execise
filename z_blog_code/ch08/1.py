# 加油加油加油
# 认真对待每一行代码
# 努力写好每一行代码
# 搞懂每一行代码
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    """
    DiabetesDataset类继承Dataset类
    重写__init__方法：
    文件读取，传给xy
    去xy的shape[0],即取数据的数量，也即有多少行
    通过torch模块中的from_numpy函数进行数据的读取操作
    返回的  x_data  y_data 都是张量
    重写__getitem__方法：
    返回x_data的索引 和 y_data的索引
    重写__len__方法：
    返回数据第一位的数值
    """

    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


"""
读取文件diabetes文件；返回dataset
通过DataLoader函数，传入参数：数据集，单词训练的样本数量；是否打乱数据集，多线程工作
"""
dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)


class Model(torch.nn.Module):
    """
    继承Module类，
    搭建不同层数中权值参数维度的变化。
    搭建一个sigmoid函数
    都是用torch库中封装好的函数
    """

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        :param x: 特征
        :return: 预测数据
        """
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


"""
模型实例化--model
调用BCELoss损失函数
选择SGD优化器，其中初始化模型参数，学习率设置为0.01
"""
model = Model()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            """
            i：第几次循环；
            data：【张量每个数据集，张量每个数据对应的标签】
            张量每个数据集：一个列表，batch个元素；每个元素是由一条数据中的特征所构成的列表。
            张量每个数据对应的标签：一个列表，batch个元素，每个元素是由数据中的标签构成的列表。
            [tensor([[], [], []]), tensor([[], [], []])]
            """
            # print("i", i)
            # print("data", data)
            inputs, labels = data
            """
            传出列表中的两个元素，都是张量
            inputs:tensor([[], [], []])
            labels:tensor([[], [], []])
            """
            # print("inputs", inputs)
            # print("labels", labels)
            """
            将数据集传入模型，得到预测值的张量形式
            通过criterion函数返回损失值--loss-张量
            输出:第几次循环所有数据集，当前循环下，训练的第几组（一个batch长度为一组）的数据集，和相应的损失值的高精度格式
            梯度值清零
            反向传播
            利用优化器进行参数更新
            """
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            