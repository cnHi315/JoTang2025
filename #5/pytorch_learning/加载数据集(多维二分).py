import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''
Dataset是一个抽象函数,不能直接实例化,所以我们要创建一个自己类,继承Dataset
继承Dataset后我们必须实现三个函数:
__init__()是初始化函数，可以提供数据集路径进行数据的加载
__getitem__()帮助通过索引找到某个样本
__len__()帮助返回数据集大小
'''
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        #shape本身是一个二元组（x,y）对应数据集的行数和列数，[0]取行数,即样本数
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len


#糖尿病dataset
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,batch_size=100,shuffle=True,num_workers=0)
# batch_size是一个组中有多少个样本，shuffle表示要不要对样本进行随机排列，num_workers表示可以用多少进程并行的运算


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear1 = torch.nn.Linear(8,6) # x有八维，即含8个feature
        self.linear2 = torch.nn.Linear(6,4) #
        self.linear3 = torch.nn.Linear(4,1) # 神经层 矩阵化向量化运算 降维升维

        self.sigmoid = torch.nn.Sigmoid() 
        self.softmax = torch.nn.Softmax() # 指数化求概率，避免0的出现
        self.activate = torch.nn.Sigmoid()  # 可更换激活函数
        # 实则结果与torch.sigmoid等价，不过torch.nn.Sigmoid有Class标注，故将其看作是神经网络中进行非线性变化的一层，而不是简单的函数使用
 
    def forward(self, x):
        x = self.activate(self.linear1(x)) #输出作输入，故直接用x表示
        x = self.activate(self.linear2(x))
        x = self.sigmoid(self.linear3(x)) #留sigmoid避免y_pd为0，导致对数计算时出错
        return x
    

model = LogisticRegressionModel()
 

criterion = torch.nn.BCELoss(reduction='mean')  # sigmoid对应的损失函数
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)


epoch_list = []
loss_list = []
if __name__=='__main__':#if这条语句在windows系统下一定要加，否则会报错
    for epoch in range(2000):
        for i,data in enumerate(train_loader,0):#取出一个batch
            input, lable = data
            # forward
            y_pred = model(input)
            loss = criterion(y_pred, lable)
            print(epoch, i, loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # update
            optimizer.step()
 

# test


