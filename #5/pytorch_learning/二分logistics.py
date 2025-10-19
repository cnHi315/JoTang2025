import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
 
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])#学习时长
# y_data = torch.Tensor([[0], [0], [1]])#及格概率

#糖尿病dataset
xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32) # 文件 分隔符 数据类型
x_data = torch.from_numpy(xy[:, :-1]) # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
y_data = torch.from_numpy(xy[:, [-1]]) # [-1] 取最后一列，最后得到的是个矩阵

# MNIST数据集导入
# train_set=torchvision.datasets.MNIST(root='../dataset/mnist',train=True,download=True)
# test_set=torchvision.datasets.MNIST(root='../dataset/mnist',train=False,download=True)

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear1 = torch.nn.Linear(8,6) # x有八维，即含8个feature
        self.linear2 = torch.nn.Linear(6,4) #
        self.linear3 = torch.nn.Linear(4,1) # 神经层 矩阵化向量化运算 降维升维

        self.sigmoid = torch.nn.Sigmoid() 
        self.activate = torch.nn.Sigmoid()  # 可更换激活函数
        # 实则结果与torch.sigmoid等价，不过torch.nn.Sigmoid有Class标注，故将其看作是神经网络中进行非线性变化的一层，而不是简单的函数使用
 
    def forward(self, x):
        x = self.activate(self.linear1(x)) #输出作输入，故直接用x表示
        x = self.activate(self.linear2(x))
        x = self.sigmoid(self.linear3(x)) #留sigmoid避免y_pd为0，导致对数计算时出错
        return x
    

model = LogisticRegressionModel()
 
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
criterion = torch.nn.BCELoss(size_average = True)  # sigmoid对应的损失函数
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


epoch_list = []
loss_list = []

for epoch in range(10):
    # forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()
    
    # update
    optimizer.step()
 
""" 现任务：
输出训练日志，使能了解训练效果(以acc呈现)
 """
# x_test = torch.Tensor([[4.0]])
# y_test = model(x_test)
# print('y_pred = ', y_test.data)
plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()

'''
#一维二分时的可视化
x=np.linspace(0,10,200)# 使用NumPy创建一个从0到10的等间距数组，包含200个点
x_t=torch.tensor(x).view((200,1)).float()
y_t=model(x_t)
y=y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid() # 添加网格线
plt.show()
'''

