import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(), #变化为张量 归一化
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化   
    # MNIST数据集的灰度图像的均值为0.1307，标准差为0.3081
    # 将数据减去其均值，再除以其标准差，使得数据呈现均值为0、方差为1的分布
    # 标准化后的数据在经过激活函数后，其导数较大，加快梯度下降算法
])
 
train_dataset = datasets.MNIST(root='./#5/pytorch_learning/dataset/mnist/', 
                               train=True, 
                               download=True, 
                               transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='./#5/pytorch_learning/dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(784,512) 
        self.l2 = torch.nn.Linear(512,256) 
        self.l3 = torch.nn.Linear(256,128) 
        self.l4 = torch.nn.Linear(128,64) 
        self.l5 = torch.nn.Linear(64,10) # 从784维(宽高)数据到10维(数字类别)数据

        self.sigmoid = torch.nn.Sigmoid() 
        self.activate = torch.nn.ReLU()  # 可更换激活函数
 
    def forward(self, x):
        x = x.view(-1,784) # -1是自动获取mini_batch
        x = self.activate(self.l1(x)) #输出作输入，故直接用x表示
        x = self.activate(self.l2(x))
        x = self.activate(self.l3(x))
        x = self.activate(self.l4(x))
        return self.l5(x)
    
# 训练模型
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader,0):
        input, target = data

        # forward
        output = model(input)
        loss = criterion(output, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # update
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(epoch+1, batch_idx+1, running_loss/300)
            running_loss = 0.0

#测试模型
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, lables = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1) # max取出最大预测类别
            # 最大值，下标
            total += lables.size(0) # lable为N*1元组
            correct += (predicted == lables).sum().item()
    print("acc =",correct/total)


model = Model()
 
 

criterion = torch.nn.CrossEntropyLoss()  
# 交叉熵损失函数，包含softmax过程
# softmax通过指数化求概率占比，避免0的出现
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


if __name__=='__main__':#if这条语句在windows系统下一定要加，否则会报错
    for epoch in range(100):
        train(epoch)
        if epoch % 10 == 9:
            test()