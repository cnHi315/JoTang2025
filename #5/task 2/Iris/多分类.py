import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns

iris = datasets.load_iris()
iris = datasets.load_iris()
X = iris.data  # 特征
y = iris.target  # 标签
class DatasetCreate(Dataset):
    def __init__(self, features, labels):
        # 标准化特征数据
        std = StandardScaler()
        self.x_data = torch.FloatTensor(std.fit_transform(features))
        # 将标签转换为LongTensor用于分类任务
        self.y_data = torch.LongTensor(labels)
        self.len = features.shape[0]

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len



# 创建数据集
dataset = DatasetCreate(X, y)

# 划分数据集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset,batch_size=5,shuffle=True,num_workers=0)
# batch_size是一个batch中有多少个样本，shuffle表示要不要对样本进行随机排列，num_workers表示可以用多少进程并行的运算
test_loader = DataLoader(dataset=test_dataset, shuffle=False,num_workers=0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(4,10) 
        self.l2 = torch.nn.Linear(10,6)
        self.l3 = torch.nn.Linear(6,3) 

        self.sigmoid = torch.nn.Sigmoid() 
        self.activate = torch.nn.ReLU()  # 可更换激活函数
 
    def forward(self, x):
        x = x.view(-1,4) # -1是自动获取mini_batch
        x = self.activate(self.l1(x)) #输出作输入，故直接用x表示
        x = self.activate(self.l2(x))
        return self.l3(x)
    
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
        if batch_idx % 10 == 9:
            print(epoch+1, batch_idx+1, running_loss/10)
            running_loss = 0.0

#测试模型
def test():
    correct = 0
    total = 0
    pred_all = []
    lable_all = []
    with torch.no_grad():
        for data in test_loader:
            input, lables = data
            outputs = model(input)
            _, predicted = torch.max(outputs.data, dim=1) # max取出最大预测类别
            # 最大值，下标
            pred_all.append(predicted)
            lable_all.append(lables)
            total += lables.size(0) # lable为N*1元组
            correct += (predicted == lables).sum().item()
    print("acc =",correct/total)
    # 计算混淆矩阵
    cm = confusion_matrix(lable_all, pred_all)

    # 使用seaborn绘制混淆矩阵
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['setosa', 'versicolor', 'virginica'], yticklabels=['setosa', 'versicolor', 'virginica'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()

model = Model()
 
 

criterion = torch.nn.CrossEntropyLoss()  
# 交叉熵损失函数，包含softmax过程
# softmax通过指数化求概率占比，避免0的出现
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


if __name__=='__main__':#if这条语句在windows系统下一定要加，否则会报错
    for epoch in range(200):
        train(epoch)
        if epoch % 10 == 9:
            test()