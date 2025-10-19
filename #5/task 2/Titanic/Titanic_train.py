import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import joblib

# ## 读取模型
# model = net()
# state_dict = torch.load('model_name.pth')
# model.load_state_dict(state_dict['model'])

# 准备数据集
class TitanicDataset(Dataset):
    def __init__(self, filepath):
        xy = pd.read_csv(filepath)
        self.len = xy.shape[0]
        # 选取相关的数据特征
        feature = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
        # np.array()将数据转换成矩阵，方便进行接下来的计算
        # 要先进行独热表示，然后转化成array，最后再转换成矩阵

         # 数据预处理
        data = xy[feature].copy()#防止对原数据进行修改
         # 处理缺失值
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        # 独热编码
        data = pd.get_dummies(data)

         # 数据归一化
        self.scaler = MinMaxScaler()
        x_data = self.scaler.fit_transform(data.astype(float))
        joblib.dump(self.scaler, './#5/task 2/Titanic/scaler.pkl')
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(np.array(xy["Survived"])).float()

    
    # getitem函数，可以使用索引拿到数据
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    # 返回数据的条数/长度
    def __len__(self):
        return self.len
    
    def split(self,train_rate):
            train_size = int(train_rate * len(self))
            valid_size = self.len - train_size
            train_dataset, valid_dataset = torch.utils.data.random_split(self, [train_size, valid_size])
            return train_dataset, valid_dataset


# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(6, 32) 
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 8)
        self.linear4 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.activate = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.dropout(x)
        x = self.activate(self.linear2(x))
        x = self.dropout(x)
        x = self.activate(self.linear3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.linear4(x))
        return x

    
# 预测函数
def predict(y_pred):
    return (y_pred > 0.5).int()


# 创建数据集和数据加载器
dataset_all = TitanicDataset('./#5/task 2/Titanic/train.csv')
train_loader = DataLoader(dataset=dataset_all, batch_size=16, shuffle=True, num_workers=0)

# 初始化模型、损失函数和优化器
model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    loss_sum=0
    for i,data in enumerate(train_loader,0):#取出一个batch
        input, target = data
        # 确保y的维度正确
        target = target.view(-1, 1)#将一维数组升为一个二维的

        # forward
        y_pred = model(input)
        loss = criterion(y_pred, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        
        loss_sum += loss.item()
        # update
        optimizer.step()

        if i % 10== 9:
            print(epoch+1, i+1, loss_sum/10)
    
    
if __name__=='__main__':#if这条语句在windows系统下一定要加，否则会报错

    # explore_data('./#5/task 2/Titanic/train.csv')
    
    # complete_cv_pipeline(dataset_all)

    for epoch in range(50):
        train(epoch)
    # 保存模型
    torch.save({'model': model.state_dict()}, './#5/task 2/Titanic/Titanic.pth')


    # test()
