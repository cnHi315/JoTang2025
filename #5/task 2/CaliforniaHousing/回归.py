import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torchmetrics.regression import R2Score



class DatasetCreate(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        # xy为numpy二维数组

        self.len = xy.shape[0]
        #shape本身是一个二元组（x,y）对应数据集的行数和列数，[0]取行数,即样本数

        std = StandardScaler()
        self.y_data = torch.from_numpy(xy[:, [-1]])
        xy=std.fit_transform(xy)
        self.x_data = torch.from_numpy(xy[:, :-1]) # 二维数组

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

    def split(self,train_rate):
            train_size = int(train_rate * len(self))
            valid_size = self.len - train_size
            train_dataset, valid_dataset = torch.utils.data.random_split(self, [train_size, valid_size])
            return train_dataset, valid_dataset

dataset = DatasetCreate('./#5/task 2/CaliforniaHousing/cal_housing.data')
train_dataset, test_dataset = dataset.split(0.8) # 划分数据集
train_loader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=0)
# batch_size是一个batch中有多少个样本，shuffle表示要不要对样本进行随机排列，num_workers表示可以用多少进程并行的运算
test_loader = DataLoader(dataset=test_dataset,batch_size=len(test_dataset), shuffle=False,num_workers=0)



class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.l1 = torch.nn.Linear(8,1) 
 
    def forward(self, x):
        x = self.l1(x) #输出作输入，故直接用x表示
        return x
    

model = LogisticRegressionModel()
 

criterion = torch.nn.MSELoss(reduction='mean') 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

def train(epoch):
    loss_sum=0
    for i,data in enumerate(train_loader,0):#取出一个batch
        input, target = data
        # forward
        y_pred = model(input)
        loss = criterion(y_pred, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        
        loss_sum += loss.item()
        # update
        optimizer.step()
        
        if i % 300 == 299:
            print(epoch+1, i+1, loss_sum/(i+1))
    scheduler.step(loss_sum)
    

def test():
    R2 = 0.0
    input, target = next(iter(test_loader)) #iter()转化数据集为迭代器，next()获取下一个batch
    pred=model(input)
    r2 = R2Score()
    R2 = r2(pred, target)
    print('R²:', R2)

    # 绘制散点图
    plt.scatter(list(target.data), list(pred.data), alpha=0.5)
    plt.xlabel('True House Price')
    plt.ylabel('Predicted House Price')
    plt.title('True vs Predicted House Prices')

    x_line = np.linspace(0, 500000, 100000)
    y_line = x_line
    # 绘制 y=x 线
    plt.plot(x_line, y_line, 'r--')
    plt.show()



if __name__=='__main__':#if这条语句在windows系统下一定要加，否则会报错
    for epoch in range(200):
        train(epoch)
        if epoch % 100 == 99:
            test()

