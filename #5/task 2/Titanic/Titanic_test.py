import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import time

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
        x = self.activate(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x

    
# 预测函数
def predict(y_pred):
    return (y_pred > 0.5).int()


## 读取模型
model = Model()
state_dict = torch.load('./#5/task 2/Titanic/Titanic.pth')
model.load_state_dict(state_dict['model'])

# 初始化模型、损失函数和优化器
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 测试集
class TestTitanicDataset(Dataset):
  
    def __init__(self, filepath, scaler):
        xy = pd.read_csv(filepath)
        self.len = xy.shape[0]
        self.passenger_ids = xy['PassengerId'].values  # 保存乘客ID
        
        # 选取相关的数据特征（与训练集相同）
        feature = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
        data = xy[feature].copy()
        
        # 处理缺失值（与训练集相同处理方式）
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        
        # 独热编码
        data = pd.get_dummies(data)
        
        # 使用训练集的scaler进行标准化
        x_data = scaler.transform(data.astype(float))
        
        self.x_data = torch.from_numpy(x_data).float()
    
    def __getitem__(self, index):
        return self.x_data[index]
    
    def __len__(self):
        return self.len
    
    def get_passenger_ids(self):
        return self.passenger_ids


def test():
   
    scaler = joblib.load('./#5/task 2/Titanic/scaler.pkl')
    # 创建测试数据集（使用训练集的scaler）
    test_dataset = TestTitanicDataset('./#5/task 2/Titanic/test.csv', scaler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)
    
    # # 获取乘客ID
    passenger_ids = test_dataset.get_passenger_ids()
    

    torch.cuda.synchronize()
    start = time.time()
    # 预测
    model.eval()  # 设置为评估模式
    all_predictions = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        for x in test_loader:
            y_pred = model(x)
            predicted = predict(y_pred)
            total += x.size(0)
            correct += predicted.sum()
            all_predictions.extend(predicted.numpy().flatten())
        print(f"预测结果统计:")
        print(f"  总样本数: {total}")
        print(f"  生存预测: {correct} 人")
        print(f"  死亡预测: {total - correct} 人")
        print(f"  生存率: {correct/total:.2%}")
    torch.cuda.synchronize()
    end = time.time()
    print(f'模型处理时间：{(end-start)*1000:.4}ms')

    # 创建提交DataFrame
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': all_predictions
    })

    submission.to_csv('./#5/task 2/Titanic/submission.csv', index=False)
    
if __name__=='__main__':#if这条语句在windows系统下一定要加，否则会报错
    t0 = time.time()
    test()
    t1 = time.time()