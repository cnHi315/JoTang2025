import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# ## 读取模型
# model = net()
# state_dict = torch.load('model_name.pth')
# model.load_state_dict(state_dict['model'])


# 数据探索
def explore_data(filepath):
    print("=== 数据探索 ===")
    explore_data = pd.read_csv(filepath)
    print(f"数据集形状: {explore_data.shape}")
    print("\n前5行数据:")
    print(explore_data.head())
    print("\n数据信息:")
    print( explore_data.info())
    print("\n缺失值统计:")
    print(explore_data.isnull().sum())
    print("\n目标变量分布:")
    print(explore_data['Survived'].value_counts())
    print(f"生存率: {explore_data['Survived'].mean():.2%}")


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

        self.scaler = MinMaxScaler()
        x_data = self.scaler.fit_transform(data.astype(float))
        
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
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 8)
        self.linear4 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.activate = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.linear4(x))
        return x

    
# 预测函数
def predict(y_pred):
    return (y_pred > 0.5).int()


# 创建数据集和数据加载器
dataset_all = TitanicDataset('./#5/task 2/Titanic/train.csv')




# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# K折交叉验证
def k_fold_cross_validation(dataset, k, epochs):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    # 初始化模型、损失函数和优化器
    model = Model()
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'\n第 {fold+1} 折训练中...')
        
        # 创建训练集和验证集 
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch, sampler=val_subsampler)
        
        # 训练模型
        for epoch in range(epochs):
            # 训练
            for input, target in train_loader:
                # 确保y的维度正确
                target = target.view(-1, 1)

                # forward
                y_pred = model(input)
                loss = criterion(y_pred, target)

                # backward
                optimizer.zero_grad()
                loss.backward()
                
                # update
                optimizer.step()
        
        # 验证
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for x, y in val_loader:
                y_pred = model(x)
                y = y.view(-1, 1)
                predicted = predict(y_pred)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            val_acc = 100 * correct / total
        
        fold_results.append(val_acc)
    
    return fold_results

    




if __name__=='__main__':#if这条语句在windows系统下一定要加，否则会报错

    param_grid = {
        'lr': [0.001, 0.01, 0.1],
        'dropout_rate': [0.1, 0.2, 0.3, 0.5],
        'batch_size': [8, 16, 32, 64]
    }
    best_score = 0
    best_params = None
    for lr in param_grid['lr']:
        for dropout in param_grid['dropout_rate']:
            for batch in param_grid['batch_size']:
                cv_scores = k_fold_cross_validation(dataset_all, 5, 100)
                mean_score = np.mean(cv_scores)
                print("\n交叉验证结果:")
                print(f"各折准确率: {[f'{score:.2f}%' for score in cv_scores]}")
                print(f"平均准确率: {mean_score:.2f}%")
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'lr': lr, 'dropout_rate': dropout, 'batch_size': batch}
    
    print(f"  最优参数: {best_params}")
    print(f"  准确率: {best_score:.4f}")
                    
        # 最优参数: {'hidden1': 32, 'hidden2': 32, 'hidden_size3': 8, 'lr': 0.01, 'dropout_rate': 0.3, 'batch_size': 16}
        # 准确率: 81.3715