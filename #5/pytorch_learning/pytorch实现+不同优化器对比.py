import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [3.0], [5.0]])
y_data = torch.tensor([[3.0], [9.0], [15.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__() # 继承父类
        self.linear = torch.nn.Linear(1,1) # 添加Linear对象

    # Module中含有forward,此处进行override
    def forward(self,x):
        y_pd=self.linear(x)
        return y_pd

model = LinearModel()
 
criterion = torch.nn.MSELoss(reduction = 'mean') 
# criterion = torch.nn.MSELoss(size_average = False) 等价，现在少用了 改为reduction='sum'/'mean'/'none'
# 损失函数,参数只需y，y_pd

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # SGD优化器，冒充SGD，实则批量梯度下降
# model.parameters()获取模型的所有参数

# 对比不同优化器效果
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) 
# optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.01)
# optimizer = torch.optim.Adamax(model.parameters(), lr = 0.01)
# optimizer = torch.optim.ASGD(model.parameters(), lr = 0.01)
# optimizer = torch.optim.LBFGS(model.parameters(), lr = 0.01)
# optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.01)
# optimizer = torch.optim.Rprop(model.parameters(), lr = 0.01)


epoch_list=[]
mse_list=[]

for epoch in range(1000):
    y_pd=model(x_data)
    loss=criterion(y_pd,y_data)
    # print(epoch,loss)
    epoch_list.append(epoch)
    mse_list.append(loss.item())

    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()  # 更新

print("w=",model.linear.weight.item())
print("b=",model.linear.bias.item())

x_test=torch.tensor([7.0])
y_test=model(x_test)
print('y_pred=',y_test.item())

plt.plot(epoch_list,mse_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()
