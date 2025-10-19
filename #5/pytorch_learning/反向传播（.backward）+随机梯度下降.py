import torch
import numpy as np
import matplotlib.pyplot as plt

x_data=[1.0,3.0,5.0]
y_data=[3.0,9.0,15.0]

w=torch.tensor([1.0])
w.requires_grad = True # 需要计算梯度

#torch.Tensor默认创建的是torch.FloatTensor类型的张量(单精度浮点)
#torch.tensor会根据数据类型或dtype参数来确定创建的张量类型

def forward(x):
    return x*w

def loss(x,y):
    y_pd=forward(x)
    return (y_pd-y)**2

'''
# 试了下对图求和时的显存情况
sum=0 
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l =loss(x,y) 
        l.backward() 
        print('\tgrad:', x, y, w.grad.item())
        sum+=l 
        w.data -= 0.01 * w.grad.data
        w.grad.data.zero_() 
    print('progress:', epoch, l.item())
'''

print("predict (before training)", 7, forward(7).item())
for epoch in range(200):
    for x, y in zip(x_data, y_data):
        l =loss(x,y) 
        l.backward() # 从 l 开始，反向传播整个计算图，把每一个参数对 loss 的导数（也叫梯度）都算出来，存在每个张量的 .grad 属性里
        # print('\tgrad:', x, y, w.grad.item())
        w.data -= 0.01 * w.grad.data
        w.grad.data.zero_() 
        # pytorch每次不清零梯度难道是为了方便batch?
    print('progress:', epoch, l.item())

print("predict (after training)", 7, forward(7).item())

