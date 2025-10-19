import torch
import numpy as np
import matplotlib.pyplot as plt

# y= w1 * x^2 + w2 * x +b
x_data=[1.0,3.0,5.0]
y_data=[3.0,9.0,15.0]

w1=torch.tensor([1.0])
w2=torch.tensor([1.0])
b=torch.tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True # 需要计算梯度


def forward(x):
    return w1*(x**2)+w2*x+b

def loss(x,y):
    y_pd=forward(x)
    return (y_pd-y)**2

print("predict (before training)", 7, forward(7).item())
for epoch in range(10000):
    l=(1,3)
    for x, y in zip(x_data, y_data):
        l =loss(x,y) 
        l.backward()
        print(' grad:', x, y, w1.item(), w2.item(),b.item(), '\n\t', w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data -= 0.001 * w1.grad.data
        w2.data -= 0.01 * w2.grad.data
        b.data -= 0.001 * b.grad.data  # learnrate为0.01时学习率过大导致w1梯度爆炸了
        w1.grad.data.zero_() 
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('progress:', epoch, l.item())

print("predict (after training)", 7, forward(7).item())

