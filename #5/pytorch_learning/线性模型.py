import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
x_data=[1.0,3.0,5.0]
y_data=[3.0,9.0,15.0]

def forward(x):
    return x*w

def loss(x,y):
    y_pd=forward(x)
    return (y_pd-y)**2

w_list=[]
mse_list=[]
for w in np.arange(0.0,4.1,0.1):
    print('w=',w)
    l_sum=0
    for x_val,y_val in zip(x_data,y_data):
        l_sum+=loss(x_val,y_val)
    print('MSE=',l_sum/len(x_data))
    w_list.append(w)
    mse_list.append(l_sum/len(x_data))

plt.plot(w_list,mse_list)
plt.xlabel('w')
plt.ylabel('mse')
plt.show()
'''

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

def forward(x):
    return x*w+b

def loss(x,y):
    y_pd=forward(x)
    return (y_pd-y)**2

w_list=np.arange(0.0,4.1,0.1)
b_lsit=np.arange(-2.0,2.1,0.1)
[w,b]=np.meshgrid(w_list,b_lsit)
mse=np.zeros(w.shape) 
#此处直接用数组计算便于画图，但实际训练模型时需适当输出cost日志了解训练情况
for x_val,y_val in zip(x_data,y_data):
    mse+=loss(x_val,y_val)
mse/=len(x_data) 

fig=plt.figure()
ax=fig.add_axes(Axes3D(fig))
ax.plot_surface(w, b, mse)
plt.show()
