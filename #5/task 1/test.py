import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from matplotlib.colors import ListedColormap

# 设置随机种子：固定随机数
torch.manual_seed(42)
np.random.seed(42)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[128, 64, 32], output_size=2):
        super(NeuralNetwork, self).__init__()
        
        # 构建网络层
        self.layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # 隐藏层之间
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # 最后一个隐藏层到输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # 参数初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """参数初始化"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, x):
        """前向传播"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 隐藏层使用ReLU
                x = self.relu(x)
        
        # 输出层使用Softmax保证概率和为1, 二分中只算一个概率然后取反即可
        x = self.softmax(x)
        return x

def generate_dataset(dataset_type, noise=0.2, n_samples=1000):
    """生成数据集"""
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    else:  # circles
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=42, factor=0.5)
    
    return X, y

def visualize_dataset(X, y, title):
    """可视化数据集"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6)
    plt.title(f'{title} Dataset (Noise Level)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def train_model(model, X_train, y_train, X_val, y_val, epochs=1000, lr=0.01):
    """训练模型"""
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        
        # 前向传播
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        # 验证模式
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, predicted = torch.max(val_outputs.data, 1)
            accuracy = (predicted == y_val_tensor).float().mean()
        
        train_losses.append(loss.item())
        val_accuracies.append(accuracy.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
    
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    return train_losses, val_accuracies, training_time

def plot_training_curves(train_losses, val_accuracies):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(model, X, y, title):
    """绘制决策边界热力图"""
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点的概率
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model(grid_tensor)
        Z = Z[:, 1].numpy()  # 取类别1的概率
    
    Z = Z.reshape(xx.shape)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu, levels=50)
    plt.colorbar(contour, label='Probability of Class 1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', edgecolors='white')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', edgecolors='white')
    plt.title(f'Decision Boundary - {title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def compare_noise_levels():
    """比较不同噪声水平下的模型效果"""
    noise_levels = [0.1, 0.3, 0.5]
    
    for noise in noise_levels:
        print(f"\n{'='*50}")
        print(f"Training with noise level: {noise}")
        print(f"{'='*50}")
        
        # 生成数据
        X, y = generate_dataset(dataset_name, noise=noise)
        
        # 数据标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # 可视化数据集
        visualize_dataset(X_train, y_train, f'{dataset_name} (Noise={noise})')
        
        # 创建模型
        model = NeuralNetwork()
        
        # 训练模型
        train_losses, val_accuracies, training_time = train_model(model, X_train, y_train, X_val, y_val, epochs=500, lr=0.01)
        training_times.append(training_time)

        # 绘制训练曲线
        plot_training_curves(train_losses, val_accuracies)
        
        # 绘制决策边界
        plot_decision_boundary(model, X_train, y_train, f'Noise={noise}')

         # 最终测试准确率
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test_tensor).float().mean()
            print(f"Final Test Accuracy on {dataset_name}: {accuracy.item():.4f}")

# 主程序
if __name__ == "__main__":
    
    # 对比不同数据集
    print(f"\n{'='*60}")
    print("对比不同数据集上的表现")
    print(f"{'='*60}")
    
    datasets = ['moons','circles']
    training_times = []
    
    for dataset_name in datasets:
        print(f"\nTraining on {dataset_name} dataset:")
        # 比较不同噪声水平
        compare_noise_levels()
       
    
    print(f"\nTraining Time Comparison:")
    for i, (dataset_name, _) in enumerate(datasets):
        print(f"{dataset_name}: {training_times[i]:.2f} seconds")