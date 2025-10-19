import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

torch.manual_seed(123)


def create_dataloaders():
    batch_size = 64
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4363, 0.4329, 0.4201], 
                           std=[0.2207, 0.2187, 0.2217])
    ]) # TODO: 定义训练集的数据预处理与增强

    """     
    Resize(64, 64) - 统一图像尺寸
    RandomHorizontalFlip - 随机水平翻转，增强泛化能力
    RandomRotation - 随机旋转±10度, 增加数据多样性
    ColorJitter - 颜色抖动, 模拟不同光照条件s
    RandomAffine - 随机平移, 让模型关注景物本身而非位置
    ToTensor - 转换为PyTorch张量
    Normalize - 使用训练集标准化的均值和标准差 
    """

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4363, 0.4329, 0.4201], 
                           std=[0.2207, 0.2187, 0.2217])
    ]) # TODO: 定义验证集的数据预处理

    train_dataset = datasets.ImageFolder(root='./#5/task 3/custom_image_dataset/train', transform=train_tf) # TODO: 加载训练集，并确保应用训练集的 transform
    val_dataset = datasets.ImageFolder(root='./#5/task 3/custom_image_dataset/val', transform=val_tf) # TODO: 加载验证集

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size) # TODO: 创建训练集 dataloader
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size) # TODO: 创建验证集 dataloader

    return train_loader, val_loader

# 计算数据集图像通道中的均值和标准差
def mean_std_compute():
    mean = []
    std = []
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载图像数据集
    dataset = datasets.ImageFolder(root='./#5/task 3/custom_image_dataset/train', transform=transform)

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # 遍历数据集并计算均值和标准差
    for images, _ in dataloader:
        batch_mean = torch.mean(images, dim=(0, 2, 3))
        batch_std = torch.std(images, dim=(0, 2, 3))
        mean.append(batch_mean)
        std.append(batch_std)
    # dim 0: batch_size - 批次中的图像数量
    # dim 1: channels - 颜色通道数 (RGB = 3)
    # dim 2: height - 图像高度
    # dim 3: width - 图像宽度

    # 计算所有批次的均值和标准差的平均值
    mean = torch.stack(mean).mean(dim=0)
    std = torch.stack(std).mean(dim=0)

    print("图像通道的均值：", mean)
    print("图像通道的标准差：", std)
    # 图像通道的均值： tensor([0.4363, 0.4329, 0.4201])
    # 图像通道的标准差： tensor([0.2207, 0.2187, 0.2217])  