import torch
import os
from torchvision import transforms
from PIL import Image
from glob import glob
from model import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
model.to(device)
model.load_state_dict(torch.load("./#5/task 3/results/q1_model.pt", weights_only=True))
model.eval()

test_images = sorted(glob("./#5/task 3/custom_image_dataset/test_unlabeled/*.png"))

# TODO: 创建测试时的图像 transformations.
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4363, 0.4329, 0.4201], 
                        std=[0.2207, 0.2187, 0.2217])
])

test_write = open("./#5/task 3/results/q1_test.txt", "w")
for imgfile in test_images:
    filename = os.path.basename(imgfile)
    img = Image.open(imgfile)
    img = test_tf(img)
    img = img.unsqueeze(0).to(device) # 添加批次维度并移动到设备

    # TODO: 使模型进行前向传播并获取预测标签，predicted 是一个 PyTorch 张量，包含预测的标签，值为 0 到 9 之间的单个整数（包含 0 和 9）
    predicted = None
    with torch.no_grad():  # 禁用梯度计算以节省内存
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)  # 获取最大值的索引

    test_write.write(f"{filename},{predicted.item()}\n")
test_write.close()