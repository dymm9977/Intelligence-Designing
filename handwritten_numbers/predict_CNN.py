import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# 定义CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def load_model(model_path, device):
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('L')  # 将图片转换为灰度图像
    image = transform(image).unsqueeze(0)  # 将图像转换为张量并添加批次维度
    return image


def predict(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型路径
    model_path = 'D:/python object/pythonProject3/handwritten_numbers/models/model.pth'

    # 图片路径
    image_path = 'D:/python object/pythonProject3/handwritten_numbers/a.png'  # 替换为你的图片路径

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整图像大小为28x28
        transforms.ToTensor(),  # 转换为张量
    ])

    # 加载模型
    model = load_model(model_path, device)

    # 预处理图片
    image = preprocess_image(image_path, transform)

    # 进行预测
    prediction = predict(model, image, device)

    # 显示结果
    print(f"Predicted class: {prediction}")

    # 可选：显示图片
    image_np = image.squeeze().cpu().numpy()  # 将张量转换为numpy数组并移除批次维度
    plt.imshow(image_np, cmap='gray')
    plt.title(f"Predicted Class: {prediction}")
    plt.axis('off')
    plt.show()