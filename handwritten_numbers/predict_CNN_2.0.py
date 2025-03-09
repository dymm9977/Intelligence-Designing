import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import gradio as gr


# 定义CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        # 第三个卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        return x


def load_model(model_path, device):
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image):
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整图像大小为28x28
        transforms.Grayscale(),      # 将图像转换为灰度图像
        transforms.ToTensor(),       # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])

    # 预处理图片
    image = transform(image).unsqueeze(0)  # 将图像转换为张量并添加批次维度
    return image


def predict(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'D:/python object/pythonProject3/handwritten_numbers/models/model.pth'
    model = load_model(model_path, device)

    # 预处理图片
    image = preprocess_image(image)

    # 进行预测
    output = model(image.to(device))
    _, predicted = torch.max(output, 1)
    return predicted.item()


# 创建 Gradio 界面
def gradio_interface():
    iface = gr.Interface(
        fn=predict,  # 预测函数
        inputs=gr.Image(type="pil"),  # 输入类型为 PIL 图像
        outputs="text",  # 输出类型为文本
        title="MNIST 手写数字识别",
        description="上传一张手写数字图片，模型将预测该数字。"
    )
    iface.launch()


if __name__ == '__main__':
    gradio_interface()