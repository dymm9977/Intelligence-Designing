#导入相关库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomAffine, Normalize, Compose
import argparse #命令行参数解析库： 添加参数 解析参数 输出参数


# 定义CNN网络/模型
class CNN(nn.Module):#括号内装父类
    def __init__(self):
        super().__init__()#调用父类初始化方法
        # 第一个卷积层 *参数
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
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        # 第三个卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):#前向传播 （用）
        x = self.conv1(x)#第一个卷积层
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)#展平特征图
        x = nn.ReLU()(self.fc1(x))
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        return x

#载入数据
def load_data(batch_size):
    # 数据增强
    train_transform = Compose([ #组合
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),  # MNIST 数据集的均值和标准差
        RandomHorizontalFlip(),#随机水平翻转
        RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))#随机仿射变换
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

#数据变量
    training_data = datasets.MNIST(
        root="./data",#优先调用保存的数据集
        train=True,
        download=True,#如果数据不在，自动下载
        transform=train_transform,#调用增强函数变换
    )

    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

#数据加载器
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader #返回加载器

#训练函数
def mnist_train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) #原图 标签
        pred = model(X) #前向传播 包含预测结果
        loss = loss_fn(pred, y)
        optimizer.zero_grad() #优化器梯度清零
        loss.backward() #反向传播
        optimizer.step() #更新参数
        if batch % 100 == 0: #打印训练进度
            loss = loss.item() #标量形式
            current = batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

#验证函数 ？测试？验证
def mnist_test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset) #每个批次内容
    num_batches = len(dataloader) #获取批次数量
    model.eval()#验证模式
    test_loss, correct = 0, 0
    with torch.no_grad():#禁用梯度计算
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #*
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model on MNIST')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    batch_size = args.batch_size
    learn_rate = args.learning_rate
    epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_dataloader, test_dataloader = load_data(batch_size)

    # 实例化模型
    model = CNN().to(device)

    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # 正式开始训练和验证
    for t in range(epochs):
        print(f"Epoch {t + 1}\n --------")
        mnist_train(train_dataloader, model, loss_fn, optimizer, device)
        mnist_test(test_dataloader, model, loss_fn, device)

    print("Done")

    # 保存模型
    save_path = 'D:/python object/pythonProject3/handwritten_numbers/models/model.pth'  # 保存模型：1.绝对路径 2.加模型尾缀
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")