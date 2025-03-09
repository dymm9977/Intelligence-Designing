from main_CNN import datasets, batch_size, loss_fn,  mnist_train,  mnist_test, device

import torch
from torch.utils.data import  DataLoader
import torchvision.models as models
from torchvision.transforms import ToTensor, Compose, Grayscale, Normalize


training_data = datasets.MNIST(
    root="data",  # 存储训练数据集的路径
    train=True,  # 指定训练数据集
    download=True,  # 如果root路径不可用，则从Internet下载数据
    transform=Compose([
        Grayscale(num_output_channels=3),
        ToTensor(),
    ])  # 特征转换为3通道，然后转换为张量
)
test_data = datasets.MNIST(
    root="data",  # 存储测试数据集的路径
    train=False,  # 指定测试数据集
    download=True,
    transform=Compose([
        Grayscale(num_output_channels=3),
        ToTensor(),
    ])  # 特征转换为3通道，然后转换为张量
)

# 实例化训练数据加载器
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# 实例化测试数据加载器
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = models.mobilenet_v2(pretrained=True)  # 加载预训练模型



class_num = 10  # 假设要分类数目是10
channel_in = model.classifier[1].in_features  # 获取fc层的输入通道数
# 然后把原模型的fc层替换成10个类别的fc层
model.classifier[1] = torch.nn.Linear(channel_in,class_num)
# # 对于模型的每个权重，使其不进行反向传播，即固定参数
# for param in model.parameters():
#     param.requires_grad = False
# # 不固定最后一层，即全连接层fc的权值，即最终要训练的权值
# for param in model.classifier.parameters():
#     param.requires_grad = True

print(model)  # 打印模型
model.to(device)

learn_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)  # 设置优化器
if __name__=='__main__':
    #  正式开始训练和验证
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n --------")
        mnist_train(train_dataloader, model, loss_fn, optimizer)
        mnist_test(test_dataloader, model, loss_fn)
    print("Done")