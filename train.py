import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt

# 数据集的类别
NUM_CLASSES = 3
# 训练时batch的大小
BATCH_SIZE = 4
#训练轮数
NUM_EPOCHS = 150
# 预训练模型的存放位置
# 下载地址：https://download.pytorch.org/models/resnet50-19c8e357.pth
PRETRAINED_MODEL = 'pretraing/resnet50-19c8e357.pth'

# 训练完成，权重文件的保存路径,默认保存在trained_models下
TRAINED_MODEL = 'class_checkpoint/'

# 数据集的存放位置
TRAIN_DATASET_DIR = 'dataset/train/'
VALID_DATASET_DIR = 'dataset/val/'

# 数据预处理
# 训练集预处理参数设置
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), # 将图像随机裁剪到大小256*256，比例0.8:1.0
    transforms.RandomRotation(degrees=15), # 将图像以-15°~+15°随机旋转
    transforms.RandomHorizontalFlip(), # 将图像以默认概率0.5随机水平翻转
    transforms.CenterCrop(size=224), # 将图像中心裁剪到大小224*224
    transforms.ToTensor(), # 将图像转化成张量：0~255→0~1
    transforms.Normalize( # 归一化:0~1→-1~1
        [0.485, 0.456, 0.406], # mean
        [0.229, 0.224, 0.225], # std
        )
])

# 验证集预处理参数设置
test_valid_transforms = transforms.Compose([
         transforms.Resize(256), # 将图像大小调整为256*256
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(
             [0.485, 0.456, 0.406],
             [0.229, 0.224, 0.225])
])

train_directory = TRAIN_DATASET_DIR # 训练集目录
valid_directory = VALID_DATASET_DIR # 验证集目录

batch_size = BATCH_SIZE # 每batch训练的样本数目
num_classes = NUM_CLASSES # 数据集的类别数目

# 训练集预处理
train_datasets = datasets.ImageFolder( # 加载训练集，ImageFolder默认训练数据是按照类别存放在文件夹的
    train_directory,
    transform = train_transforms, # 按设置好的参数预处理训练集
)
train_data_size = len(train_datasets) # 训练集图像数目
train_data = torch.utils.data.DataLoader( # 从训练集中加载数据
    train_datasets, # 加载的数据集
    batch_size = batch_size,
    shuffle = True, # 对每次epoch重新排序
)

# 验证集预处理
valid_datasets = datasets.ImageFolder(
    valid_directory,
    transform = test_valid_transforms,
)
valid_data_size = len(valid_datasets)
valid_data = torch.utils.data.DataLoader(
    valid_datasets,
    batch_size = batch_size,
)

print(train_data_size, valid_data_size)

print("="*50) # 打印一个分隔行
# print(train_datasets.class_to_idx) # 类名对应的索引 检测用
print("="*50)

resnet50 = models.resnet50(pretrained=True) # 返回在ImageNet上预先训练的模型
# print('before:{%s}\n'%resnet50) # 查看resnet50前参数量 检测用
for param in resnet50.parameters():
    param.requires_grad = False # 所有param都不获得梯度 默认为True
fc_inputs = resnet50.fc.in_features # 全连接层fc中的输入个数为fc_inputs
# print(fc_inputs) # 2048 检测用
resnet50.fc = nn.Sequential( # 搭建神经网络
    nn.Linear(fc_inputs, 1024), # 输入个数，输出个数
    nn.LeakyReLU(0.1), # 负斜率角度为0.1

    nn.Linear(1024, 512),
    nn.LeakyReLU(0.1),

    nn.Linear(512, num_classes), # 输出类别数
    nn.LogSoftmax(dim=1), # LogSoftmax(xi) = log(exp(xi)/∑exp(xj))
)
# print('after:{%s}\n'%resnet50) # 查看resnet50后参数量 检测用

loss_func = nn.NLLLoss() # Negative log likelihood loss(负对数似然损失) 在网络的最后一层加LogSoftmax层来获得log-probabilities
# print("optimizer:", resnet50.parameters()) # 检测用
optimizer = optim.Adam(resnet50.parameters(), lr=1e-4) # Adam优化器 要优化的可迭代参数，lr默认1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义训练与验证过程
def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU device=cuda:0
    record = [] # 存入平均训练/验证的误差/准确率
    best_acc = 0.0 # 最佳准确率初始为0.0
    best_epoch = 0 # 最佳epoch数(训练轮数)初始为0

    for epoch in range(epochs): # 每epoch(训练轮)中进行以下操作
        epoch_start = time.time() # 返回自epoch开始的秒数(浮点型) epoch默认1970.1.1
        print("Epoch: {}/{}".format(epoch+1, epochs)) #
        model = model.cuda()
        model.train() # 训练

        train_loss = 0.0 # 训练误差初始为0.0
        train_acc = 0.0 # 训练准确率初始为0.0
        valid_loss = 0.0 # 验证误差初始为0.0
        valid_acc = 0.0 # 验证准确率初始为0.0

        for i, (inputs, labels) in enumerate(train_data): # 获取图像和标签
            inputs = inputs.to(device) # 返回使用gpu的张量inputs
            labels = labels.to(device) # 返回使用gpu的张量labels
            # print(labels)
            # 记得清零
            optimizer.zero_grad() # 梯度清零

            outputs = model(inputs) # 输入经过resnet50的输出

            loss = loss_function(outputs, labels) # 计算误差

            loss.backward() # 误差反向传播

            optimizer.step() # 优化器Adam开始执行优化

            train_loss += loss.item() * inputs.size(0) # 计算总训练误差 loss.item()为误差数值，inputs.size(0)=32为batchsize

            ret, predictions = torch.max(outputs.data, 1) # 返回一维张量outputs.data每行的max
            correct_counts = predictions.eq(labels.data.view_as(predictions)) # 统计预测与标签相同的数目

            acc = torch.mean(correct_counts.type(torch.FloatTensor)) # 计算准确率
            print("epoch =", epoch, "i =", i, "acc =", acc)

            train_acc += acc.item() * inputs.size(0) # 计算总训练准确率

        with torch.no_grad(): # 使requires_grad = False
            model.eval() # 验证

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0) # 计算总验证误差 inputs.size(0)=32

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0) # 计算总验证准确率

        avg_train_loss = train_loss / train_data_size # 计算平均训练误差
        avg_train_acc = train_acc / train_data_size # 计算平均训练准确率

        avg_valid_loss = valid_loss / valid_data_size # 计算平均验证误差
        avg_valid_acc = valid_acc / valid_data_size # 计算平均验证准确率

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc]) # 记录平均训练/验证的误差/准确率到record

        if avg_valid_acc > best_acc: # 记录得到最高准确率的模型
            best_acc = avg_valid_acc # 记录最佳准确率
            best_epoch = epoch+1 # 记录获得最佳准确率的epoch(轮次)

        epoch_end = time.time() # 记录当前epoch到此结束的秒数

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        torch.save(model, TRAINED_MODEL + 'train_model_' + str(epoch+1) + '.pth') # 保存训练模型
    return model, record


if __name__=='__main__': # 以下代码只在此train程序中可被执行，其他程序调用此(import train)也无法执行以下代码块
    num_epochs = NUM_EPOCHS
    trained_model, record = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
    # torch.save(record, TRAINED_RECORD + 'train_model_3000.pth') # 将权重文件record保存
    record = np.array(record)
    plt.plot(record[:, 0:2]) # 显示record中所有元素，前两位置元素(平均训练/验证误差)
    plt.legend(['Train Loss', 'Valid Loss']) # 图例
    plt.xlabel('Epoch Number') # x轴标注
    plt.ylabel('Loss') # y轴标注
    plt.ylim(0, 1) # y轴可视范围0~1
    plt.savefig('loss.png') # 保存名为loss.png
    plt.show()
    plt.close()

    plt.plot(record[:, 2:4]) # 显示record中所有元素，后两位置元素(平均训练/验证准确率)
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('accuracy.png')
    plt.show()
