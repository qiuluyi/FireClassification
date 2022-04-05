import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

# 针对每一张图像进行分类，包括三种：正常图像，烟雾图像，火灾图像
def FireClassification(image_path):
    # 测试集预处理参数设置
    test_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu
    model = torch.load('class_checkpoint/train_model_140.pth')  # 用来加载torch.save()保存的模型文件
    model = model.to(device)
    label_dict = {0: "norm", 1: "smoke", 2: "fire"}  # 标签字典


    img_src = Image.open(image_path)  # 打开图像
    img = test_transforms(img_src)  # 测试集预处理
    img = torch.unsqueeze(img, 0)  # 扩展行
    img = img.to(device)  # 返回使用GPU的张量

    with torch.no_grad():
        model.eval()  # 验证
        outputs = model(img)  # 经过resnet50的输出
        index = outputs.argmax()  # 返回outputs所有元素最大值的索引
        index = index.item()  # 转为数
        plt.figure(label_dict[index])

        return label_dict[index]

# 检查测试效果
def test():
    test_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu
    model = torch.load('class_checkpoint/train_model_140.pth')  # 用来加载torch.save()保存的模型文件
    model = model.to(device)

    label_dict = {0: "正常图像", 1: "烟雾图像", 2: "火灾图像"}  # 标签字典

    img_dir = "dataset/test/"
    list_img = os.listdir(img_dir)  # 列举指定目录(test/show)中的文件名
    for img_name in list_img:  # 迭代目录下所有图像
        path_img = img_dir + img_name  # 图像路径
        img_src = Image.open(path_img)  # 打开图像
        img = test_transforms(img_src)  # 测试集预处理
        img = torch.unsqueeze(img, 0)  # 扩展行
        img = img.to(device)  # 返回使用GPU的张量

        with torch.no_grad():
            model.eval()  # 验证
            outputs = model(img)  # 经过resnet50的输出
            index = outputs.argmax()  # 返回outputs所有元素最大值的索引
            index = index.item()  # 转为数
            print(label_dict[index])

            plt.figure(label_dict[index])
            plt.imshow(img_src) # 显示图像img_src
            plt.show()

# 调用方法
if __name__=="__main__":
    # test()
    image= "dataset/test/6.jpg"
    if(FireClassification(image)=="norm"):
        print("正常图像")
    elif FireClassification(image)=="smoke":
        print("烟雾图像，系统预警！")
    elif FireClassification(image)=="fire":
        print("火灾图像")