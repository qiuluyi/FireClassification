import torch
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import os
from PIL import Image
# import cv2 检测用
TRAINED_MODEL = "class_checkpoint/"
# 测试集预处理参数设置
test_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用gpu
model = torch.load(TRAINED_MODEL + 'train_model_80.pth') # 用来加载torch.save()保存的模型文件
model = model.to(device)


label_dict = {0:"正常图像",1:"烟雾图像",2:"火灾图像"} # 标签字典

img_dir = "dataset/test/"
list_img = os.listdir(img_dir) # 列举指定目录(test/show)中的文件名
for img_name in list_img: # 迭代目录下所有图像
    path_img = img_dir + img_name # 图像路径
    img_src = Image.open(path_img) # 打开图像
    # img = cv2.imread(path_img) 检测用
    # img = img[:, :, ::-1]

    img = test_transforms(img_src) # 测试集预处理
    img = torch.unsqueeze(img, 0) # 扩展行
    img = img.to(device) # 返回使用GPU的张量

    with torch.no_grad():
        model.eval()  # 验证
        outputs = model(img) # 经过resnet50的输出
        index = outputs.argmax() # 返回outputs所有元素最大值的索引
        index = index.item() # 转为数
        print(label_dict[index])

        plt.figure(label_dict[index])
        # plt.imshow(img_src) # 显示图像img_src
        plt.show()