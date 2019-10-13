# -*- coding:utf-8 -*-
# @time :2019.09.06
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
import torch
from torchvision import transforms, datasets
import cfg



# 构建数据提取器，利用dataloader
# 利用torchvision中的transforms进行图像预处理
#cfg为config文件，保存几个方便修改的参数

input_size = cfg.INPUT_SIZE
batch_size = cfg.BATCH_SIZE

train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

##ImageFolder对象可以将一个文件夹下的文件构造成一类
#所以数据集的存储格式为一个类的图片放置到一个文件夹下
#然后利用dataloader构建提取器，每次返回一个batch的数据，在很多情况下，利用num_worker参数
#设置多线程，来相对提升数据提取的速度

train_dir = cfg.TRAIN_DATASET_DIR
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)


val_dir = cfg.VAL_DATASET_DIR
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=2)


##进行数据提取函数的测试
if __name__ =="__main__":

    for images, labels in train_dataloader:
        print(labels)
