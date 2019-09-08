# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @autor :lxztju
# @github : https://github.com/lxztju

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim


from load_data import train_dataloader,train_datasets
from models.densenet import densenet169
import cfg


##命令行交互，设置一些基本的参数
parser = argparse.ArgumentParser("Train the densenet")

parser.add_argument('-max', '--max_epoch', default=120,
                    help = 'maximum epoch for training')

parser.add_argument('-b', '--batch_size', default=64,
                    help = 'batch size for training')

parser.add_argument('-ng', '--ngpu', default=2,
                    help = 'use multi gpu to train')

parser.add_argument('-lr', '--learning_rate', default=5e-4,
                    help = 'initial learning rate for training')

##训练保存模型的位置
parser.add_argument('--save_folder', default='trained_model',
                    help='the dir to save trained model ')

args = parser.parse_args()


##创建训练模型参数保存的文件夹
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


#####build the network model

model = densenet169(num_classes=cfg.NUM_CLASSES)

#打印模型架构
print(model)

###load the pretrained weights，下载的预训练模型
pretrained_path = 'PRETRAINED_MODEL

print("Initializing the network ...")
#读入预训练模型的参数
#pytorch存储的模型一般采取存储网络参数格式，保存的格式为字典的格式，键为定义每层op的名字，值为保存的参数值

state_dict = torch.load(pretrained_path)


###去掉全链接层的权重，
#由于我们一般不会直接使用，imagenet的1000类，因此，我们需要更换网络最后的全链接层
#因此我们需要将前边几层的参数保存，最后一层重新初始化
#定义一个新的字典，将原始的参数字典，对应保存与更改
from collections import OrderedDict
new_state_dict = OrderedDict()

for k,v in state_dict.items():
    # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
    #torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
    # 直接利用if语句筛选不一致的键
    ###修正键值的不对应
    if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
        k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
    # print(k)
    else:
        pass
    ##最后一层的全连接层，进行初始化
    if k.split('.')[0] == 'classifier':
        if k.split('.')[-1] == 'weights':
            v = nn.init.kaiming_normal(model.state_dict()[k], mode='fan_out')
        else:
            model.state_dict()[k][...] = 0.0
            v = model.state_dict()[k][...]
    else:
        pass
    ##得到新的与定义网络对应的预训练参数
    new_state_dict[k] = v
##导入网络参数
model.load_state_dict(new_state_dict)

##进行多gpu的并行计算
if args.ngpu:
    model = nn.DataParallel(model,device_ids=list(range(args.ngpu)))
print("initialize the network done")

###模型放置在gpu上进行计算
if torch.cuda.is_available():
    model.cuda()

##定义优化器与损失函数
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_func = nn.CrossEntropyLoss()


# for epoch in range(args.max_epoch):
args.batch_size = cfg.BATCH_SIZE

#每一个epoch含有多少个batch
max_batch = len(train_datasets)//args.batch_size

##训练max——epoch个epoch
for epoch in range(args.max_epoch):
    model.train()  ##在进行训练时加上train()，测试时加上eval()
    ##在测试时加上eval()会将BN与Dropout的进行固定
    batch = 0

    for batch_images, batch_labels in train_dataloader:
        # print(batch_labels)
        # print(torch.cuda.is_available())
        average_loss = 0
        train_acc = 0

        ##在pytorch0.4之后将Variable 与tensor进行合并，所以这里不需要进行Variable封装
        if torch.cuda.is_available():
            batch_images, batch_labels = batch_images.cuda(),batch_labels.cuda()
        out = model(batch_images)
        loss = loss_func(out,batch_labels)

    #    print(loss)
        average_loss = loss
        prediction = torch.max(out,1)[1]
        # print(prediction)

        train_correct = (prediction == batch_labels).sum()
        ##这里得到的train_correct是一个longtensor型，需要转换为float
        # print(train_correct.type())
        train_acc = (train_correct.float()) / args.batch_size
 #       print(train_acc.type())

        optimizer.zero_grad() #清空梯度信息，否则在每次进行反向传播时都会累加
        loss.backward()  #loss反向传播
        optimizer.step()  ##梯度更新

        batch+=1
        print("Epoch: %d/%d || batch:%d/%d average_loss: %.3f || train_acc: %.2f"
              %(epoch, args.max_epoch, batch, max_batch, average_loss, train_acc))

##每10epoch保存一次模型
    if epoch%10 ==0 and epoch>0:
        torch.save(model.state_dict(), args.save_folder+'/'+'dense169'+'_'+str(epoch)+'.pth')







