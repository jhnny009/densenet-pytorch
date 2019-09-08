# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @autor :lxztju
# @github : https://github.com/lxztju


import torch
from models.densenet import densenet121
import cfg
from load_data import val_dataloader,val_datasets


##定义模型的框架
model = densenet121(num_classes=cfg.NUM_CLASSES)
print(model)
##将模型放置在gpu上运行
if torch.cuda.is_available():
    model.cuda()




###读取网络模型的键值对
trained_model = cfg.TRAINED_MODEL
state_dict = torch.load(trained_model)


# create new OrderedDict that does not contain `module.`
##由于之前的模型是在多gpu上训练的，因而保存的模型参数，键前边有‘module’，需要去掉，和训练模型一样构建新的字典
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

print('Finished loading model!')



##进行模型测试时，eval（）会固定下BN与Dropout的参数
model.eval()
eval_acc = 0.0

for batch_images, batch_labels in val_dataloader:
    # print(batch_labels)
    with torch.no_grad():
        if torch.cuda.is_available():
            batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()
##在pytorch0.4的版本之前，使用Variable封装，里边采用volatile=True放置进行反传训练
#在0.4之后，官方推荐torch.no_grad()，Variable PI已经被弃用

    out = model(batch_images)


    prediction = torch.max(out, 1)[1]
    num_correct = (prediction == batch_labels).sum()
    eval_acc += num_correct
    print(eval_acc)
print(' Accuracy of batch : {:.6f}'.format((eval_acc.float()) / (len(val_datasets))))







