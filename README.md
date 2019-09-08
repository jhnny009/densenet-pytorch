# pytorch训练自己的图像分类数据集

这是一个从头开始训练自己的图像数据集的pytorch应用，对于初学者来说一个很好的学习实践pytorch的应用案例，
采用的网络结构为densenet169,看懂代码的情况下，可以简单的修改几个地方，就可以更换为torchvision的各种其他的网络架构

## 准备自己的分类数据集

数据集的格式为，一个总的文件夹，然后将同一类的图片放置在同一个文件夹下，将包含每类图片的文件夹放置在总的文件夹下。
在load_data.py中利用torchvision.datasets中的ImageFolder类直接进行封装，这个类直接将每一个文件夹下的图片与类别对应，返回结果为迭代器。
然后将这个迭代器传入dataloader，按每个batch提取数据

## 编写模型网络架构

参考torchvision中的models

## 训练网络

训练网络代码放置在train.py中
根据提供在cfg.py中的url下载预训练权重文件，然后在cfg.py中设置好预训练模型的路径，训练集的路径，然后直接在终端中运行 

``` python
python3 train.py 
```

## 参数

模型训练的一些参数放置在cfg.py中

## 评估模型准确率

评估模型的代码放置在eval.py中， 在cfg.py中设置好训练好模型保存的路径TRAINED_MODEL,然后直接运行

``` python
python3 eval.py

```

## 预测

输入单张图片进行预测的代码放置在predict.py中,设置好TRAINED_MODEL,和网络输出值与类别的字典labels_to_classes,
然后直接运行 

``` python
python3 predicts.py
```

代码很简单，适合初学者来熟悉pytorch使用流程，虽然简单但是很实用，中间介绍了从数据集读入到训练模型，再到输入图片进行预测，
这些过程均包含在内，会对简单的图像分类有一个整体的清晰的认识，代码文件中的注释及一些注意事项的注释介绍也比较详细

## 华为modelarts平台部署pytorch模型

这个project主要是自己简单写了一下，用于华为云智能垃圾分类大赛的，自己写完之后，就简单整理了一下，帮助入门pytorch的初学者入门参考吧

这个代码可以应用于华为云的modelarts平台，进行模型的导入与模型的部署，config.json和customize_service.py放置在deployment文件夹的model中，
当训练好之后，将得到的pth也放置在model文件夹中，然后导入华为云的obs存储，然后导入模型即可
这一部分参考华为云bbs，https://bbs.huaweicloud.com/forum/thread-14845-1-1.html


