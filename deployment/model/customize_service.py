#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@version: python3.6
@author: 浅蓝苜蓿(QLMX)
@contact: wenruichn@gmail.com
@time: 2019-08-14 16:07
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61
"""
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torch
from PIL import Image
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


from model_service.pytorch_model_service import PTServingBaseService


num_classes = 40

class classfication_service(PTServingBaseService):
    def __init__(self, model_name, model_path):
        super(classfication_service, self).__init__(model_name, model_path)
        self.model = self.build_model(model_path, num_classes)
        self.model.eval()
        self.label_id_name_dict = \
            {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉",
                "10": "厨余垃圾/茶叶渣",
                "11": "厨余垃圾/菜叶菜根",
                "12": "厨余垃圾/蛋壳",
                "13": "厨余垃圾/鱼骨",
                "14": "可回收物/充电宝",
                "15": "可回收物/包",
                "16": "可回收物/化妆品瓶",
                "17": "可回收物/塑料玩具",
                "18": "可回收物/塑料碗盆",
                "19": "可回收物/塑料衣架",
                "20": "可回收物/快递纸袋",
                "21": "可回收物/插头电线",
                "22": "可回收物/旧衣服",
                "23": "可回收物/易拉罐",
                "24": "可回收物/枕头",
                "25": "可回收物/毛绒玩具",
                "26": "可回收物/洗发水瓶",
                "27": "可回收物/玻璃杯",
                "28": "可回收物/皮鞋",
                "29": "可回收物/砧板",
                "30": "可回收物/纸板箱",
                "31": "可回收物/调料瓶",
                "32": "可回收物/酒瓶",
                "33": "可回收物/金属食品罐",
                "34": "可回收物/锅",
                "35": "可回收物/食用油桶",
                "36": "可回收物/饮料瓶",
                "37": "有害垃圾/干电池",
                "38": "有害垃圾/软膏",
                "39": "有害垃圾/过期药物"
            }

    def build_model(self, model_path, num_classes=40):
        # create model
        model = models.densenet169(num_classes=40)


        modelState = torch.load(model_path, map_location='cpu')
        d = OrderedDict()
        for key, value in modelState.items():
            tmp = key[7:]
            d[tmp] = value
        model.load_state_dict(d)
        return model

    def preprocess_img(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        infer_transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        img = infer_transformation(img)
        return img

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            input_batch = []
            for file_name, file_content in v.items():
                with Image.open(file_content) as image1:
                    image1 = image1.convert("RGB")
                    input_batch.append(self.preprocess_img(image1))
            input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
            preprocessed_data[k] = input_batch_var
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data['input_img']
        pred_score = self.model(img)
        pred_score = F.softmax(pred_score.data, dim=1)
        if pred_score is not None:
            pred_label = torch.argsort(pred_score[0], descending=True)[:1][0].item()
            result = {'result': self.label_id_name_dict[str(pred_label)]}
        else:
            result = {'result': 'predict score is None'}
        return result

    def _postprocess(self, data):
        return data
