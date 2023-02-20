#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/6 14:13
# @Author  : lordtan
# 读取通过label生成的原始图片和label文件

import torch
import cv2
import os
from os.path import isfile, isdir, join
import glob
from torch.utils.data import Dataset
from torch.utils.data import Dataset as BaseDataset
from PIL import ExifTags, Image, ImageOps
import torchvision
import numpy as np
from torchvision.transforms import PILToTensor
from torchvision import transforms

class EyeDataSet(BaseDataset):

    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        imgpath = os.path.join(data_path, "images")
        labpath = os.path.join(data_path, "masks")

        imgs_path = []
        labels_path = []
        for file in os.listdir(imgpath):
            imgs_path.append(os.path.join(imgpath, file))

        for file in os.listdir(labpath):
            labels_path.append(os.path.join(labpath, file))


        self.imgs_path = imgs_path
        self.labels_path = labels_path


    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = self.labels_path[index]

        # image = cv2.imread(image_path, cv2.IMREAD_BGR)
        # label = cv2.imread(label_path)

        '''
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(label_path, 0)
        '''

        image = Image.open(image_path)
        label = Image.open(label_path)

        # 数据类型转换为tensor
        image_tensor, label_tensor = PILToTensor()(image), PILToTensor()(label)

        # 需要对其resize
        resize = transforms.Resize([672, 1024])
        image_tensor = resize(image_tensor)  # size torch.Size([3, 672, 1024])
        label_tensor = resize(label_tensor)

        return image_tensor, label_tensor

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":

    data_path = path = r"D:\code\python\ciliary_body_segmentation\dataeye\train"

    eye_dataset = EyeDataSet(data_path)
    print("数据个数：", len(eye_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=eye_dataset,
                                               batch_size=20,
                                               shuffle=True)
    print(train_loader.batch_size)

