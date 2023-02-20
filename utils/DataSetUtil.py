#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lordtan
@Date: 2023/2/19 22:41
@Desc: 
'''
from utils.EyeDataSet import EyeDataSet
import torch
from torch.utils.data import random_split

'''
path：数据目录，其子目录必定为 images 和 masks
rate： 测试数据占比
'''
def make_dataloaders(path, rate, params):

    batch_size = 20  #默认20
    if params.__contains__("batch_size"):
        batch_size = params["batch_size"]

    eye_dataset = EyeDataSet(path)
    length = len(eye_dataset)
    vl = int(length * rate)
    tl = length - vl

    train_dataset, val_dataset = random_split(dataset=eye_dataset, lengths=[tl, vl])

    print("数据总数：", length, "其中训练数据：", tl, "测试数据:", vl)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    return train_data_loader, val_data_loader

if __name__ == "__main__":
    # data_path = "dataeye/train"
    data_path = r"D:\code\python\ciliary_body_segmentation\dataeye\train"
    train_data_loader, val_data_loader = make_dataloaders(data_path, 0.2, {})
    print(len(train_data_loader), len(val_data_loader))
