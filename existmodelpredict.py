#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lordtan
@Date: 2023/2/5 21:37
@Desc: 
'''

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import PILToTensor
import segmentation_models_pytorch as smp
import torch
import numpy as np
from labelme import utils



if __name__ == '__main__':

    modelpath = "weights/eye_SoftCE_dice.pth"
    # imagepath = "img.png"
    imagepath = r"img.png"

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # device = 'cpu'  # 模拟没有GPU的情况

    net = smp.Unet(
        encoder_name="resnext50_32x4d",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7, resnet34
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4,  # model output channels (number of classes in your dataset)
        activation='softmax',  # 二分类需要换成sigmoid
    )


    # 加载模型
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load(modelpath, map_location=device))

    '''
    val_img, label = sample["image"].to(DEVICE, dtype=torch.float32), sample["mask"].to(DEVICE)
    因为在训练时，图片被加载到了GPU种，所以会报如下错误：
    RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
    '''
    # net.to(device='cpu') # 强制加载到cpu试试

    # 测试模式
    net.eval()

    # 加载图片，resize
    image = Image.open(imagepath)
    image_tensor = PILToTensor()(image)

    resize = transforms.Resize([672, 1024])
    image_tensor = resize(image_tensor) # size torch.Size([3, 672, 1024])

    image_tensor = image_tensor.unsqueeze(0) # 增加一个维度，由[3, 672, 1024] 变为[4, 3, 672, 1024]

    # image_tensor = torch.tensor(image_tensor,  dtype=torch.float32)
    image_tensor = image_tensor.to(device, dtype=torch.float32)

    # 预测结果
    pred = net(image_tensor)

    predict = torch.argmax(pred, axis=1)
    resize1 = transforms.Resize([655, 1024]) # resize为实际大小
    predict = resize1(predict)

    # pred = torch.sequeeze(pred) # 降1维

    # target = torch.tensor(target, dtype=torch.int64)

    #TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    tensorarray = predict.cpu().numpy()  # tensor转换为numpy,并降维 flatten()
    low = np.squeeze(tensorarray)  # 使用squeeze降维
    utils.lblsave("predict.png", low)
