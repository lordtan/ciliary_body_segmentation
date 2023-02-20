#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/1 22:27
# @Author  : lordtan
# 使用现成的模型进行训练

# -*- coding: utf-8 -*-
import time
import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.swa_utils import AveragedModel, SWALR
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
from pytorch_toolbelt import losses as L
from sklearn import metrics
from utils import DataSetUtil
import os

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
n_classes = 4  # 4分类

n_cpu = os.cpu_count()  # CPU个数

def cal_cm(y_true, y_pred):
    y_true = y_true.reshape(1, -1).squeeze()
    y_pred = y_pred.reshape(1, -1).squeeze()
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm


def iou_mean(pred, target, n_classes=n_classes):
    # n_classes ：the number of classes in your dataset,not including background
    # for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    # pred = torch.from_numpy(pred)
    pred = pred.view(-1)
    # print(type(pred))
    target = np.array(target.cpu())
    target = torch.from_numpy(target)
    # print(type(target))
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum / n_classes


def multi_acc(pred, label):
    probs = torch.log_softmax(pred, dim=1)
    _, tags = torch.max(probs, dim=1)
    corrects = torch.eq(tags, label).int()
    acc = corrects.sum() / corrects.numel()
    return acc


def train(EPOCHES, BATCH_SIZE, path, channels, optimizer_name,
          model_path, loss, early_stop):

    '''
    train_dataset = ImageFolder(data_root, mode='train')
    val_dataset = ImageFolder(data_root, mode='val')

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0)
    '''

    # 加载测试与验证数据
    # path = r"eyedata/train/"
    params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 4}
    train_data_loader, val_data_loader = DataSetUtil.make_dataloaders(path, 0.2,  params)


    # 定义模型,优化器,损失函数
    # model = smp.UnetPlusPlus(
    #         encoder_name="efficientnet-b7",
    #         encoder_weights="imagenet",
    #         in_channels=channels,
    #         classes=17,
    # )
    # model = smp.UnetPlusPlus(
    #         encoder_name="timm-resnest101e",
    #         encoder_weights="imagenet",
    #         in_channels=channels,
    #         classes=2,
    # )

    model = smp.Unet(
        encoder_name="resnext50_32x4d",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7, resnet34
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=n_classes,  # model output channels (number of classes in your dataset)
        activation='softmax',  # 二分类需要换成sigmoid
    )

    model.to(DEVICE)
    # 加载预模型可以打开下面这句，model_path给预模型路径
    # model.load_state_dict(torch.load(model_path))
    if (optimizer_name == "sgd"):
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=1e-4, weight_decay=1e-3, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=1e-3, weight_decay=1e-3)
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2,  # T_0就是初始restart的epoch数目
        T_mult=2,  # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
        eta_min=1e-5  # 最低学习率
    )

    if (loss == "SoftCE_dice"):  # mode: Loss mode 'binary', 'multiclass' or 'multilabel'
        # 损失函数采用SoftCrossEntropyLoss+DiceLoss
        # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
        # DiceLoss_fn = DiceLoss(mode='binary')
        DiceLoss_fn = DiceLoss(mode='multiclass')  # 多分类改为multiclass
        # Bceloss_fn = nn.BCELoss()
        # 软交叉熵,即使用了标签平滑的交叉熵,会增加泛化性
        SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)  # 用于多分类
        loss_fn = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn, first_weight=0.8, second_weight=0.2).cuda()
    # loss_fn = smp.utils.losses.DiceLoss()
    else:
        # 损失函数采用SoftCrossEntropyLoss+LovaszLoss
        # LovaszLoss是对基于子模块损失凸Lovasz扩展的mIoU损失的直接优化
        # LovaszLoss_fn = LovaszLoss(mode='binary')
        LovaszLoss_fn = LovaszLoss(mode='multiclass')
        # 软交叉熵,即使用了标签平滑的交叉熵,会增加泛化性
        SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)  # 这里我没有改，这里是多分类的，有需求就改下
        loss_fn = L.JointLoss(first=LovaszLoss_fn, second=SoftCrossEntropy_fn,
                              first_weight=0.5, second_weight=0.5).cuda()

    best_miou = 0
    best_miou_epoch = 0
    train_loss_epochs, val_mIoU_epochs, lr_epochs = [], [], []
    for epoch in range(1, EPOCHES + 1):
        losses = []
        start_time = time.time()
        model.train()

        print(f"epoche: {epoch} 开始训练")
        for i, sample in tqdm(enumerate(train_data_loader), ncols=20, total=len(train_data_loader)):
            # print("----------", i)
            # image, target = sample["image"].to(DEVICE, dtype=torch.float32), sample["mask"].to(DEVICE)
            image, target = sample[0].to(DEVICE, dtype=torch.float32), sample[1].to(DEVICE)

            # image, target = image.to(device=DEVICE, dtype=torch.float32), target.to(DEVICE)
            # print(image.shape)
            output = model(image)
            target = torch.tensor(target, dtype=torch.int64)
            loss = loss_fn(output, target)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoche: {epoch} 训练结束")

        '''
        val_data_loader_num = iter(val_data_loader).next
        for image, target in tqdm(val_data_loader_num, ncols=20, total=len(val_data_loader_num)):
            # image, target = image.to(device=DEVICE, dtype=torch.float32), target.to(DEVICE)
            image, target = image.to(DEVICE), target.to(DEVICE)
            output = model(image)
            target = torch.tensor(target, dtype=torch.int64)
            loss = loss_fn(output, target)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
'''

        scheduler.step()

        val_acc = []
        val_iou = []

        '''
        val_data_loader_num = iter(val_data_loader)
        for val_img, val_mask in tqdm(val_data_loader_num, ncols=20, total=len(val_data_loader_num)):
            val_img, label = val_img.to(DEVICE), val_mask.to(DEVICE)
            predict = model(val_img)
            label = label.squeeze(1)

            acc = multi_acc(predict, label)
            val_acc.append(acc.item())

            predict = torch.argmax(predict, axis=1)
            iou = iou_mean(predict, label, n_classes)
            val_iou.append(iou)
        '''
        print(f"epoche: {epoch} 开始验证")
        for i, sample in tqdm(enumerate(val_data_loader), ncols=20, total=len(val_data_loader)):
            # val_img, label = sample["image"].to(DEVICE, dtype=torch.float32), sample["mask"].to(DEVICE)
            val_img, label = sample[0].to(DEVICE, dtype=torch.float32), sample[1].to(DEVICE)

            predict = model(val_img)
            # label = label.squeeze(1)  # 这个地方感觉不需要压缩

            acc = multi_acc(predict, label)
            val_acc.append(acc.item())

            predict = torch.argmax(predict, axis=1)
            iou = iou_mean(predict, label, n_classes)
            val_iou.append(iou)
        print(f"epoche: {epoch} 验证结束")

        train_loss_epochs.append(np.array(losses).mean())
        val_mIoU_epochs.append(np.mean(val_iou))
        lr_epochs.append(optimizer.param_groups[0]['lr'])

        print('Epoch:' + str(epoch) + ' Loss:' + str(np.array(losses).mean()) + ' Val_Acc:' + str(
            np.array(val_acc).mean()) + ' Val_IOU:' + str(np.mean(val_iou)) + ' Time_use:' + str(
            (time.time() - start_time) / 60.0))

        if best_miou < np.stack(val_iou).mean(0).mean():
            best_miou = np.stack(val_iou).mean(0).mean()
            best_miou_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("  valid mIoU is improved. the model is saved.")
        else:
            print("")
            if (epoch - best_miou_epoch) >= early_stop:
                break

    return train_loss_epochs, val_mIoU_epochs, lr_epochs


if __name__ == '__main__':
    EPOCHES = 100
    BATCH_SIZE = 4
    loss = "SoftCE_dice"
    # loss = "SoftCE_Lovasz"
    channels = 3
    optimizer_name = "adamw"

    # data_root = "CamVid/"
    # path = r"dataeye/train/"
    # path = r"dataeye/train/"
    path = r"D:\code\python\ciliary_body_segmentation\dataeye\train"
    model_path = "D:\code\python\ciliary_body_segmentation\weights\eye_" + loss + '.pth'

    early_stop = 400
    train_loss_epochs, val_mIoU_epochs, lr_epochs = train(EPOCHES, BATCH_SIZE, path, channels, optimizer_name,
                                                          model_path, loss, early_stop)

    if (True):
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_loss_epochs) + 1)
        plt.plot(epochs, train_loss_epochs, 'r', label='train loss')
        plt.plot(epochs, val_mIoU_epochs, 'b', label='val mIoU')
        plt.title('train loss and val mIoU')
        plt.legend()
        plt.savefig("train loss and val mIoU.png", dpi=300)
        plt.figure()
        plt.plot(epochs, lr_epochs, 'r', label='learning rate')
        plt.title('learning rate')
        plt.legend()
        plt.savefig("learning rate.png", dpi=300)
        plt.show()
