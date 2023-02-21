#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/3 18:19
# @Author  : lordtan


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import cv2
import os,sys
import scipy.ndimage
import time
import scipy
from labelme import utils




'''
红色区域,HSV值范围
'''
lower_red = np.array([250, 0, 0])
upper_red = np.array([255, 0, 0])

'''
绿色区域,HSV值范围
'''
lower_green = np.array([0, 250, 0])
upper_green = np.array([0, 255, 0])

'''
蓝色区域,HSV值范围
'''
lower_blue = np.array([0, 0, 250])
upper_blue = np.array([0, 0, 255])



# 读取图片
def getMasks(img):
    # frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    red = cv2.inRange(frame, lower_red, upper_red)
    green = cv2.inRange(frame, lower_green, upper_green)
    blue = cv2.inRange(frame, lower_blue, upper_blue)

    return red, green, blue

# 测量多边形的相关参数
# 测量多边形的相关参数
def measureLabelPolygon(image, label):
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    # print("threshold value: %s" % ret)
    # cv2.imshow("binary_image", binary)

    dst = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # dst = cv2.cvtColor(binary, cv2.COLOR_BGR2RGB)

    # 获取轮廓
    contours, hireachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[0]  # 面积最大的

    params = []
    for i, contour in enumerate(contours):
        # print("----------这是第%s个形状" % i)
        area = cv2.contourArea(contour)  # 面积

        # 除去一些手误打出的点
        if area < 8:
            continue

        approxCurve = cv2.approxPolyDP(contour, 3, True)

        # 如果是多边形
        if approxCurve.shape[0] > 3:
            # 绘制绿色边框
            # cv2.drawContours(dst, contours, i, (0, 255, 0), 2)  # 绘制绿色边框

            # 绘制红色外接矩形框
            x, y, w, h = cv2.boundingRect(contour)  # 外接矩形

            rate = min(w, h) / max(w, h)  # 曲折率
            # print("曲折率 rectangle rate is %s" % rate)  # 曲折率
            # print("AREA IS %s" % area)  # 面积
            mm = cv2.moments(contour)  # 几何矩
            cx = mm['m10'] / mm['m00']  # 中心x
            cy = mm['m01'] / mm['m00']  # 中心y

            # print(x, y, w, h)
            # x1 = x + w
            # y1 = y + h
            # cv2.rectangle(dst, (x, y), (x1, y1), (0, 0, 255), 1)  # 绘制红色矩形框

            param = {"area": area, "width": w, "height": h, "rate": rate, "label": label}
            params.append(param)

    return params


# 读取图像参数
if __name__ == '__main__':

    # img = cv2.imread(r"label410.png")
    # img = cv2.imread(imagepath)

    # 中文处理
    imagepath = "predict.png"
    # imagepath = r"D:\code\python\ciliary_body_segmentation\dataeye\train\masks\丁雪娇标注-右眼术前-丁雪娇右眼2-8大图-label.png"

    img = cv2.imdecode(np.fromfile(imagepath, dtype=np.uint8), -1)
    left, right, top = getMasks(img)

    # measure_object(left)


    leftrel = measureLabelPolygon(left, "left")
    print(leftrel)

    rightrel = measureLabelPolygon(right, "right")
    print(rightrel)

    toprel = measureLabelPolygon(top, "top")
    print(toprel)


    # measure_object(right)
    # measure_object(top)

    # cv2.imshow("mask", top)
    # cv2.imshow("res_blue",res_red)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


