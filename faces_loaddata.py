import os
from PIL import Image
import numpy as np
import cv2

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，
#如果是将彩色图作为输入,则将1替换为3，
# 并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
IMAGE_SIZE=64
#按照指定图像大小调整尺寸
def resize_image(image,height=IMAGE_SIZE,width=IMAGE_SIZE):
    top,bottom,left,right=(0,0,0,0)
    #获取图像尺寸
    h,w,_=image.shape
    #对于长宽不相等的图片，找到最长的一边
    longest_edge=max(h,w)
    #计算短边需要增加多少像素使其与长边等长
    if h<longest_edge:
        dh=longest_edge-h
        top=dh//2
        bottom=dh-top
    elif w<longest_edge:
        dw=longest_edge-w
        left=dw//2
        right=dw-right
    else:
        pass
    #RGB颜色
    BLACK=[0,0,0]
    #给图像增加边界，使图像长宽相等
    constant=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)
    #调整图像大小并返回
    return cv2.resize(constant,(height,width))
def load_data():
    # num=4200
    data=np.empty((1899,3,64,64),dtype="float32")
    label=np.empty((1899,),dtype='uint8')
    imgs=os.listdir("E:/keras_data/face6")
    num=len(imgs)
    for i in range(num):
        # img=Image.open("E:/keras_data/face6/"+imgs[i])
        img=cv2.imread("E:/keras_data/face6/"+imgs[i])
        img=resize_image(img,IMAGE_SIZE,IMAGE_SIZE)
        arr=np.asarray(img,dtype="float32")
        # data[i,:,:,:]=arr
        data[i, :, :, :] = [arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]
        label[i]=int(imgs[i].split('.')[0])
    data = data.reshape(1899, 64, 64, 3)
    return data,label