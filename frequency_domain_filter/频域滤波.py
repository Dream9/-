# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 14:59:06 2018

@author: yuanz
"""



import pdb
from PIL import Image,ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import skimage
from pylab import mpl  
mpl.rcParams['font.sans-serif'] = ['SimHei']#中文显示问题


#定义距离计算，方便滤波
def cal_distance(pa,pb):
    from math import sqrt
    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
    return dis

#低通滤波器接口
def lowPassFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    #低通滤波器核心实现，  参数：阈值+图像数据
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                dis = cal_distance(center_point,(i,j))
                if dis <= d:
                    transfor_matrix[i,j]=1
                else:
                    transfor_matrix[i,j]=0
        return transfor_matrix
    d_matrix = make_transform_matrix(d)
    #正反傅里叶变换
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img

    
 #高通滤波器接口   
def highPassFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f) 
    #高通滤波器实现，  参数：阈值+图像数据
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                dis = cal_distance(center_point,(i,j))
                if dis <= d:
                    transfor_matrix[i,j]=0
                else:
                    transfor_matrix[i,j]=1
        return transfor_matrix
    d_matrix = make_transform_matrix(d)
    #正反傅里叶变换
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img

#高斯高通滤波接口
def GaussianHighFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
#    pdb.set_trace()
    
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = 1-np.exp(-(dis**2)/(2*(d**2)))
        return transfor_matrix
        
    d_matrix = make_transform_matrix(d)
    #正反傅里叶变换
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img
  
    #高斯低通滤波器接口
def GaussianLowFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = np.exp(-(dis**2)/(2*(d**2)))
        return transfor_matrix
        
    d_matrix = make_transform_matrix(d)
    #正反傅里叶变换
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img

#巴斯特沃滤波器接口
def butterworthPassFilter(image,d,n):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = 1/((1+(d/dis))**n)
        return transfor_matrix
    d_matrix = make_transform_matrix(d)
    #正反傅里叶变换
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img


data=Image.open('E:/实验四/4-1.jpg')
data2=Image.open('E:/实验四/4-2.tif')


img=np.array(data)
img2=np.array(data2)


#使用numpy带的fft库完成从频率域到空间域的转换。
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到0-255
s1 = np.log(np.abs(fshift))
plt.subplot(231),plt.imshow(s1,'gray')
plt.title('4-1Frequency Domain')

#进行傅立叶变换，并显示结果
fft2 = np.fft.fft2(img)
plt.subplot(232),plt.imshow(np.abs(fft2),'gray'),plt.title('fft2')
 
#将图像变换的原点移动到频域矩形的中心，并显示效果
shift2center = np.fft.fftshift(fft2)
plt.subplot(233),plt.imshow(np.abs(shift2center),'gray'),plt.title('shift2center')
 
#对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(235),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')
 
#对中心化后的结果进行对数变换，并显示结果
log_shift2center = np.log(1 + np.abs(shift2center))
plt.subplot(236),plt.imshow(log_shift2center,'gray'),plt.title('log_shift2center')




#低通滤波，分别采用三个阈值
img_d1 = lowPassFilter(img,10)
img_d2 = lowPassFilter(img,30)
img_d3 = lowPassFilter(img,50)
plt.figure(2)
plt.subplot(131)
plt.axis("off")
plt.imshow(img_d1,cmap="gray")
plt.title('4-1距离：10')
plt.subplot(132)
plt.axis("off")
plt.title('4_2距离：30')
plt.imshow(img_d2,cmap="gray")
plt.subplot(133)
plt.axis("off")
plt.title("4-3距离：50")
plt.imshow(img_d3,cmap="gray")



#测试高通滤波器：；；；；；；；
img_d1 = highPassFilter(img,10)
img_d2 = highPassFilter(img,30)
img_d3 = highPassFilter(img,50)
plt.figure(3)
plt.subplot(131)
plt.axis("off")
plt.imshow(img_d1,cmap="gray")
plt.title('4-1距离：10')
plt.subplot(132)
plt.axis("off")
plt.title('4_2距离：30')
plt.imshow(img_d2,cmap="gray")
plt.subplot(133)
plt.axis("off")
plt.title("4-3距离：50")
plt.imshow(img_d3,cmap="gray")



##测试高斯高通滤波
img_d1 = GaussianHighFilter(img,10)
img_d2 = GaussianHighFilter(img,30)
img_d3 = GaussianHighFilter(img,50)
plt.figure(4)
plt.subplot(131)
plt.axis("off")
plt.imshow(img_d1,cmap="gray")
plt.title('4-1距离：10')
plt.subplot(132)
plt.axis("off")
plt.title('4_2距离：30')
plt.imshow(img_d2,cmap="gray")
plt.subplot(133)
plt.axis("off")
plt.title("4-3距离：50")
plt.imshow(img_d3,cmap="gray")

 ##测试高斯低通滤波
img_d1 = GaussianLowFilter(img,10)
img_d2 = GaussianLowFilter(img,30)
img_d3 = GaussianLowFilter(img,50)
plt.figure(5)
plt.subplot(131)
plt.axis("off")
plt.imshow(img_d1,cmap="gray")
plt.title('4-1距离：10')
plt.subplot(132)
plt.axis("off")
plt.title('4_2距离：30')
plt.imshow(img_d2,cmap="gray")
plt.subplot(133)
plt.axis("off")
plt.title("4-3距离：50")
plt.imshow(img_d3,cmap="gray")




#巴斯特沃滤波器测试
plt.figure(6)
plt.subplot(231)
butter_100_1 = butterworthPassFilter(img,100,1)
plt.imshow(butter_100_1,cmap="gray")
plt.title("d=100,n=1")
plt.axis("off")
plt.subplot(232)
butter_100_2 = butterworthPassFilter(img,100,2)
plt.imshow(butter_100_2,cmap="gray")
plt.title("d=100,n=2")
plt.axis("off")
plt.subplot(233)
butter_100_3 = butterworthPassFilter(img,100,3)
plt.imshow(butter_100_3,cmap="gray")
plt.title("d=100,n=3")
plt.axis("off")
plt.subplot(234)
butter_100_1 = butterworthPassFilter(img,30,1)
plt.imshow(butter_100_1,cmap="gray")
plt.title("d=30,n=1")
plt.axis("off")
plt.subplot(235)
butter_100_2 = butterworthPassFilter(img,30,2)
plt.imshow(butter_100_2,cmap="gray")
plt.title("d=30,n=2")
plt.axis("off")
plt.subplot(236)
butter_100_3 = butterworthPassFilter(img,30,3)
plt.imshow(butter_100_3,cmap="gray")
plt.title("d=30,n=3")
plt.axis("off")



#图像增强
#e代表增强系数
def highPassEnhance(image,d,e):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f) 
    e=1-e
#    yreal=fshift.real               # 获取实数部分
#    yimag=fshift.imag               # 获取虚数部分
    #高通滤波器实现，  参数：阈值+图像数据
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                dis = cal_distance(center_point,(i,j))
                if dis <= d:
                    transfor_matrix[i,j]=1
                else:
                    transfor_matrix[i,j]=10
        return transfor_matrix
    d_matrix=make_transform_matrix(d)
    #正反傅里叶变换
    new_img=np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img
    
    
    
    
plt.figure(7)
out42=highPassEnhance(img2,10,0.7)
out43=GaussianHighFilter(img2,2)
plt.imshow(out43,cmap='gray')
plt.title("图像增强结果")




# 高斯滤波
def imgGaussian(sigma):
    '''
    :param sigma: σ标准差
    :return: 高斯滤波器的模板
    '''
    img_h = img_w = 2 * sigma + 1
    gaussian_mat = np.zeros((img_h, img_w))
    for x in range(-sigma, sigma + 1):
        for y in range(-sigma, sigma + 1):
            gaussian_mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    return gaussian_mat
#均值滤波   调用入口
def imgAverageFilter(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:均值滤波后的矩阵
    '''
    return imgConvolve(image, kernel) * (1.0 / kernel.size)
    
    
def imgConvolve(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:卷积后的矩阵
    '''
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_W = int(img_w + 2 * padding_w)

    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_W))
    # 中心填充图片
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    # 卷积结果
    image_convolve = np.zeros(image.shape)
    # 卷积
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1] * kernel))

    return image_convolve
#调用方法：

    
plt.figure(8)
img2_2 = imgAverageFilter(img2, imgGaussian(3))
plt.imshow(img2_2,cmap='gray')
plt.title("图像增强(空域）结果")



plt.show()



