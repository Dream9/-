# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:54:38 2018

@author: yuanz
"""

import pdb
from PIL import Image,ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import skimage
import cv2
#from skimage.morphology import disk

img=Image.open('E:/实验三/图3-1.tif')
img2=Image.open('E:/实验三/图3-2.jpg')
img3=Image.open('E:/实验三/图3-3.tif')

arr=np.array(img)
arr_med=np.array(img)
#随机生成5000个椒盐
rows,cols=arr_med.shape
for i in range(100000):
    x=np.random.randint(0,rows)
    y=np.random.randint(0,cols)
    arr_med[x][y]=255
    
#scipy模拟中值滤波
#arr_out=signal.medfilt(arr_med,3)###3窗口
arr_out=np.array(img.filter(ImageFilter.MedianFilter(3)))

plt.figure("添加噪声及滤波",figsize=(8,8))
plt.subplot(131)
plt.imshow(img,plt.cm.gray)
plt.subplot(132)
plt.imshow(arr_med,plt.cm.gray)
plt.subplot(133)
plt.imshow(arr_out,plt.cm.gray)



#==============================================================================
# 2
#==============================================================================

#中值滤波
def med_filter(x,y,step,im):
    #对im图像第x,y位置的像素按照step大小的窗口大小中值滤波
    sum_s=[]
    for k in range(-int(step/2),int(step/2)+1):
        for m in range(-int(step/2),int(step/2)+1):
            sum_s.append(im[x+k][y+m])
    sum_s.sort()
    return sum_s[(int(step*step/2)+1)]#返回中间值
#均值滤波
def mean_filter(x,y,step,im):
    #对im图像第x,y位置的像素按照step大小的窗口大小均值滤波
    sum_s = 0
    for k in range(-int(step/2),int(step/2)+1):
        for m in range(-int(step/2),int(step/2)+1):
            sum_s += im[x+k][y+m] / (step*step)
    return sum_s#返回平均值
#统计差异
def std(first,second):
#    pdb.set_trace()
    ret=0.0
#    for i in range(first.shape[0]):
#        for j in range(first.shape[1]):
#            ret+=(first[i][j]-second[i][j])*(first[i][j]-second[i][j])
    #以下时不考虑边缘时的统计        
    print(1,first.shape[0]-1)#测试
    for i in range(1,first.shape[0]-1):
        for j in range(1,first.shape[1]-1):
            ret+=(first[i][j]-second[i][j])*(first[i][j]-second[i][j])
    return ret/(first.shape[0]*first.shape[1])
#对数据进行处理
med_out=np.copy(arr_med)
mean_out=np.copy(arr_med)#防止引用拷贝
medStep=3#窗口布长为3
#边缘的不处理
for i in range(int(medStep/2),med_out.shape[0]-int(medStep/2)):
    for j in range(int(medStep/2),med_out.shape[1]-int(medStep/2)):
        med_out[i][j]=med_filter(i,j,medStep,arr_med)


meaStep=3#窗口布长为3
for i in range(int(meaStep/2),mean_out.shape[0]-int(meaStep/2)):
    for j in range(int(meaStep/2),mean_out.shape[1]-int(meaStep/2)):
        mean_out[i][j]=mean_filter(i,j,meaStep,arr_med)

plt.figure("DIY滤波",figsize=(8,8))
plt.subplot(131)
plt.imshow(arr_med,plt.cm.gray)
plt.subplot(132)
plt.imshow(med_out,plt.cm.gray)
plt.subplot(133)
plt.imshow(mean_out,plt.cm.gray)
##比较两种差异
print("中值对椒盐处理与原始图像的std：",std(med_out,arr))
print("均值对椒盐处理与原始图像的std：",std(mean_out,arr))
print("库函数中值对椒盐处理与原始图像的std：",std(arr_out,arr))


#==============================================================================
#3
#==============================================================================

# # sobel算子的实现
def sobel(arr):
    r,c= arr.shape
    new_image=np.zeros((r,c))
    new_imageX=np.zeros(arr.shape)
    new_imageY=np.zeros(arr.shape)
    s_suanziX=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])# X方向
    s_suanziY=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])     
    for i in range(r-2):
        for j in range(c-2):
            new_imageX[i+1, j+1] = abs(np.sum(arr[i:i+3, j:j+3] * s_suanziX))
            new_imageY[i+1, j+1] = abs(np.sum(arr[i:i+3, j:j+3] * s_suanziY))
            new_image[i+1, j+1] = (new_imageX[i+1, j+1]*new_imageX[i+1,j+1] + new_imageY[i+1, j+1]*new_imageY[i+1,j+1])**0.5
    
    return np.uint8(new_imageX),np.uint8(new_imageY),np.uint8(new_image)  # X,Y无方向算子处理的图像
 
# Laplace算子
# 常用的Laplace算子模板  [[0,1,0],[1,-4,1],[0,1,0]]   [[1,1,1],[1,-8,1],[1,1,1]]
def laplace(arr):
    r,c=arr.shape
    new_image=np.zeros((r, c))
    L_sunnzi=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])     
    # L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])      
    for i in range(r-2):
        for j in range(c-2):
            new_image[i+1, j+1] = abs(np.sum(arr[i:i+3, j:j+3] * L_sunnzi))
    return np.uint8(new_image)
 
arr2=np.array(img2)
arr3=np.array(img3)

arr2_sobelX,arr2_sobelY,arr2_sobel=sobel(arr2)
arr3_laplace=laplace(arr3)

plt.figure("Sobel",figsize=(8,8))
plt.subplot(221)
plt.imshow(arr2,plt.cm.gray)
plt.subplot(222)
plt.imshow(arr2_sobelX,plt.cm.gray)
plt.subplot(223)
plt.imshow(arr2_sobelY,plt.cm.gray)
plt.subplot(224)
plt.imshow(arr2_sobel,plt.cm.gray)


plt.figure("Laplace",figsize=(8,8))
plt.subplot(121)
plt.imshow(arr3,plt.cm.gray)
plt.subplot(122)
plt.imshow(arr3_laplace,plt.cm.gray)



#==============================================================================
# 
#==============================================================================

# sobel算子 Y 方向
#suanzi2X=np.array([[-1,-2,-1],
#                    [0,0,0],
#                    [1,2,1]])
## sobel算子 X 方向
#suanzi2Y=np.array([[-1,0,1],
#                    [-2,0,2],
#                    [-1,0,1]])
## laplace算子
#suanzi3=np.array([[0,1,0],  
#                    [1,-4,1],
#                    [0,1,0]])

# 利用signal的convolve计算卷积
#edges_sobel= skimage.filter.sobel(img)
#
#suanzi3=skimage.filter.laplace(img)
#
#image_suanzi2X=signal.convolve2d(arr2,suanzi2X,mode="same")
#image_suanzi2Y=signal.convolve2d(arr2,suanzi2Y,mode="same")
image_suanzi2X=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
image_suanzi2Y=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
image_suanzi3=cv2.Laplacian(img3,cv2.CV_64F)
#image_suanzi3=signal.convolve2d(arr3,suanzi3,mode="same")

plt.figure("库Sobel",figsize=(8,8))
plt.subplot(131)
plt.imshow(arr2,plt.cm.gray)
plt.subplot(132)
plt.imshow(image_suanzi2X,plt.cm.gray)
plt.subplot(133)
plt.imshow(image_suanzi2Y,plt.cm.gray)


plt.figure("库Laplace",figsize=(8,8))
plt.subplot(121)
plt.imshow(arr3,plt.cm.gray)
plt.subplot(122)
plt.imshow(image_suanzi3,plt.cm.gray)

print("sobel与库函数对比std：",std(image_suanzi2X,arr2_sobelX))
print("sobelY方向与库函数对比std：",std(image_suanzi2Y,arr2_sobelY))
print("laplace与库函数对比std：",std(image_suanzi3,arr3_laplace))




plt.figure("DIY比较",figsize=(8,8))
plt.subplot(131)
ttt=med_out-mean_out
plt.imshow(ttt,plt.cm.gray)
plt.subplot(132)
kkk=med_out-arr_out
plt.imshow(kkk,plt.cm.gray)
plt.subplot(133)
jjj=image_suanzi3-arr3_laplace
plt.imshow(jjj,plt.cm.gray)




plt.show()



