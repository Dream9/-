import cv2
import numpy as np
import matplotlib.pyplot as plt



def ban(img,mask):
    f1 = np.fft.fft2(img)
    f1shift = np.fft.fftshift(f1)
    f1shift = f1shift*mask
    f2shift = np.fft.ifftshift(f1shift) #对新的进行逆变换
    img_new = np.real(np.fft.ifft2(f2shift))
    return img_new



img=cv2.imread('E:/test5/5-1.tif',cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('E:/test5/5-2.tif',cv2.IMREAD_GRAYSCALE)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# #取绝对值：将复数变化成实数
# #取对数的目的为了将数据变化到较小的范围（比如0-255）
# s1 = np.log(np.abs(f))
# s2 = np.log(np.abs(fshift))
# print(np.shape(s1))
# print(s1[0:20,0:20])
# cv2.imshow('s1',np.array(s1,dtype=int))
# cv2.imshow('s2',s2)
# cv2.waitKey()
# plt.subplot(321),plt.imshow(s1,'gray'),plt.title('original')
# plt.subplot(322),plt.imshow(s2,'gray'),plt.title('center')
# ph_f = np.angle(f)
# ph_fshift = np.angle(fshift)
# # print(ph_f)
# # print(ph_fshift)
# plt.subplot(323),plt.imshow(ph_f,'gray'),plt.title('original')
# plt.subplot(324),plt.imshow(ph_fshift,'gray'),plt.title('center')
#
# # 逆变换
# f1shift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f1shift)
# # 出来的是复数，无法显示
# img_back = np.abs(img_back)
# plt.subplot(325), plt.imshow(img_back, 'gray'), plt.title('img back')
# plt.show()

plt.subplot(121)
plt.imshow(img,'gray'),plt.title('origial')
plt.xticks([]),plt.yticks([])
#--------------------------------
rows,cols = img.shape
# mask = np.ones(img.shape,np.uint8)
# mask[rows/2-30:rows/2+30,cols/2-30:cols/2+30] = 0 #高通滤波
# mask = np.zeros(img.shape,np.uint8)
# mask[rows/2-80:rows/2+80,cols/2-80:cols/2+80] = 1 #低通滤波
#--------------------------------
#--------------------------------理想的带通滤波器
rows,cols = img.shape
print(rows,cols)
inner=10
outer=100
mask1 = np.ones(img.shape,np.uint8)
mask1[int(rows/2-inner):int(rows/2+inner),int(cols/2-inner):int(cols/2+inner)] = 0
mask2 = np.zeros(img.shape,np.uint8)
mask2[int(rows/2-outer):int(rows/2+outer),int(cols/2-outer):int(cols/2+outer)] = 1
mask = mask1*mask2
#--------------------------------
mask=np.ones(mask.shape)-mask
#出来的是复数，无法显示
img_new = ban(img,mask)

plt.subplot(122),plt.imshow(img_new,'gray'),plt.title('1')
plt.xticks([]),plt.yticks([])
plt.show()
