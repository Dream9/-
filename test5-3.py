import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

_A = 180
_C= 0.5
_W =540
_STEP=1
# 当c=0.5时，该函数称为汉宁窗（RichardHamming）
# 当c=0.54时，称为汉明窗（Juliusvon Hann

#定义汉宁函数
def windowFuc(S,M=_W,c=_C):
    hw=[]
    for x in range(0,M):
        hw.append(float((c-1)*np.cos(2*np.pi*x/(M-1))+c))###Hamming函数*R-L函数
    med=[0]*int((S-M)/2)
    hw=med+hw+med
    return hw

#矩形窗
def window_Rec(S,M):
    hw = []
    for x in range(0, M):
        hw.append(1)  ###矩形
    med = [0] * int((S - M)/2)
    hw = med+hw+med
    return hw

#汉明窗
def window_Hamning(S,M,c=0.54):
    h1=window_Rec(S,M)
    h2=windowFuc(S,M,c)
    hw=h1*h2


#一维窗函数滤波
def Filter(img,window):
    img=img*window
    return img
    f = np.fft.fft(img)
    # f=f*window
    # new_img=np.real(np.fft.ifft(f))

    # 移频
    fshift = (np.fft.fftshift(f))

    fshift=fshift*window
    new_img = np.real(np.fft.ifft(np.fft.ifftshift(fshift)))

    # _figure(new_img, 'fs')
    return new_img

#理想高通滤波
def _highPassFilter(image, d):
    f = np.fft.fft(image)
    # 移频
    fshift = (np.fft.fftshift(f))
    new = fshift
    new[len(fshift)//2 - d:len(fshift)//2 + d] = 0  ###理想高通
    # ceshi
    # print(fshift)
    # _figure(fshift,'wer')
    # 反傅里叶变换
    new_img = np.real(np.fft.ifft(np.fft.ifftshift(new)))
    return new_img

#RL滤波   ok
def _high_RL_Filter(image, d):
    f = np.fft.fft(image)
    # 移频
    fshift = (np.fft.fftshift(f))
    #ceshi
    # for d,i in enumerate(fshift):
    #     if np.abs(i)>d:
    #         print(d)
    new =np.where(np.abs(fshift)<d,fshift,0)
    # new[0:len(fshift) - d] = 0  ###RL滤波
    # 反傅里叶变换
    new_img = np.real(np.fft.ifft(np.fft.ifftshift(new)))
    # _figure(new_img,'RL')
    return new_img


# radon变换
def radon_transform(image):
    rows, cols = image.shape
    angles = range(0, _A, _STEP)
    height = len(angles)
    width = cols
    sinogram = np.zeros((height, width))
    for index, alpha in enumerate(angles):
        # 得到变换矩阵
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), alpha, 1)
        # 进行仿射变换
        rotated = cv2.warpAffine(image, M, (cols, rows))
        # 投影求和
        sinogram[index] = rotated.sum(axis=0)  # 按列求和
    return sinogram


# Back变换
def back_project(sinogram):
    # 单步转动角度，默认范围是0-180
    rotation_angle = _A / len(sinogram)

    # 输出图像大小（都等于原始列数）
    width = height = len(sinogram[0])
    reconstructed = np.zeros((width, height))
    wd=window_Rec(600,400)
    wd = windowFuc(600)
    for index, projection in enumerate(sinogram):
        print(index, time.strftime("%H:%M:%S", time.localtime()))
        # 当前仿射变换6参数（反向）
        M = cv2.getRotationMatrix2D(
            (width / 2, height / 2), -rotation_angle * index, 1)
        # 减小数值
        scaled_projection = projection / height
        ##进行一维傅里叶变换并滤波
        #理想
        # new_projection = _highPassFilter(scaled_projection, 30)
        #RL  矩形窗
        new_projection = _high_RL_Filter(scaled_projection, 1000)####2500 大概对应于+-30位置
        #汉明窗
        # new_projection =scaled_projection
        # new_projection =Filter(scaled_projection,wd)
        # 当前的radon赋值
        back_projected = np.zeros((width, height))
        for row in back_projected:
            row += new_projection  # scaled_projection
        # 逆变换回去并加到结果图像上
        reconstructed += cv2.warpAffine(back_projected, M, (width, height))

    return reconstructed
    #####有光晕


def _Affine(image, M, range_list):
    image_c = np.copy(image)

    pass


# 画图封装函数
def _figure(img, title, type="gray", colorbar=False):
    plt.figure()
    plt.title(title)
    try:
        if (len(img.shape) > 1):
            plt.imshow(img, cmap=type)
        else:
            plt.plot(img)
    except:
        plt.plot(img)
        print("Something wrong has been ignored!")
    if (colorbar):
        plt.colorbar()

    # plt.savefig("E:/test5/"+title+'.png')


if (__name__ == '__main__'):
    img = cv2.imread('E:/test5/5-4.tif')
    img = img[:, :, 0:1].reshape(img.shape[0], img.shape[1])
    _figure(img, 'Original')

    # wd = windowFuc(600)
    # for i in wd:
    #     print(i)

    R = radon_transform(img)

    # # 部分成图
    # for i, r in enumerate(R):
    #     _figure(r, 'Radon:{0}'.format(i))
    #     plt.savefig("E:/test5/" + 'Radon:{0}'.format(i)+ '.png')


    # for


    _figure(R, '180° Radon', type="GnBu", colorbar=True)

    # 反变换
    reconstruct = back_project(R)
    _figure(reconstruct, "Reconstruct(Back_Transform)")

    # 动画演示
    # def anomation(R,title,frame=18):
    size = len(R)
    print(size)
    step = int(size / 18)
    # 创建画布
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    x = np.arange(0, 600, 1)
    line, = ax.plot(x, x + 6, 'r-', linewidth=2)
    plt.xlim(0, 600)
    plt.ylim(0, 40000)


    def update(i):
        label = 'Radon step {0}'.format(i)
        line.set_ydata(R[(i)])  # 更新y轴的数据
        # print(R[(i)%size])
        ax.set_xlabel(label)  # 更新x轴的标签
        return line, ax


    # FuncAnimation 会在每一帧都调用“update” 函数。
    # 在这里设置一个18帧的动画，每帧之间间隔200毫秒
    anim = FuncAnimation(fig, update, frames=np.arange(0, 18), interval=200)

    plt.show()
