import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

x=[0,1,2]
y=[2,1,0]

plt.figure()
angle=np.linspace(0,np.pi*2,20)
for i in range(3):
    r=x[i]*np.cos(angle)+y[i]*np.sin(angle)
    plt.subplot(220+i+1)
    plt.plot(angle,r)
    try:
        plt.title('过定点 ({0}，{1})'.format(x[i],y[i]))
    except:
        print(i)
        exit(-1)
plt.subplot(224)
for i in range(3):
    r=x[i]*np.cos(angle)+y[i]*np.sin(angle)
    plt.plot(angle,r)
plt.figure()
for i in range(1):
    r=x[i]*np.cos(angle)+y[i]*np.sin(angle)
    plt.plot(angle, r)


'''
function [ Hough, theta_range, rho_range ] = naiveHough(I)
%NAIVEHOUGH Peforms the Hough transform in a straightforward way.
%
[rows, cols] = size(I);
 
theta_maximum = 90;
rho_maximum = floor(sqrt(rows^2 + cols^2)) - 1;
theta_range = -theta_maximum:theta_maximum - 1;
rho_range = -rho_maximum:rho_maximum;
 
Hough = zeros(length(rho_range), length(theta_range));
for row = 1:rows
    for col = 1:cols
        if I(row, col) > 0 %only find: pixel > 0
            x = col - 1;
            y = row - 1;
            for theta = theta_range
                rho = round((x * cosd(theta)) + (y * sind(theta)));  %approximate
                rho_index = rho + rho_maximum + 1;
                theta_index = theta + theta_maximum + 1;
                Hough(rho_index, theta_index) = Hough(rho_index, theta_index) + 1;
            end
        end
    end
end

'''
def hough(img):
    rows, cols = img.shape

    theta_maximum = 90;
    rho_maximum = int(np.floor(np.sqrt(rows**2 + cols**2))) - 1;
    theta_range = np.arange(-theta_maximum,theta_maximum - 1,1)
    rho_range = np.arange(-rho_maximum,rho_maximum,1)

    Hough = np.zeros((rho_range.shape[0], theta_range.shape[0]))#初始化

    for row in range(rows):
        for col in range(cols):
            if img[row][col] > 0 : #####已经与处理好了
                x = col - 1;
                y = row - 1;
                for theta in theta_range:
                    rho = round((x * np.cos(theta)) + (y * np.sin(theta)))####根据空间划分，累加结果
                    rho_index = int(rho + rho_maximum )
                    theta_index =int( theta + theta_maximum)
                    Hough[rho_index][theta_index] = Hough[rho_index][theta_index] + 1
    return Hough


plt.figure()
path='E:/text6/ceshi.tif'
img = np.array(cv2.imread(path,0))
# img=np.reshape(img,(img.shape[0],img.shape[1]))
img=cv2.Canny(img,50,100)
plt.imshow(img)



'''Hough图'''
plt.figure()
h=hough(img)
plt.imshow(h,cmap='Reds')
plt.title('Hough变换')



'''直方图均衡化'''
plt.figure()
# gray = cv2.cvtColor(h, cv2.COLOR_BGR2GRAY)
ymax = 255
ymin=0
xmax=np.max(h)
xmin=np.min(h)

for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        h[i][j] = round((ymax - ymin) * (h[i][j] - xmin) / (xmax - xmin) + ymin)


def histeq(img, nbr_bins=256):
    """ Histogram equalization of a grayscale image. """

    # 获取直方图p(r)
    imhist, bins =np.histogram(img.flatten(), nbr_bins, normed=True)

    # 获取T(r)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]

    # 获取s，并用s替换原始图像对应的灰度值
    result = np.interp(img.flatten(), bins[:-1], cdf)

    return result.reshape(img.shape), cdf
h,jk=histeq(h)

plt.imshow(h,cmap='Reds')
plt.title("均衡化后")
# plt.hist(img.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf', 'histogram'), loc='upper left')







img = cv2.imread("E:/text6/ceshi.tif", 0)

img = cv2.GaussianBlur(img, (3, 3), 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)

# (函数参数3和参数4) 通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
# 118 --是经过某一点曲线的数量的阈值
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # 这里对最后一个参数使用了经验型的值
result = img.copy()
for ig in range(lines.shape[0]):
    for line in lines[ig]:
        rho = line[0]  # 第一个元素是距离rho
        theta = line[1]  # 第二个元素是角度theta
        print(rho)
        print(theta)
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            # 该直线与第一行的交点
            pt1 = (int(rho / np.cos(theta)), 0)
            # 该直线与最后一行的焦点
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            # 绘制一条白线
            cv2.line(result, pt1, pt2, (60))
        else:  # 水平直线
            # 该直线与第一列的交点
            pt1 = (0, int(rho / np.sin(theta)))
            # 该直线与最后一列的交点
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            # 绘制一条直线
            cv2.line(result, pt1, pt2, (60), 1)

plt.figure()
plt.imshow( edges,cmap='Greys')
plt.figure()
plt.imshow(result,cmap='Greys')

plt.show()
