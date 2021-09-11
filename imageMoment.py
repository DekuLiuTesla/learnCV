import cv2
import numpy as np
from skimage import img_as_float, data
import matplotlib.pyplot as plt

img_path = "./images/girl.jpg"
img = cv2.imread(img_path)  # 以ndarray对象存储
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = img_as_float(img_gray)  # 转化为float类型
moments = cv2.moments(img_gray)
# 计算零阶矩
M00 = np.sum(img_gray).item()
print("M00: ", M00)
# 计算一阶矩
(M, N) = img_gray.shape
x = np.linspace(0, N-1, N)
y = np.linspace(0, M-1, M)
x_grid, y_grid = np.meshgrid(x, y)  # 使用的坐标系与图像坐标系不同
M10 = np.sum(img_gray*x_grid).item()
miu_x = M10/M00
print("M10: ", M10)
print("miu_x: ", miu_x)
M01 = np.sum(img_gray*y_grid).item()
miu_y = M01/M00
print("M01: ", M01)
print("miu_y: ", miu_y)
# 计算二阶矩
M20 = np.sum(img_gray*x_grid*x_grid).item()
M02 = np.sum(img_gray*y_grid*y_grid).item()
M11 = np.sum(img_gray*x_grid*y_grid).item()
print("M20: ", M20)
print("M02: ", M02)
print("M11: ", M11)

print("Moments: ", moments)  # opencv中，x为列，y为行
cv2.imshow("Origin", img_gray)
cv2.waitKey()
# plt.imshow(img_gray, cmap="gray")
# plt.title("Girl")
# plt.pause(0.001)