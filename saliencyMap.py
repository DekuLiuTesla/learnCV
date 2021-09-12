import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "./images/"
img = cv2.imread(img_path+"house2.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

dft = np.fft.fft2(img_gray)  # 傅里叶变换
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.abs(dft_shift)
phase_spectrum = np.angle(dft_shift)

img_dp = np.abs(np.fft.ifft2(np.e**(1j*phase_spectrum)))
imgS = cv2.GaussianBlur(img_dp*img_dp, (7, 7), 2)

plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.axis("off")
plt.imshow(img_gray, cmap="gray")
plt.title("Origin")
plt.subplot(212)
plt.axis("off")
plt.imshow(imgS, cmap="jet")
plt.title("Saliency Map")
plt.pause(0.001)

