from __future__ import print_function
import cv2
import argparse
import numpy as np
import random
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('-i', '--input', help='Path to input image.', default='lena.jpg')
args = parser.parse_args()

## [Load image]
src = cv2.imread(cv2.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
## [Load image]



"""
Histogram Equalization Gray
"""

## [Convert to grayscale(binary)]
src_gr = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
## [Convert to grayscale(binary)]

## [Apply Histogram Equalization]
dst_gr = cv2.equalizeHist(src_gr)
## [Apply Histogram Equalization]

## [Convert to bgrimage]
dst_gr_bgr = cv2.cvtColor(dst_gr, cv2.COLOR_GRAY2BGR)
## [Convert to bgrimage]

"""
Histogram Equalization Color-hsv
"""

# hsv 컬러 형태로 변형합니다.
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# h(hue: 색상), s(saturation: 채도), v(value: 명도(색의 밝기))로 컬러 영상을 분리 합니다. 
h, s, v = cv2.split(src_hsv)
# v(value: 명도)값을 히스토그램 평활화를 합니다.
dst_hsv_v = cv2.equalizeHist(v)
# h,s,equalizedV를 합쳐서 새로운 hsv 이미지를 만듭니다.
dst_hsv_merged = cv2.merge([h,s,dst_hsv_v])
# 마지막으로 hsv2를 다시 BGR 형태로 변경합니다.
dst_hsv_merged_bgr = cv2.cvtColor(dst_hsv_merged, cv2.COLOR_HSV2BGR)

"""
Histogram Equalization Color-yCrCb(ycc)
"""

# YCrCb 컬러 형태로 변환합니다.
src_ycc = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
# y(휘도(색의 밝기)), Cr(색차 성분), Cb(색차 성분)로 컬러 영상을 분리 합니다.
y, Cr, Cb = cv2.split(src_ycc)
# y값을 히스토그램 평활화를 합니다.
dst_ycc_y = cv2.equalizeHist(y)
# equalizedY, Cr, Cb를 합쳐서 새로운 yCrCb 이미지를 만듭니다.
dst_ycc_merged = cv2.merge([dst_ycc_y, Cr, Cb])
# 마지막으로 yCrCb2를 다시 BGR 형태로 변경합니다.
dst_ycc_merged_bgr = cv2.cvtColor(dst_ycc_merged, cv2.COLOR_YCrCb2BGR)





# Calculate Image Histogram 
hist1 = cv2.calcHist([src_gr],[0],None,[256],[0,256])
hist2 = cv2.calcHist([dst_gr],[0],None,[256],[0,256])
hist3 = cv2.calcHist([dst_hsv_v],[0],None,[256],[0,256])
hist4 = cv2.calcHist([dst_ycc_y],[0],None,[256],[0,256])

# plot(show) Image
plt.subplot(241),plt.imshow(src_gr, cmap="gray"),plt.title('src_gr')
plt.subplot(242),plt.imshow(dst_gr, cmap="gray"),plt.title('dst_gr')
plt.subplot(243),plt.imshow(dst_hsv_v, cmap="gray"),plt.title('dst_hsv_v')
plt.subplot(244),plt.imshow(dst_ycc_y, cmap="gray"),plt.title('dst_ycc_y')

plt.subplot(245),plt.plot(hist1)
plt.subplot(246),plt.plot(hist2)
plt.subplot(247),plt.plot(hist3)
plt.subplot(248),plt.plot(hist4)
plt.xlim([0,256])
plt.show()

# Resize Factor
fx_f = 0.2
fy_f = 0.2

# Image Resize
src_r = cv2.resize(src, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv2.INTER_AREA)
dst_gr_bgr_r = cv2.resize(dst_gr_bgr, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv2.INTER_AREA)
dst_hsv_merged_bgr_r = cv2.resize(dst_hsv_merged_bgr, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv2.INTER_AREA)
dst_ycc_merged_bgr_r = cv2.resize(dst_ycc_merged_bgr, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv2.INTER_AREA)

## [Display results]
cv2.imshow('Source image', src_r)
cv2.imshow('dst_gr_bgr', dst_gr_bgr_r)
cv2.imshow('dst_hsv_merged_bgr', dst_hsv_merged_bgr_r)
cv2.imshow('dst_ycc_merged_bgr', dst_ycc_merged_bgr_r)
## [Display results]

## [Wait until user exits the program]
cv2.waitKey()
## [Wait until user exits the program]