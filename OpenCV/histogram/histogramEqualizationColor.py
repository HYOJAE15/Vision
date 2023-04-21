import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from utils import BGR_hist_subplot

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image path to the directory")
args = vars(ap.parse_args())
path = args['image']

# 입력 받은 이미지를 불러옵니다. opencv: cv2.imread(img, cv2.IMREAD_COLOR(default)), cv2.IMREAD_COLOR: If set, always convert image to the 3 channel BGR color image.
src = cv2.imread(path)

"""
BGR
"""
# B, G, R 컬러 영상을 분리 
b, g, r = cv2.split(src)
# 히스토그램 평탄화
equalized_b = cv2.equalizeHist(b)
equalized_g = cv2.equalizeHist(g)
equalized_r = cv2.equalizeHist(r)
# merged
bgrDst = cv2.merge([equalized_b, equalized_g, equalized_r])

"""
HSV
"""
# hsv 컬러 형태로 변형합니다.
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# h, s, v로 컬러 영상을 분리 합니다. 
h, s, v = cv2.split(hsv)
# v값을 히스토그램 평활화를 합니다.
equalizedV = cv2.equalizeHist(v)
# h,s,equalizedV를 합쳐서 새로운 hsv 이미지를 만듭니다.
hsv2 = cv2.merge([h,s,equalizedV])
# 마지막으로 hsv2를 다시 BGR 형태로 변경합니다.
hsvDst = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

"""
YCrCb
"""
# YCrCb 컬러 형태로 변환합니다.
yCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
# y, Cr, Cb로 컬러 영상을 분리 합니다.
y, Cr, Cb = cv2.split(yCrCb)
# y값을 히스토그램 평활화를 합니다.
equalizedY = cv2.equalizeHist(y)
# equalizedY, Cr, Cb를 합쳐서 새로운 yCrCb 이미지를 만듭니다.
yCrCb2 = cv2.merge([equalizedY, Cr, Cb])
# 마지막으로 yCrCb2를 다시 BGR 형태로 변경합니다.
yCrCbDst = cv2.cvtColor(yCrCb2, cv2.COLOR_YCrCb2BGR)


# Resize Factor
fx_f = 0.3
fy_f = 0.3

# Image Resize
src_r = cv2.resize(src, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv2.INTER_AREA)
bgrDst_r = cv2.resize(bgrDst, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv2.INTER_AREA)
hsvDst_r = cv2.resize(hsvDst, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv2.INTER_AREA)
yCrCbDst_r = cv2.resize(yCrCbDst, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv2.INTER_AREA)


# src, hsv, YCrCb 각각을 출력합니다.
cv2.imshow('src', src_r)
cv2.imshow('bgr dst', bgrDst_r)
cv2.imshow('hsv dst', hsvDst_r)
cv2.imshow('YCrCb dst', yCrCbDst_r)

BGR_hist_subplot(src, 141, "src")
BGR_hist_subplot(bgrDst, 142, "bgrDst")
BGR_hist_subplot(hsvDst, 143, "hsvDst")
BGR_hist_subplot(yCrCbDst, 144, "yCrCbDst")
plt.xlim([0,256])
plt.show()


cv2.waitKey()
cv2.destroyAllWindows()
