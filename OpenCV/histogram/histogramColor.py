import cv2
import os
import argparse
from utils import compareColorHistogram, compareColorHistogram2d, BGR_histogram

ap = argparse.ArgumentParser()

# 이미지 경로, 히스토그램의 간격 사이즈를 받습니다.
ap.add_argument("-i", "--image", required=True, help="Image path")
ap.add_argument("-b", "--bins", required=True, help="Each hist size")
args = vars(ap.parse_args())

path = args['image']
histSize = int(args['bins'])


# 그래프 제목을 출력하기 위해 입력받은 사진의 이름을 저장합니다.
fname = os.path.basename(path)
# 이미지를 color로 불러옵니다.
src = cv2.imread(path, cv2.IMREAD_COLOR)

compareColorHistogram(src, histSize, fname)
compareColorHistogram2d(src, histSize, fname)
