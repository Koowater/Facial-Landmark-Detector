import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from libs.eval import Inferencer

import cv2

import numpy as np 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

# model = tf.saved_model.load('data/checkpoint/cp_77.ckpt')
model = Inferencer('data/checkpoint/cp_77.ckpt')

result = model('data/test/0.jpg')
detector = cv2.dnn.readNetFromCaffe("data/test/deploy.prototxt" , "data/test/res10_300x300_ssd_iter_140000.caffemodel")

image = cv2.imread('data/test/0.jpg')
height, width, _ = image.shape

for landmark in result[0]:
    mapped_landmark = (int(landmark[0] / 64 * width), int(landmark[1] / 64 * height))
    image = cv2.circle(image, mapped_landmark, 10, (255, 0, 0))

cv2.imshow('result', image)
cv2.waitKey() 
cv2.destroyAllWindows()
