import os
import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 限制显存占用
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0

try:
    # Currently, memory growth needs to be the same across GPUs
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except :
    # Memory growth must be set before GPUs have been initialized
    print(e)

print('gpu:', tf.test.is_gpu_available())

print(tf.__version__)

a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)


cv.namedWindow('hello')
cv.imshow('hello', np.ones([100,200]))
cv.waitKey()
cv.destroyAllWindows()





