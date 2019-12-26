# !usr/bin/python
# -*- coding:utf-8 -*-

"""A comparison between gpu and cpu"""

__author__ = 'huzhenhong@2019-12-19'


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import timeit


# 确保是2以上版本
assert tf.__version__.startswith('2.')

cpu_data = []
gpu_data = []

for n in range(9):
    n = 10**n
    # 创建在CPU上运行的两个矩阵
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([1, n])
        cpu_b = tf.random.normal([n, 1])
        print('cpu_a device: {}, cpu_b device: {}', cpu_a, cpu_b)

    # 创建在GPU上运行的两个矩阵
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([1, n])
        gpu_b = tf.random.normal([n, 1])
        print('gpu_a device: {}, gpu_b device: {}', gpu_a, gpu_b)

    def cpu_run():
        with tf.device('/cpu:0'):
            ret = tf.matmul(cpu_a, cpu_b)
        return ret

    def gpu_run():
        with tf.device('/gpu:0'):
            ret = tf.matmul(gpu_a, gpu_b)
        return ret


    # 第一次计算需要热身，避免将初始化阶段时间计算在内
    cpu_costtime = timeit.timeit(cpu_run, number=10)
    gpu_costtime = timeit.timeit(gpu_run, number=10)
    print('warm up cpu cost: {}, gpu cost: {}', cpu_costtime, gpu_costtime)

    # 正式计算10次，取平均值
    cpu_costtime = timeit.timeit(cpu_run, number=10)
    gpu_costtime = timeit.timeit(gpu_run, number=10)
    cpu_data.append(cpu_costtime/10)
    gpu_data.append(gpu_costtime/10)

    del cpu_a, cpu_b, gpu_a, gpu_b
