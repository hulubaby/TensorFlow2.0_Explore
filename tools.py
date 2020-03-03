# !usr/bin/python
# -*- coding:utf-8 -*-

"""A tensorflow2.0 tools"""

__author__ = "huluwa@2020-03-02"

import os
import tensorflow as tf


def set_gpu_memory_growth():
    """
    设置gpu显存按需申请
    :return:
    """
    print(tf.version.VERSION)

    # 设置tensorflow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 仅仅使用第一块GPU

    # 设置gpu显存按需申请
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

    for gpu in physical_devices:
        print(physical_devices[0])
        print(gpu)
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            return False

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')  # 这句不能少，why？
    print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    print(tf.test.is_gpu_available())

    return True
