# !usr/bin/python
# -*- coding:utf-8 -*-

"""A mnist classfication demo with keras"""

__author__ = 'huluwa@2020-02-25'

import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, losses
import matplotlib.pyplot as plt


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


def preprocess(x, y):
    """
    数据预处理
    :param x:
    :param y:
    :return:
    """
    # x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])  # 将图片[b, 28, 28] 打平成 [b, 28*28]

    return x, y


def make_dataset():
    """

    :return:
    """
    # 加载mnist数据集
    (train_data, train_label), (test_data, test_label) = datasets.mnist.load_data()
    # 构建训练张量数据集
    train_db = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_db = train_db.shuffle(BATCH_SIZE * 4).batch(BATCH_SIZE).map(preprocess).prefetch(BATCH_SIZE)

    if DEBUG is True:
        x, y = next(iter(train_db))
        print('train x shape: ', tf.shape(x))
        print('train x dtype: ', x.dtype)
        print('train x max: ', int(tf.reduce_max(x)))
        print('train x min: ', int(tf.reduce_min(x)))
        print('train y shape: ', tf.shape(y))
        print('train y dtype: ', y.dtype)
        print('train y max: ', int(tf.reduce_max(y)))
        print('train y min: ', int(tf.reduce_min(y)))

    test_db = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_db = test_db.shuffle(BATCH_SIZE * 4).batch(BATCH_SIZE).map(preprocess).prefetch(BATCH_SIZE)

    if DEBUG is True:
        x, y = next(iter(test_db))
        print('test x shape: ', tf.shape(x))
        print('test x dtype: ', x.dtype)
        print('test x max: ', int(tf.reduce_max(x)))
        print('test x min: ', int(tf.reduce_min(x)))
        print('test y shape: ', tf.shape(y))
        print('test y dtype: ', y.dtype)
        print('test y max: ', int(tf.reduce_max(y)))
        print('test y min: ', int(tf.reduce_min(y)))

    return train_db, test_db


def draw_train_result(epoch, loss, acc):
    """
    绘制训练结果
    :param epoch:
    :param loss:
    :param acc:
    :return:
    """
    plt.plot(epoch, loss, color='C1', marker='s', label='loss')
    plt.plot(epoch, acc, color='C0', marker='^', label='acc')

    plt.ylim([0, 1])
    plt.xlabel('train epoch')
    plt.ylabel('trian result')
    plt.legend()  # 子图说明
    plt.savefig('mnist_with_keras.svg')
    plt.show()


def get_img(img_path):
    """
    获取图片
    :param img_path:
    :return:
    """
    # 根据路径读取图片
    img = tf.io.read_file(img_path)
    # 解码图片，这里应该是解码成了png格式
    img = tf.image.decode_png(img, channels=1)
    # 大小缩放
    img = tf.image.resize(img, [28, 28])
    # img = tf.cast(img, dtype=tf.uint8)
    # img = tf.image.encode_png(img)
    #
    # with tf.io.gfile.GFile(img_path, 'wb') as file:
    #     file.write(img.numpy())

    # 转换成张量
    # img = 2 * tf.convert_to_tensor(img, dtype=tf.float32) / 255.0 - 1  # [0, 255]=>[-1.0, 1.0]
    img = tf.cast(img, dtype=tf.float32) / 255.
    img = tf.reshape(img, (-1, 28 * 28))

    return img


"""训练主逻辑"""
print(os.getcwd())
# 设置显存按需增长
assert set_gpu_memory_growth()
# 设置全局超参数
DEBUG = True
BATCH_SIZE = 32
EPOCH_SIZE = 10
LR = 0.01

# 准备数据集
(train_db, test_db) = make_dataset()

# 构建训练模型
model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2()),
    layers.Dense(256, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(10, activation='softmax')
    layers.Dense(10)
])

model.compile(optimizer=optimizers.Adam(lr=LR),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              # loss=losses.mse,
              metrics=['accuracy'])

print('Begin train...')
history = model.fit(train_db, epochs=EPOCH_SIZE)
print('Finished train...')

draw_train_result(history.epoch, history.history['loss'], history.history['accuracy'])

print('Begin evaluate...')
val = model.evaluate(test_db)
print('Finished evaluate...')
print(val)

# 测试
img_280x280_1 = get_img('280x280/1.png')
img_280x280_2 = get_img('280x280/2.png')
img_280x280_4 = get_img('280x280/4.png')
img_280x280_7 = get_img('280x280/7.png')
imgs_280x280 = tf.concat([img_280x280_1, img_280x280_2, img_280x280_4, img_280x280_7], axis=0)
result_280x280 = model.predict(imgs_280x280)
print(tf.argmax(result_280x280, axis=1))

img_20x20_1 = get_img('20x20/1.png')
img_20x20_2 = get_img('20x20/2.png')
img_20x20_4 = get_img('20x20/4.png')
img_20x20_7 = get_img('20x20/7.png')
imgs_20x20 = tf.concat([img_20x20_1, img_20x20_2, img_20x20_4, img_20x20_7], axis=0)
result_20x20 = model.predict(imgs_20x20)
print(tf.argmax(result_20x20, axis=1))
