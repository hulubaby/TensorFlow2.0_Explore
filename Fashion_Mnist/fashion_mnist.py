# !usr/bin/python
# -*- coding:utf-8 -*-

"""A fashion mnist demo"""

__author__ = "huluwa@2020-03-05"

import tools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

print(tf.version.VERSION)


def creat_dataset():
    """
    创建数据集
    :return:
    """
    # 加载数据集
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    # 图片预处理
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)


def show_images(images):
    """
    批量显示图片
    :param images:
    :return:
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover',
                   'Dress', 'Coat', 'Sandal', 'Shirt',
                   'Sneaker', 'Bag', 'Ankle boot']
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])  # 横坐标设为空
        plt.yticks([])  # 纵坐标设为空
        plt.grid(False)  # 默认为False
        plt.imshow(images[i], cmap='gray')  # 显示灰度图
        plt.xlabel(class_names[train_labels[i]])  # 设置横坐标说明
    plt.show()


def create_model():
    """
    # 搭建模型
    :return:
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # 展平输入图像
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def test_model(model):
    """

    :param model:
    :return:
    """
    # 验证模型
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\ntest loss: {}, test acc: {:5.2f}'.format(test_loss, test_acc * 100))

    # 预测模型
    predictions = model.predict(test_images)
    print(predictions[0])
    print(np.argmax(predictions[0]), test_labels[0])


"""
主逻辑
"""
assert tools.set_gpu_memory_growth()

CHECKPOINT_PATH = 'fashion_mnist_train/cp-{epoch:02d}.h5'
# CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

# 加载数据
(train_images, train_labels), (test_images, test_labels) = creat_dataset()

# 显示前25张训练图片
show_images(train_images)

# 创建模型
model = create_model()

# 保存模型初始参数
model.save_weights(CHECKPOINT_PATH.format(epoch=0, loss=0, accuracy=0.0))

# 创建模型保存回调，仅仅保存参数信息，不保存网络结构和优化器配置
save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                         save_weights_only=True,
                                                         # save_best_only=True,  # 只有准确率比上一次高时才会保存
                                                         save_freq=1,
                                                         verbose=0)
# 训练模型
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_split=0.2,  # 20%的训练集作为验证集使用
          validation_freq=1,
          callbacks=[save_model_callback])

# 保存整个模型
model.save('my_model.h5')

# 删除旧模型，创建一个新模型
del model
new_model_without_weights = create_model()

# 加载最后保存的模型参数,并测试模型
# latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
new_model_without_weights.load_weights('fashion_mnist_train/cp-10.h5')
test_model(new_model_without_weights)

# 加载最后保存的整个模型,并测试模型
new_model_with_weights = tf.keras.models.load_model('my_model.h5')
new_model_with_weights.summary()
test_model(new_model_with_weights)
