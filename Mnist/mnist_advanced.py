# !usr/bin/python
# -*- coding:utf-8 -*-

"""A mnist demo copy from tensorflow official tutorials"""

__author__ = "huluwa@2020-03-02"

import tools
import tensorflow as tf
from tensorflow.keras import Model, datasets
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import matplotlib.pyplot as plt
import timeit


assert tools.set_gpu_memory_growth()


# 使用keras模型子类化API构建模型
class MyModel(Model):
    def __init__(self):
        """
        初始化
        """
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.epochs = []
        self.result_train_loss = []
        self.result_train_acc = []
        self.result_test_loss = []
        self.result_test_acc = []

        self.make_dataset()

    def calc(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    def make_dataset(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        # 拓展一个通道
        self.x_train = self.x_train[..., tf.newaxis]
        self.x_test = self.x_test[..., tf.newaxis]

        self.train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)) \
            .shuffle(10000).batch(32)

        self.test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(32)

    @tf.function  # 将python代码编译成tf静态图结构，加快执行
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.calc(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.calc(images)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self, epochs=5):
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in self.train_ds:
                self.train_step(images, labels)

            for images, labels in self.test_ds:
                self.test_step(images, labels)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test loss: {}, Test accuracy: {}'
            print(template.format(epoch+1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result()*100,
                                  self.test_loss.result(),
                                  self.test_accuracy.result()*100))

            self.epochs.append(epoch+1)
            self.result_train_loss.append(self.train_loss.result())
            self.result_train_acc.append(self.train_accuracy.result())
            self.result_test_loss.append(self.test_loss.result())
            self.result_test_acc.append(self.test_accuracy.result())

    def draw_result(self):
        plt.plot(self.epochs, self.result_train_loss, color='C1', marker='s', label='train_loss')
        plt.plot(self.epochs, self.result_train_acc, color='C0', marker='^', label='train_acc')

        plt.plot(self.epochs, self.result_test_loss, color='C2', marker='o', label='test_loss')
        plt.plot(self.epochs, self.result_test_acc, color='C3', marker='D', label='test_acc')

        plt.ylim([0, 1])
        plt.xlabel('train epoch')
        plt.ylabel('trian result')
        plt.legend()  # 子图说明
        plt.savefig('mnist_advanced.svg')
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
    img = tf.cast(img, dtype=tf.float32) / 255.

    return img


EPOCHS = 5
my_model = MyModel()

start_time = timeit.default_timer()

my_model.train(EPOCHS)

finish_time = timeit.default_timer()
print('Train cost time : {}'.format(finish_time - start_time))

my_model.draw_result()

# 测试
img_280x280_1 = get_img('280x280/1.png')
img_280x280_2 = get_img('280x280/2.png')
img_280x280_4 = get_img('280x280/4.png')
img_280x280_7 = get_img('280x280/7.png')
imgs_280x280 = [img_280x280_1, img_280x280_2, img_280x280_4, img_280x280_7]
result_280x280 = my_model.calc(imgs_280x280)
print(tf.argmax(result_280x280, axis=1))

img_20x20_1 = get_img('20x20/1.png')
img_20x20_2 = get_img('20x20/2.png')
img_20x20_4 = get_img('20x20/4.png')
img_20x20_7 = get_img('20x20/7.png')
imgs_20x20 = [img_20x20_1, img_20x20_2, img_20x20_4, img_20x20_7]
result_20x20 = my_model.calc(imgs_20x20)
print(tf.argmax(result_20x20, axis=1))
