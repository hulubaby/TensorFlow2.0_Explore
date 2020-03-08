# ！usr/bin/pyhton
# -*- coding:utf-8 -*-

"""A advanced tensorboard demo"""

__athour__ = "huluwa@2020-03-08"

import tools
import datetime
import tensorflow as tf
import numpy as np

BATCH_SIZE = 64
EPOCHS = 5

assert tools.set_gpu_memory_growth()


def load_dataset():
    """

    :return:
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(len(x_train)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    return train_dataset, test_dataset


# 定义训练
def train_step(model, optimizer, x_train, y_train):
    """

    :param model:
    :param optimizer:
    :param x_train:
    :param y_train:
    :return:
    """
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)  # 标签和预测值顺序不能错
    # 计算梯度更新
    grads = tape.gradient(loss, model.trainable_variables)
    # 更新梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 记录训练损失和准确率
    train_loss(loss)
    train_accuracy(y_train, predictions)  # 标签和预测值顺序不能错


# 定义测试
def test_step(model, x_test, y_test):
    """

    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    # 记录测试损失和准确率
    test_loss(loss)
    test_accuracy(y_test, predictions)


# 创建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 自定义损失和优化对象
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# 创建测量对象
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# 创建日志记录
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# 加载数据集
train_dataset, test_dataset = load_dataset()

for epoch in range(EPOCHS):
    for step, (x_train, y_train) in enumerate(train_dataset):
        train_step(model, optimizer, x_train, y_train)
        # 每个batch随机保存一张图片
        with train_summary_writer.as_default():
            img = np.reshape(x_train[0], (-1, 28, 28, 1))
            tf.summary.image('image', img, step=step)

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for step, (x_test, y_test) in enumerate(test_dataset):
        test_step(model, x_test, y_test)
        # 每个batch随机保存一张图片
        with train_summary_writer.as_default():
            img = np.reshape(x_test[0], (-1, 28, 28, 1))
            tf.summary.image('image', img, step=step)

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))

    # 重置测量
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
