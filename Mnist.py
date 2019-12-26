# !usr/bin/python
# -*- coding:utf-8 -*-

"""A mnist classfication demo"""

__author__ = 'huluwa@2019-12-13'

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 加载数据集
(train_x, train_y), (validate_x, validate_y) =  datasets.mnist.load_data()
print(train_x.shape, train_y.shape)
print(validate_x.shape, validate_y.shape)

# 归一化训练数据，缩放到（-1，1）
train_x = (tf.convert_to_tensor(train_x, dtype=tf.float32) / 255) * 2 - 1
train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)

# 对标签进行one-hot编码
# train_y = tf.one_hot(train_y, depth=10)
print(train_x.shape, train_y.shape)
print(validate_x.shape, validate_y.shape)

# 构建数据集对象
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.batch(512)    # 批训练

# 构建训练模型
model = keras.Sequential([layers.Dense(256, activation='relu'),
                          layers.Dense(128, activation='relu'),
                          layers.Dense(10)])

# 构建优化器
optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()

# 训练
for epoch in range(2):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:  # 构建梯度计算环境
            out = model(tf.reshape(train_x, (-1, 28 * 28)))
            # 计算预测值与标签的平方和
            train_y_onehot = tf.one_hot(train_y, depth=10)
            loss = tf.square(out - train_y_onehot)
            # 计算每个样本的平均误差
            loss = tf.reduce_sum(loss) / x.shape[0]

        # 计算准确率
        acc_meter.update_state(tf.argmax(out, axis=1), train_y)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 200 ==0:
            print(f'step: {step}, loss: {float(loss.numpy())}, acc: {acc_meter.result().numpy()}')
            acc_meter.reset_states()







