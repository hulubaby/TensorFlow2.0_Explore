# !usr/bin/python
# -*- codong:utf-8 -*-

"""A extract feature function"""

__author__ = "huzhenhong@2020-03-29"

import tools
import os
import tensorflow as tf
import numpy as np
import datetime
from matplotlib import pyplot as plt

tools.set_gpu_memory_growth()


def feature_train():
    """

    :return:
    """
    train_x = np.load('extract/feature.npy')
    train_y = np.load('extract/label.npy')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2048,)),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'],)

    model.summary()

    # 创建tensorboard回调
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 创建提前结束回调
    # patience：没有进步的训练轮数，在这之后训练就会被停止。
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50)

    # 创建模型保存回调
    save_path = os.path.join('saved_model', datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    os.makedirs(save_path)
    checkpoint_path = os.path.join(save_path, 'epoch-{epoch:02d}.h5')
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             monitor='val_accuracy',
                                                             save_weights_only=False,
                                                             save_best_only=True,
                                                             save_freq='epoch',
                                                             verbose=0)

    history = model.fit(train_x,
                        train_y,
                        batch_size=96,
                        epochs=100,
                        validation_split=0.2,
                        callbacks=[tensorboard_callback,
                                   early_stopping_callback,
                                   save_model_callback])
    return history


def draw_result(history):
    """

    :return:
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = history.epoch

    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()


history = feature_train()
draw_result(history)
