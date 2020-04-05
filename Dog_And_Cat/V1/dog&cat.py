# !usr/bin/python
# -*- coding:utf-8 -*-

"""A dog and cat ca classify demo"""

__author__ = "huluwa@2020-03-18"

import tools
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

assert tools.set_gpu_memory_growth()


class DogVsCat:
    def __init__(self, dataset_path):
        """

        :param dataset_path:
        """
        tools.set_gpu_memory_growth()

        self.path = dataset_path
        self.train_dir = os.path.join(self.path, 'train')
        self.valid_dir = os.path.join(self.path, 'valid')

        self.train_cats_dir = os.path.join(self.train_dir, 'cats')
        self.train_dogs_dir = os.path.join(self.train_dir, 'dogs')
        self.valid_cats_dir = os.path.join(self.valid_dir, 'cats')
        self.valid_dogs_dir = os.path.join(self.valid_dir, 'dogs')

        self.num_cats_train = len(os.listdir(self.train_cats_dir))
        self.num_dogs_train = len(os.listdir(self.train_dogs_dir))

        self.num_cats_val = len(os.listdir(self.valid_cats_dir))
        self.num_dogs_val = len(os.listdir(self.valid_dogs_dir))

        self.total_train = self.num_cats_train + self.num_dogs_train
        self.total_val = self.num_cats_val + self.num_dogs_val

        self.BATCH_SIZE = 96
        self.EPOCHS = 10
        self.LR = 0.01
        self.IMG_HEIGHT = 150
        self.IMG_WITH = 150

        self.train_image_generator = None
        self.valid_image_generator = None
        self.train_data_gen = None
        self.valid_data_gen = None

        self.history = None
        self.tensorboard_callback = None
        self.early_stopping_callback = None
        self.save_model_callback = None
        self.model = None

    def create_image_generator(self):
        """

        :return:
        """
        self.train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                                     horizontal_flip=True,
                                                                                     rotation_range=45,
                                                                                     width_shift_range=0.15,
                                                                                     height_shift_range=0.15,
                                                                                     zoom_range=0.5)

        self.valid_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        self.train_data_gen = self.train_image_generator.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                                             directory=self.train_dir,
                                                                             shuffle=True,
                                                                             target_size=(self.IMG_HEIGHT,
                                                                                          self.IMG_WITH)
                                                                             )

        self.valid_data_gen = self.valid_image_generator.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                                             directory=self.valid_dir,
                                                                             shuffle=True,
                                                                             target_size=(self.IMG_HEIGHT,
                                                                                          self.IMG_WITH)
                                                                             )

        print(self.train_data_gen.class_indices)
        print(self.valid_data_gen.class_indices)

    def check_dataset(self):
        """

        :return:
        """
        # 查看数据集前五张图片
        sample_training_images, _ = next(self.train_data_gen)
        self.plot_images(sample_training_images[:5])

        # # 查看第一张图片增强操作后的前五张
        # augmented_images = [self.train_data_gen[0][0][0] for i in range(5)]
        # self.plot_images(augmented_images)

    @staticmethod
    def plot_images(images):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def create_model(self):
        """

        :return:
        """
        base_model = tf.keras.applications.ResNet50V2(input_shape=(self.IMG_HEIGHT, self.IMG_WITH, 3),
                                                      include_top=False,
                                                      weights='imagenet')
        base_model.trainable = False

        self.model = tf.keras.models.Sequential([
            base_model,
            # tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.LR),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

        self.model.summary()

    def create_callback(self):
        """

        :return:
        """
        # 创建tensorboard回调
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # 创建提前结束回调
        # patience：没有进步的训练轮数，在这之后训练就会被停止。
        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

        # 创建模型保存回调
        save_path = os.path.join('saved_model', datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        os.makedirs(save_path)
        checkpoint_path = os.path.join(save_path, 'epoch-{epoch:02d}.h5')
        self.save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                      monitor='val_accuracy',
                                                                      save_weights_only=False,
                                                                      save_best_only=True,
                                                                      save_freq='epoch',
                                                                      verbose=0)

    def fit(self):
        """

        :return:
        """
        self.history = self.model.fit_generator(
            self.train_data_gen,
            # steps_per_epoch=self.total_train // self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=self.valid_data_gen,
            # validation_steps=self.total_val // self.BATCH_SIZE,
            callbacks=[self.tensorboard_callback,
                       self.early_stopping_callback,
                       self.save_model_callback]
        )

        self.model.save('dog&cat.h5')

    def draw_result(self):
        """

        :return:
        """
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs_range = self.history.epoch

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


print(os.getcwd())

dog_vs_cat = DogVsCat('/home/huluwa/PycharmProjects/TensorFlow2.0_Explore/Dog_And_Cat/dataset/original1')
dog_vs_cat.create_image_generator()
# dog_vs_cat.check_dataset()
dog_vs_cat.create_model()
dog_vs_cat.create_callback()
dog_vs_cat.fit()
dog_vs_cat.draw_result()
