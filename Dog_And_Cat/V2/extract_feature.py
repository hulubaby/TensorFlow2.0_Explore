# !usr/bin/python
# -*- codong:utf-8 -*-

"""A extract feature function"""

__author__ = "huzhenhong@2020-03-29"

import tools
import tensorflow as tf
import numpy as np

tools.set_gpu_memory_growth()

def extract_feature(specify_model, image_size):
    base_model = specify_model(input_shape=image_size, weights='imagenet', include_top=False)
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    model.summary()


    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                             horizontal_flip=True,
                                                                             rotation_range=45,
                                                                             width_shift_range=0.15,
                                                                             height_shift_range=0.15,
                                                                             zoom_range=0.5)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=32,
                                                               directory='../dataset/original/train',
                                                               shuffle=False,
                                                               target_size=image_size[:2]
                                                               )

    feature = model.predict_generator(train_data_gen)

    np.save('extract/feature', feature)
    np.save('extract/label', train_data_gen.classes)


extract_feature(tf.keras.applications.ResNet50V2, (150, 150, 3))
