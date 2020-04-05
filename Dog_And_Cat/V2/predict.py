# !usr/bin/python
# -*- codong:utf-8 -*-

"""A predict function"""

__author__ = "huzhenhong@2020-03-20"

import tools
import os
import tensorflow as tf
import numpy as np

tools.set_gpu_memory_growth()


def creat_predict_model(feature_model_path, image_size):
    """

    :param feature_model_path:
    :param image_size:
    :return:
    """
    base_model = tf.keras.applications.ResNet50V2(input_shape=image_size, weights='imagenet', include_top=False)
    base_model.trainable = False

    feature_model = tf.keras.models.load_model(feature_model_path)
    feature_model.trainable = False

    predict_model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        feature_model
    ])

    predict_model.summary()

    return predict_model


def predict_images(model, src_path, dst_path):
    """

    :param model:
    :param src_path:
    :param dst_path:
    :return:
    """
    dog_path = os.path.join(dst_path, 'dog')
    if not os.path.isdir(dog_path):
        os.makedirs(dog_path)

    cat_path = os.path.join(dst_path, 'cat')
    if not os.path.isdir(cat_path):
        os.makedirs(cat_path)

    test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_data_gen = test_image_generator.flow_from_directory(batch_size=32,
                                                             directory=src_path,
                                                             target_size=(150, 150),
                                                             class_mode='categorical',
                                                             shuffle=False)

    result = model.predict_generator(test_data_gen)
    prediction = np.argmax(result, axis=1)

    for index, pred in enumerate(prediction):
        name = test_data_gen.filenames[index].split('/')[1]
        if pred == 0:
            os.symlink(os.path.join(src_path, test_data_gen.filenames[index]),
                       os.path.join(dst_path, 'cat', name))
        else:
            os.symlink(os.path.join(src_path, test_data_gen.filenames[index]),
                       os.path.join(dst_path, 'dog', name))


print(os.getcwd())

predict_model = creat_predict_model('saved_model/2020-04-05-14:27:25/epoch-55.h5', (150, 150, 3))

# 保存最终的预测模型
predict_model.save('predict_model.h5')

predict_images(predict_model,
               '/home/huluwa/PycharmProjects/TensorFlow2.0_Explore/Dog_And_Cat/dataset/test',
               '/home/huluwa/PycharmProjects/TensorFlow2.0_Explore/Dog_And_Cat/dataset/predict')

# 测试网上下载的图片
print("--------------------")
imgs = [get_img('cats_and_dogs/dowload_from_network/' + str(i+1) + '.jpg') for i in range(8)]
imgs = tf.concat(imgs, axis=0)
results = predict_model.predict(imgs)
print(results)


