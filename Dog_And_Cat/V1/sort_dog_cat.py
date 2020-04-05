
import os
import tools
import tensorflow as tf
import numpy as np
import shutil

class DogVsCatTest:
    def __init__(self):
        self.BATCH_SIZE = 64
        self.IMG_HEIGHT = 150
        self.IMG_WITH = 150

        self.model = None

    def load_model(self, model):
        """

        :param model:
        :return:
        """
        self.model = tf.keras.models.load_model(model)

    def test(self, test_dir, sorted_cat_dir, sorted_dog_dir):
        """

        :param test_dir:
        :param bad_dir:
        :return:
        """
        if not os.path.isdir(sorted_cat_dir):
            os.makedirs(sorted_cat_dir)

        if not os.path.isdir(sorted_dog_dir):
            os.makedirs(sorted_dog_dir)

        self.test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        self.test_data_gen = self.test_image_generator.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                                           directory=test_dir,
                                                                           target_size=(self.IMG_HEIGHT, self.IMG_WITH),
                                                                           class_mode='categorical',
                                                                           shuffle=False)

        self.prediction = self.model.predict_generator(self.test_data_gen)

        cats = []
        dogs = []
        for index, result in enumerate(self.prediction):
            if result[0] > result[1]:
                cats.append((result[0], self.test_data_gen.filepaths[index]))
            else:
                dogs.append((result[1], self.test_data_gen.filepaths[index]))

        # sorted_cats = sorted(cats, reverse=False)
        # sorted_dogs = sorted(dogs, reverse=False)
        cats.sort(key=lambda x:x[0])
        dogs.sort(key=lambda x:x[0])

        for index, (value, path) in enumerate(cats):
            if value < 1.0:
                shutil.move(path, os.path.join(sorted_cat_dir, str(index+1) + '-' + str(value) + '.jpg'))

        for index, (value, path) in enumerate(dogs):
            if value < 1.0:
                shutil.move(path, os.path.join(sorted_dog_dir, str(index+1) + '-' + str(value) + '.jpg'))



assert tools.set_gpu_memory_growth()

dog_vs_cat_test = DogVsCatTest()
dog_vs_cat_test.load_model('dog&cat.h5')

# 检查训练集
PATH = '../dataset'

train_dir = os.path.join(PATH, 'original', 'train')
train_sorted_cat_dir = os.path.join(PATH, os.getcwd(), 'sorted_dataset/train', 'cat')
train_sorted_dog_dir = os.path.join(PATH, os.getcwd(), 'sorted_dataset/train', 'dog')
dog_vs_cat_test.test(train_dir, train_sorted_cat_dir, train_sorted_dog_dir)

valid_dir = os.path.join(PATH, 'original', 'valid')
valid_sorted_cat_dir = os.path.join(PATH, os.getcwd(), 'sorted_dataset/valid', 'cat')
valid_sorted_dog_dir = os.path.join(PATH, os.getcwd(), 'sorted_dataset/valid', 'dog')
dog_vs_cat_test.test(valid_dir, valid_sorted_cat_dir, valid_sorted_dog_dir)

# PATH = 'check_train_dog'
# train_dog_dir = os.path.join(PATH, 'train')
# but_cat_dir = os.path.join(PATH, 'but_cat')
# dog_vs_cat_test.test(train_dog_dir, "", but_cat_dir)

# # 检查验证集
# PATH = 'check_valid_cat'
# valid_cat_dir = os.path.join(PATH, 'valid')
#
# but_dog_dir = os.path.join(PATH, 'but_dog')
# dog_vs_cat_test.test(valid_cat_dir, but_dog_dir, "")
#
# but_cat_dir = os.path.join(PATH, 'but_cat')
# dog_vs_cat_test.test(valid_cat_dir, "", but_cat_dir)



# # 测试测试集
# PATH = 'cats_and_dogs'
# test_dir = os.path.join(PATH, 'predict')
# pred_dog_dir = os.path.join(PATH, 'predict_dog')
# pred_cat_dir = os.path.join(PATH, 'predict_cat')
# dog_vs_cat_test.test(test_dir, pred_dog_dir, pred_cat_dir)

