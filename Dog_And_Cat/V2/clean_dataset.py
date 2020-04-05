
import os
import tools
import tensorflow as tf
import numpy as np
import shutil

assert tools.set_gpu_memory_growth()

class DogVsCatTest:
    def __init__(self):
        self.BATCH_SIZE = 96
        self.IMG_HEIGHT = 150
        self.IMG_WITH = 150
        self.model = None

    def load_model(self, model):
        """

        :param model:
        :return:
        """
        self.model = tf.keras.models.load_model(model)

        self.model.summary()

    def test(self, test_dir, mode='trian'):
        """

        :param test_dir:
        :return:
        """

        self.bad_path = os.path.join('bad', mode)

        if not os.path.isdir(self.bad_path):
            os.makedirs(self.bad_path)

        self.test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        self.test_data_gen = self.test_image_generator.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                                           directory=test_dir,
                                                                           target_size=(self.IMG_HEIGHT, self.IMG_WITH),
                                                                           class_mode='categorical',
                                                                           shuffle=False)  # 切记不能打乱

        self.prediction = self.model.predict_generator(self.test_data_gen)

        self.prediction = np.argmax(self.prediction, axis=1)

        # 预测结果和label不一样，就认为是数据错了
        index = np.where(self.prediction != self.test_data_gen.labels)
        index = list(index[0])
        for i in index:
            path = os.path.join(self.bad_path, self.test_data_gen.filenames[i].split('/')[0])
            if not os.path.isdir(path):
                os.makedirs(path)
            shutil.move(self.test_data_gen.filepaths[i], os.path.join(self.bad_path, self.test_data_gen.filenames[i]))


dog_vs_cat_test = DogVsCatTest()

dog_vs_cat_test.load_model('../V1/dog&cat.h5')

# 检查训练集
PATH = '../dataset/original'

original_train_dir = os.path.join(PATH, 'train')
dog_vs_cat_test.test(original_train_dir, 'train')
