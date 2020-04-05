# !usr/bin/python
# -*- coding:utf-8 -*-

"""extract dog and cat from the trian folder"""

__athour__ = 'huzhenhong@2020-03-15'

import os
from PIL import Image

def make_data_set(src_path, dst_path):
    """

    :param src_path:
    :param dst_path:
    :return:
    """
    dataset_trian_cats = os.path.join(dst_path, 'train', 'cats')
    dataset_trian_dogs = os.path.join(dst_path, 'train', 'dogs')
    # dataset_valid_cats = os.path.join(dst_path, 'valid', 'cats')
    # dataset_valid_dogs = os.path.join(dst_path, 'valid', 'dogs')

    small_img = os.path.join(dst_path, 'small_img')

    if not os.path.isdir(dataset_trian_cats):
        os.makedirs(dataset_trian_cats)

    if not os.path.isdir(dataset_trian_dogs):
        os.makedirs(dataset_trian_dogs)

    # if not os.path.isdir(dataset_valid_cats):
    #     os.makedirs(dataset_valid_cats)
    #
    # if not os.path.isdir(dataset_valid_dogs):
    #     os.makedirs(dataset_valid_dogs)

    if not os.path.isdir(small_img):
        os.makedirs(small_img)

    dogs_and_cats = os.listdir(src_path)

    good_dogs_and_cats = []
    for name in  dogs_and_cats:
        # 过滤掉尺寸小于150x150的图片
        img = Image.open(os.path.join(src_path, name))
        if img.width < 150 or img.height < 150:
            os.symlink(os.path.join(src_path, name), os.path.join(small_img, name))
            continue

        good_dogs_and_cats.append(name)

    cats = [s for s in good_dogs_and_cats if s.startswith('cat')]
    dogs = [s for s in good_dogs_and_cats if s.startswith('dog')]

    # train_cats = cats[:int(len(cats) * 0.8)]
    # valid_cats = cats[int(len(cats) * 0.8):]
    # train_dogs = dogs[:int(len(dogs) * 0.8)]
    # valid_dogs = dogs[int(len(dogs) * 0.8):]

    for cat in cats:
        os.symlink(os.path.join(src_path, cat), os.path.join(dataset_trian_cats, cat))

    for dog in dogs:
        os.symlink(os.path.join(src_path, dog), os.path.join(dataset_trian_dogs, dog))

    # for cat in valid_cats:
    #     os.symlink(os.path.join(src_path, cat), os.path.join(dataset_valid_cats, cat))
    #
    # for dog in valid_dogs:
    #     os.symlink(os.path.join(src_path, dog), os.path.join(dataset_valid_dogs, dog))

print(os.getcwd())
make_data_set('/home/huluwa/PycharmProjects/TensorFlow2.0_Explore/Dog_And_Cat/dataset/train',
              '/home/huluwa/PycharmProjects/TensorFlow2.0_Explore/Dog_And_Cat/dataset/original')

