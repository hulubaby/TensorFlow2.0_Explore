# !usr/bin/python
# -*- coding:utf-8 -*-

"""A mnist classfication demo without keras"""

__author__ = 'huluwa@2020-02-13'

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets


def set_gpu_memory_growth():
    """
    设置gpu显存按需申请
    :return:
    """
    print(tf.version.VERSION)

    # 设置tensorflow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 仅仅使用第一块GPU

    # 设置gpu显存按需申请
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

    for gpu in physical_devices:
        print(physical_devices[0])
        print(gpu)
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            return False

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')  # 这句不能少，why？
    print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    print(tf.test.is_gpu_available())

    return True


def preprocess(x, y):
    """
    数据预处理
    :param x:
    :param y:
    :return:
    """
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    # x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])  # 将图片[b, 28, 28] 打平成 [b, 28*28]

    return x, y


def make_dataset():
    """

    :return:
    """
    # 加载mnist数据集
    (train_data, train_label), (test_data, test_label) = datasets.mnist.load_data()
    # 构建训练张量数据集
    train_db = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_db = train_db.shuffle(BATCH_SIZE * 4).batch(BATCH_SIZE).map(preprocess) \
        .repeat(REPEAT_SIZE).prefetch(BATCH_SIZE)

    if DEBUG is True:
        x, y = next(iter(train_db))
        print('train x shape: ', tf.shape(x))
        print('train x dtype: ', x.dtype)
        print('train x max: ', int(tf.reduce_max(x)))
        print('train x min: ', int(tf.reduce_min(x)))
        print('train y shape: ', tf.shape(y))
        print('train y dtype: ', y.dtype)
        print('train y max: ', int(tf.reduce_max(y)))
        print('train y min: ', int(tf.reduce_min(y)))

    test_db = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_db = test_db.shuffle(BATCH_SIZE * 4).batch(BATCH_SIZE).map(preprocess) \
        .repeat(REPEAT_SIZE).prefetch(BATCH_SIZE)

    if DEBUG is True:
        x, y = next(iter(test_db))
        print('test x shape: ', tf.shape(x))
        print('test x dtype: ', x.dtype)
        print('test x max: ', int(tf.reduce_max(x)))
        print('test x min: ', int(tf.reduce_min(x)))
        print('test y shape: ', tf.shape(y))
        print('test y dtype: ', y.dtype)
        print('test y max: ', int(tf.reduce_max(y)))
        print('test y min: ', int(tf.reduce_min(y)))

    return train_db, test_db


def calc_forward(x):
    """
    前向计算
    :param x:
    :return:
    """
    # 计算第一层输出，[b, 784]@[784, 256] + [256] => [b, 256]
    h1 = x @ w1 + tf.broadcast_to(b1, [tf.shape(x)[0], 256])  # 手动 broadcast b1
    h1 = tf.nn.relu(h1)  # 应用 relu 激活函数

    # 计算第二层输出，[b, 256]@[256, 256] + [128] => [b, 128]
    h2 = h1 @ w2 + b2  # 自动 broadcast b2
    h2 = tf.nn.relu(h2)

    # 计算第三层输出，[b, 128]@[128, 10] + [10] => [b, 10]
    out = h2 @ w3 + b3  # 自动 broadcast b3

    return out


def update_param(tape, pred, label):
    """
    梯度更新
    :param tape:
    :param pred:
    :param label:
    :return:
    """
    # 将训练便签进行独热编码，[b]=>[b, 10]
    label_onehot = tf.one_hot(label, depth=10)

    # 计算模型计算结果和真实值的差的平方和，[b, 10]
    loss = tf.square(pred - label_onehot)

    # 计算每个样本的平均误差，[b]
    loss = tf.reduce_mean(loss)

    # 计算梯度
    grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

    # 更新梯度 w1 = w1 - lr * w1_grad
    for param, grad in zip([w1, b1, w2, b2, w3, b3], grads):
        param.assign_sub(LR * grad)

    return loss


def evaluate_acc():
    """
    计算验证准去率
    :return:
    """
    accs = []
    for (x, y) in test_db:
        x = tf.reshape(x, (-1, 28 * 28))
        pred = calc_forward(x)
        pred = tf.argmax(pred, axis=1)
        pred = tf.cast(pred, dtype=tf.uint8)
        result = pred - y
        acc = 1.0 - 1.0 * float(tf.math.count_nonzero(result) / len(result))
        accs.append(acc)

    return sum(accs) / len(accs)


def train():
    """
    训练
    :return:
    """
    # 按批次训练
    losslist = []
    acclist = []
    steplist = []

    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = calc_forward(x)
            loss = update_param(tape, out, y)

        # 每500个step打印一下训练状态
        if step % 500 == 0:
            eval_acc = evaluate_acc()
            steplist.append(step)
            losslist.append(float(loss))
            acclist.append(eval_acc)

            if DEBUG is True:
                print('step: ', step, ' loss: ', float(loss), ' acc: ', acc)

    return steplist, losslist, acclist


def draw_train_result(train_step, train_loss, train_acc):
    """
    绘制训练结果
    :param train_step:
    :param train_loss:
    :param train_acc:
    :return:
    """
    steps = [i for i in train_step]

    # plt.plot(steps, train_loss, 'C1')
    plt.plot(steps, train_loss, color='C1', marker='s', label='loss')
    # plt.plot(steps, train_acc, 'C0')
    plt.plot(steps, train_acc, color='C0', marker='^', label='acc')

    plt.ylim([0, 1])
    plt.xlabel('train step')
    plt.ylabel('trian result')
    plt.legend()  # 子图说明
    plt.savefig('mnist_without_keras.svg')
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
    # 转换成张量
    img = 2 * tf.convert_to_tensor(img, dtype=tf.float32) / 255.0 - 1  # [0, 255]=>[-1.0, 1.0]
    img = tf.reshape(img, (-1, 28 * 28))

    return img


def predict(predict_imgs):
    """
    识别
    :param predict_imgs:
    :return:
    """
    out = calc_forward(predict_imgs)
    predict_result = tf.argmax(out, axis=1)

    return predict_result


"""训练主逻辑"""
# 设置显存按需增长
assert set_gpu_memory_growth()

# 设置全局超参数
DEBUG = True
BATCH_SIZE = 32
REPEAT_SIZE = 10

# 构建训练模型
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.05))  # 初始化为截断正太分布，标准差（均方差）
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.05))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.05))
b3 = tf.Variable(tf.zeros([10]))
LR = 0.01

# 准备数据集
(train_db, test_db) = make_dataset()

# 按批次训练
losslist = []
acclist = []
steplist = []

for step, (x, y) in enumerate(train_db):
    with tf.GradientTape() as tape:
        pred = calc_forward(x)
        loss = update_param(tape, pred, y)

        # 每500个step打印一下训练状态
        if step % 500 == 0:
            # 验证准去率
            acc = evaluate_acc()
            print('step: ', step, ' loss: ', float(loss), ' acc: ', acc)
            steplist.append(step)
            losslist.append(float(loss))
            acclist.append(acc)

# 绘制训练结果
draw_train_result(steplist, losslist, acclist)

# 测试
img_280x280_1 = get_img('280x280/1.png')
img_280x280_2 = get_img('280x280/2.png')
img_280x280_4 = get_img('280x280/4.png')
img_280x280_7 = get_img('280x280/7.png')
imgs_280x280 = tf.concat([img_280x280_1, img_280x280_2, img_280x280_4, img_280x280_7], axis=0)
result_280x280 = predict(imgs_280x280)
print(result_280x280.numpy())

img_20x20_1 = get_img('20x20/1.png')
img_20x20_2 = get_img('20x20/2.png')
img_20x20_4 = get_img('20x20/4.png')
img_20x20_7 = get_img('20x20/7.png')
imgs_20x20 = tf.concat([img_20x20_1, img_20x20_2, img_20x20_4, img_20x20_7], axis=0)
result_20x20 = predict(imgs_20x20)
print(result_20x20.numpy())
