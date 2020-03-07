# !usr/bin/python
# -*- coding:utf-8 -*-

"""A linear regression demo"""
__author__ = 'huhuwa@2019-12-13'

import numpy as np
import matplotlib.pyplot as plt


def mse(w, b, points):
    """
    计算均方差
    :param w:
    :param b:
    :param points:
    :return:
    """
    total_error = 0

    for point in points:
        total_error += (point[1] - (w * point[0] + b)) ** 2

    # 返回均方差
    return total_error / float(len(points))


def step_gradient(w_current, b_current, points, lr):
    """
    通过梯度更新一次w和b的值
    :param w_current:
    :param b_current:
    :param points:
    :param lr:学习率
    :return:
    """
    w_gradient, b_gradient = 0, 0       # 当前w和b的梯度
    total_count = len(points)           # 当前样本总数

    for point in points:
        # 求误差函数对w的导数
        w_gradient += (2 / float(total_count)) * ((w_current * point[0] + b_current) - point[1]) * point[0]
        # 求误差函数对b的导数
        b_gradient += (2 / float(total_count)) * ((w_current * point[0] + b_current) - point[1])

    # 根据梯度下降算法更新w和b
    new_w = w_current - lr * w_gradient
    new_b = b_current - lr * b_gradient

    return [new_w, new_b]


def gradient_descent(points, starting_w, starting_b, lr, iterations):
    """
    根据梯度下降算法更新w和b
    :param points:
    :param starting_w:
    :param starting_b:
    :param lr:
    :param iterations:
    :return:
    """
    w, b = starting_w, starting_b

    for item in range(iterations):
        # 对所有的数据根据均方误差更新一次w和b
        w, b = step_gradient(w, b, np.array(points), lr)
        # 计算当前的均方误差，用于监控训练进度
        loss = mse(w, b, points)
        train_result.append([loss, w, b])
        # 对所有数据训练100次打印一次训练结果
        if item % 50 == 0:
            print(f"Iteration: {item}, loss: {loss}, w: {w}, b: {b}")

    # 返回最后的w和b
    return [w, b]


# 初始化超参数
learn_rate = 0.01
b_initial, w_initial = 0, 0
num_iterations = 1000
train_result = []

# 准备数据样本
data = []
for i in range(100):
    # 获取样本值
    x = np.random.uniform(-10., 10.)
    # 给样本添加均值为0，方差为0.1×0.1的高斯噪声
    eps = np.random.normal(0., 0.1)
    # 获取样本标签
    y = 1.5 * x + 0.1 + eps  # 故意让样本点在直线两侧分布
    # 保存样本数据
    data.append([x, y])
# 将样本数据转换为numpy数组
data = np.array(data)


# 通过梯度下降算法求最优解
[w_final, b_final] = gradient_descent(data, w_initial, b_initial, learn_rate, num_iterations)

# 计算最优解的均方差
loss_final = mse(w_final, b_final, data)
print(f"Final loss: {loss_final}, w: {w_final}, b: {b_final}")
train_result = np.array(train_result)
loss_all = train_result[:, 0]
best_index = np.argmin(loss_all)
best_loss, best_w, best_b = train_result[int(best_index)]
print(f'Best result epoch: {best_index + 1}, loss: {best_loss}, w: {best_w}, b: {best_b}')


# 绘制loss
X = np.linspace(0, 1000, 1000, endpoint=True)
plt.plot(X, loss_all)
plt.show()
