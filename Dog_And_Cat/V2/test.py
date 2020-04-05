
import tools
import tensorflow as tf
import numpy as np

assert tools.set_gpu_memory_growth()


def get_img(img_path):
    """
    获取图片
    :param img_path:
    :return:
    """
    # 根据路径读取图片
    img = tf.io.read_file(img_path)
    # 解码图片，这里应该是解码成了png格式
    img = tf.image.decode_jpeg(img, channels=0)
    # 大小缩放
    img = tf.image.resize(img, [150, 150])
    # 转换成张量
    img = tf.cast(img, dtype=tf.float32) / 255.
    img = tf.expand_dims(img, 0)

    return img


predict_model = tf.keras.models.load_model('predict_model.h5')

# 测试网上下载的图片
print("--------------------")
imgs = [get_img('../dataset/dowload_from_network/' + str(i+1) + '.jpg') for i in range(6)]
imgs = tf.concat(imgs, axis=0)
results = predict_model.predict(imgs)
print(results)
print(np.argmax(results, axis=1))
