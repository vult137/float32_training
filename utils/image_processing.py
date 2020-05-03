from PIL import Image

import numpy as np

import os
import random
from utils.utils import unpickle, image_trans


def add_shift_per(images, epsilon):
    minus_flag = False
    perturbation = 1.0 * epsilon
    if perturbation > 1.0:
        return images
    if perturbation < 0.0:
        perturbation = abs(perturbation)
        minus_flag = True
    perturbation = np.float32(perturbation)
    res = images.copy()
    for x in np.nditer(res, op_flags=["readwrite"]):
        if minus_flag:
            if x < perturbation:
                pass
            else:
                x -= perturbation
        else:
            if x > 1.0 - int(perturbation):
                pass
            else:
                x += perturbation
    return res


def add_random_shift_per(images, epsilon):
    if epsilon > 1.0:
        return images
    if epsilon < 0.0:
        epsilon = abs(epsilon)
    epsilon = np.float32(epsilon)
    res = images.copy()
    for x in np.nditer(res, op_flags=["readwrite"]):
        perturbation = np.random.randint(0, epsilon * 255)
        perturbation = np.float32(perturbation)
        perturbation = perturbation / 255
        minus_flag = bool(random.getrandbits(1))
        if minus_flag:
            if x < perturbation:
                pass
            else:
                x -= perturbation
        else:
            if x > 1.0 - int(perturbation):
                pass
            else:
                x += perturbation
    return res


def add_random_rate_per(images, epsilon, rate):
    if epsilon > 1.:
        return images
    if epsilon < 0.:
        epsilon = abs(epsilon)
    epsilon = np.float32(epsilon)
    res = images.copy()
    for x in np.nditer(res, op_flags=["readwrite"]):
        dice = np.random.randint(0, 100)
        if dice > rate:
            continue
        perturbation = np.random.randint(0, epsilon)
        perturbation = np.float32(perturbation)
        perturbation = perturbation / 255
        minus_flag = bool(random.getrandbits(1))
        if minus_flag:
            if x < perturbation:
                pass
            else:
                x -= perturbation
        else:
            if x > 1. - int(perturbation):
                pass
            else:
                x += perturbation
    return res


def add_filter_shift_per(images, filter_shape, epsilon):
    minus_flag = False
    if epsilon > 1.:
        return images
    if epsilon < 0.:
        minus_flag = True
        epsilon = abs(epsilon)
    image_shape = images[0].shape
    x_len, y_len, color_channel = image_shape[0], image_shape[1], image_shape[2]
    x_f, y_f = filter_shape[0], filter_shape[1]
    if x_f > x_len or y_f > y_len:
        raise ValueError
    x_start = np.random.randint(0, x_len - x_f)
    y_start = np.random.randint(0, y_len - y_f)
    res = images.copy()
    perturbation = np.float32(epsilon)
    for i in range(len(res)):
        for x in range(x_start, x_f):
            for y in range(y_start, y_f):
                for z in range(3):
                    tmp = res[i][x][y][z]
                    if minus_flag:
                        if tmp < perturbation:
                            pass
                        else:
                            res[i][x][y][z] -= perturbation
                    else:
                        if tmp > 1. - float(perturbation):
                            pass
                        else:
                            res[i][x][y][z] += perturbation
    return res


if __name__ == '__main__':
    image_name = "data_batch_1"
    image_path = os.path.join(os.getcwd(), "cifar-10-batches-py", image_name)

    batch_image = unpickle(image_path)
    x_train = image_trans(batch_image)
    y_train = batch_image[b"labels"]

    per_batch = x_train[0:50]
    index = 20
    i = Image.fromarray(per_batch[index])
    i.show()

    per_batch = add_random_rate_per(per_batch, epsilon=20, rate=30)
    i = Image.fromarray(per_batch[index])
    i.show()

    # per_batch = x_train[0:10]
    # print(per_batch[0])
    # print("==============================")
    # per_batch = add_filter_shift_per(per_batch, filter_shape=(5, 5), start=(0, 0), epsilon=25)
    # print(per_batch[0])
