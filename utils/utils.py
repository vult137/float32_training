import pickle
import numpy as np
import tensorflow as tf


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def image_trans(data_batch):
    num_train_samples = 10000
    x_data = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_data = np.empty((num_train_samples,), dtype='uint8')
    x_data[:, :, :, :] = data_batch[b"data"].reshape(x_data.shape)
    y_data[:] = data_batch[b"labels"]
    x_data = np.transpose(x_data, (0, 2, 3, 1))
    return x_data


def image_predict(images, model):
    res_list = []
    prediction = model.predict(images)
    for p in prediction:
        res_list.append(np.argmax(p))
    return res_list


def check_robustness(model, x_test, y_test, x_test_per, model_name, perturbation_name):
    score_before = model.evaluate(x_test, y_test, verbose=0)
    score_before = round(score_before[1], 3)
    print("accuracy of {0} before adding {1}: {2}".format(model_name, perturbation_name, score_before))
    score_after = model.evaluate(x_test_per, y_test, verbose=0)
    score_after = round(score_after[1], 3)
    print("accuracy of {0} after adding {1}: {2}".format(model_name, perturbation_name, score_after))
    reduce = round((score_before - score_after) / score_before, 5)
    print("reduce percent: {}%".format(round(reduce * 100, 2)))


def robust_check_num_return(model, x_test, y_test, x_test_per, model_name, perturbation_name):
    score_before = model.evaluate(x_test, y_test, verbose=0)
    score_before = round(score_before[1], 3)
    print("accuracy of {0} before adding {1}: {2}".format(model_name, perturbation_name, score_before))
    score_after = model.evaluate(x_test_per, y_test, verbose=0)
    score_after = round(score_after[1], 3)
    print("accuracy of {0} after adding {1}: {2}".format(model_name, perturbation_name, score_after))
    reduce = round((score_before - score_after) / score_before, 5)
    print("reduce percent: {}%".format(round(reduce * 100, 2)))
    return score_before, score_after, reduce


def compare_diff(a, b):
    if len(a) != len(b):
        return -1
    res = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            res += 1
    return res


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image[None, ...]
    return image
