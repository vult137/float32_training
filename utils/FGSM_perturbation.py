import tensorflow as tf
import numpy as np
import os
from utils.utils import image_trans, unpickle
import time


def add_FGSM_per(images, labels, model, epsilon):
    if epsilon > 1. or epsilon < 0.:
        return images
    res = images.copy()
    for i in range(len(res)):
        perturbation = create_adversarial_pattern(images[i], labels[i], model)
        res[i] = images[i] + perturbation * epsilon
        res[i] = np.float32(res[i])
    return res


def create_adversarial_pattern(input_image, input_label, model):
    input_image = input_image.reshape(1, 32, 32, 3)
    input_image = tf.cast(input_image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.MSE(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    grad_sign = tf.sign(gradient)
    return grad_sign


if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), "..", "lenet_dataset_size", "saved_models", "cifar10_dropout_1.h5")
    model_path = os.path.abspath(model_path)
    model = tf.keras.models.load_model(model_path)

    classes = 10
    image_name = "test_batch"
    image_path = os.path.join(os.getcwd(), "..", "cifar-10-batches-py", image_name)
    image_path = os.path.abspath(image_path)
    batch_image = unpickle(image_path)
    x_test = image_trans(batch_image)
    y_test = np.array(batch_image[b"labels"])
    y_test = tf.keras.utils.to_categorical(y_test, classes)

    print(x_test[20])
    time_start = time.time()
    x_test_FGSM_per = add_FGSM_per(x_test, y_test, model, 5.12)
    time_end = time.time()
    print('time cost for adding FGSM perturbation is', round(time_end - time_start, 3), 's')
    print(x_test_FGSM_per[20])
