from tensorflow import keras
from tensorflow.keras.models import load_model
from utils.utils import check_robustness
from utils.image_processing import *
import numpy as np
import os
import time


def do_test(epsilon, test_data, rate):
    classes = 10

    epsilon = epsilon
    filter_size = (25, 25)
    image_name = test_data
    image_path = os.path.join(os.getcwd(), "cifar-10-batches-py", image_name)

    batch_image = unpickle(image_path)
    x_test = image_trans(batch_image)
    y_test = np.array(batch_image[b"labels"])
    y_test = keras.utils.to_categorical(y_test, classes)

    regular_model_name = "cifar10_regular.h5"
    dropout_model_name = "cifar10_dropout.h5"
    large_batch_model_name = "cifar10_large_batch.h5"
    adadelta_model_name = "cifar10_adadelta.h5"
    batch_norm_model_name = "cifar10_batch_norm.h5"
    epochs100_model_name = "cifar10_100_epochs.h5"
    epochs150_model_name = "cifar10_150_epochs.h5"

    regular_model_path = os.path.join(os.getcwd(), "saved_models", regular_model_name)
    dropout_model_path = os.path.join(os.getcwd(), "saved_models", dropout_model_name)
    large_batch_model_path = os.path.join(os.getcwd(), "saved_models", large_batch_model_name)
    adadelta_model_path = os.path.join(os.getcwd(), "saved_models", adadelta_model_name)
    batch_norm_model_path = os.path.join(os.getcwd(), "saved_models", batch_norm_model_name)
    epochs100_model_path = os.path.join(os.getcwd(), "saved_models", epochs100_model_name)
    epochs150_model_path = os.path.join(os.getcwd(), "saved_models", epochs150_model_name)

    model1 = load_model(regular_model_path)
    model2 = load_model(dropout_model_path)
    model3 = load_model(large_batch_model_path)
    model4 = load_model(adadelta_model_path)
    model5 = load_model(batch_norm_model_path)
    model6 = load_model(epochs100_model_path)
    model7 = load_model(epochs150_model_path)

    print("epsilon: {0}, filter size: {1}, image batch: {2}, rate: {3}".format(epsilon, filter_size, image_name, rate))

    time_start = time.time()
    print("shift perturbation")
    x_test_shift_per = add_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    check_robustness(model1, x_test, y_test, x_test_shift_per, regular_model_name, "shift perturbation")
    check_robustness(model2, x_test, y_test, x_test_shift_per, dropout_model_name, "shift perturbation")
    check_robustness(model3, x_test, y_test, x_test_shift_per, large_batch_model_name, "shift perturbation")
    check_robustness(model4, x_test, y_test, x_test_shift_per, adadelta_model_name, "shift perturbation")
    check_robustness(model5, x_test, y_test, x_test_shift_per, batch_norm_model_name, "shift perturbation")
    check_robustness(model6, x_test, y_test, x_test_shift_per, epochs100_model_name, "shift perturbation")
    check_robustness(model7, x_test, y_test, x_test_shift_per, epochs150_model_name, "shift perturbation")
    print("\n")

    time_start = time.time()
    print("random shift perturbation")
    x_test_random_shift_per = add_random_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding random shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    check_robustness(model1, x_test, y_test, x_test_random_shift_per, regular_model_name,
                     "random shift perturbation")
    check_robustness(model2, x_test, y_test, x_test_random_shift_per, dropout_model_name,
                     "random shift perturbation")
    check_robustness(model3, x_test, y_test, x_test_random_shift_per, large_batch_model_name,
                     "random shift perturbation")
    check_robustness(model4, x_test, y_test, x_test_random_shift_per, adadelta_model_name,
                     "random shift perturbation")
    check_robustness(model5, x_test, y_test, x_test_random_shift_per, batch_norm_model_name,
                     "random shift perturbation")
    check_robustness(model6, x_test, y_test, x_test_random_shift_per, epochs100_model_name,
                     "random shift perturbation")
    check_robustness(model7, x_test, y_test, x_test_random_shift_per, epochs150_model_name,
                     "random shift perturbation")
    print("\n")

    time_start = time.time()
    print("random rate perturbation")
    x_test_random_rate_per = add_random_rate_per(x_test, epsilon=epsilon, rate=rate)
    time_end = time.time()
    print('time cost for adding random shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    check_robustness(model1, x_test, y_test, x_test_random_rate_per, regular_model_name,
                     "random rate perturbation")
    check_robustness(model2, x_test, y_test, x_test_random_rate_per, dropout_model_name,
                     "random rate perturbation")
    check_robustness(model3, x_test, y_test, x_test_random_rate_per, large_batch_model_name,
                     "random rate perturbation")
    check_robustness(model4, x_test, y_test, x_test_random_rate_per, adadelta_model_name,
                     "random rate perturbation")
    check_robustness(model5, x_test, y_test, x_test_random_rate_per, batch_norm_model_name,
                     "random rate perturbation")
    check_robustness(model6, x_test, y_test, x_test_random_rate_per, epochs100_model_name,
                     "random rate perturbation")
    check_robustness(model7, x_test, y_test, x_test_random_rate_per, epochs150_model_name,
                     "random rate perturbation")

    print("===================================================================================================")


def do_simple_test(epsilon, test_data):
    print("This is the model robustness check for the simple networks")
    classes = 10

    epsilon = epsilon
    filter_size = (25, 25)
    image_name = test_data
    image_path = os.path.join(os.getcwd(), "cifar-10-batches-py", image_name)

    batch_image = unpickle(image_path)
    x_test = image_trans(batch_image)
    y_test = np.array(batch_image[b"labels"])
    y_test = keras.utils.to_categorical(y_test, classes)

    regular_model_name = "simple_regular.h5"
    dropout_model_name = "simple_dropout.h5"
    large_batch_model_name = "simple_large_batch.h5"
    batch_norm_model_name = "simple_batch_norm.h5"
    epochs150_model_name = "simple_150_epochs.h5"

    regular_model_path = os.path.join(os.getcwd(), "saved_models", "simple", regular_model_name)
    dropout_model_path = os.path.join(os.getcwd(), "saved_models", "simple", dropout_model_name)
    large_batch_model_path = os.path.join(os.getcwd(), "saved_models", "simple", large_batch_model_name)
    batch_norm_model_path = os.path.join(os.getcwd(), "saved_models", "simple", batch_norm_model_name)
    epochs150_model_path = os.path.join(os.getcwd(), "saved_models", "simple", epochs150_model_name)

    model1 = load_model(regular_model_path)
    model2 = load_model(dropout_model_path)
    model3 = load_model(large_batch_model_path)
    model5 = load_model(batch_norm_model_path)
    model7 = load_model(epochs150_model_path)

    print("epsilon: {0}, filter size: {1}, image batch: {2}".format(epsilon, filter_size, image_name))

    time_start = time.time()
    print("shift perturbation")
    x_test_shift_per = add_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    check_robustness(model1, x_test, y_test, x_test_shift_per, regular_model_name, "shift perturbation")
    check_robustness(model2, x_test, y_test, x_test_shift_per, dropout_model_name, "shift perturbation")
    check_robustness(model3, x_test, y_test, x_test_shift_per, large_batch_model_name, "shift perturbation")
    check_robustness(model5, x_test, y_test, x_test_shift_per, batch_norm_model_name, "shift perturbation")
    check_robustness(model7, x_test, y_test, x_test_shift_per, epochs150_model_name, "shift perturbation")
    print("\n")

    time_start = time.time()
    print("random shift perturbation")
    x_test_random_shift_per = add_random_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding random shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    check_robustness(model1, x_test, y_test, x_test_random_shift_per, regular_model_name,
                     "random shift perturbation")
    check_robustness(model2, x_test, y_test, x_test_random_shift_per, dropout_model_name,
                     "random shift perturbation")
    check_robustness(model3, x_test, y_test, x_test_random_shift_per, large_batch_model_name,
                     "random shift perturbation")
    check_robustness(model5, x_test, y_test, x_test_random_shift_per, batch_norm_model_name,
                     "random shift perturbation")
    check_robustness(model7, x_test, y_test, x_test_random_shift_per, epochs150_model_name,
                     "random shift perturbation")

    print("===================================================================================================")


if __name__ == '__main__':
    pass
    # do_test(10, "test_batch")
    # do_test(15, "test_batch")
    # do_test(20, "test_batch")
    # do_test(25, "test_batch")
    # do_test(30, "test_batch")
    # do_test(40, "test_batch")
    # do_test(50, "test_batch")

    # do_test(10, "data_batch_2")
    # do_test(15, "data_batch_2")
    # do_test(20, "data_batch_2")
    # do_test(25, "data_batch_2")
    # do_test(30, "data_batch_2")
    # do_test(40, "data_batch_2")
    # do_test(50, "data_batch_2")


    # test for the simple lenet models
    # do_simple_test(10, "test_batch")
    # do_simple_test(15, "test_batch")
    # do_simple_test(20, "test_batch")
    # do_simple_test(25, "test_batch")
    # do_simple_test(30, "test_batch")
    # do_simple_test(40, "test_batch")
    # do_simple_test(50, "test_batch")
    #
    # do_simple_test(10, "data_batch_1")
    # do_simple_test(15, "data_batch_1")
    # do_simple_test(20, "data_batch_1")
    # do_simple_test(25, "data_batch_1")
    # do_simple_test(30, "data_batch_1")
    # do_simple_test(40, "data_batch_1")
    # do_simple_test(50, "data_batch_1")
