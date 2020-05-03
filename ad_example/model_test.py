from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from utils.utils import unpickle, robust_check_num_return
from utils.image_processing import add_shift_per, add_random_shift_per, add_random_rate_per
from utils.FGSM_perturbation import add_FGSM_per

import os
import time


def do_test_return(epsilon, test_data):
    classes = 10

    epsilon = epsilon
    rate = 50
    filter_size = (25, 25)
    image_name = test_data

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)

    regular_model_name = "cifar10_regular.h5"
    dropout_model_name = "cifar10_dropout.h5"
    large_batch_model_name = "cifar10_large_batch.h5"
    batch_norm_model_name = "cifar10_batch_norm.h5"
    epochs25_model_name = "cifar10_25_epochs.h5"
    epochs100_model_name = "cifar10_100_epochs.h5"
    epochs150_model_name = "cifar10_150_epochs.h5"
    epochs200_model_name = "cifar10_200_epochs.h5"

    regular_model_path = os.path.join(os.getcwd(), "saved_models", regular_model_name)
    dropout_model_path = os.path.join(os.getcwd(), "saved_models", dropout_model_name)
    large_batch_model_path = os.path.join(os.getcwd(), "saved_models", large_batch_model_name)
    batch_norm_model_path = os.path.join(os.getcwd(), "saved_models", batch_norm_model_name)
    epochs25_model_path = os.path.join(os.getcwd(), "saved_models", epochs25_model_name)
    epochs100_model_path = os.path.join(os.getcwd(), "saved_models", epochs100_model_name)
    epochs150_model_path = os.path.join(os.getcwd(), "saved_models", epochs150_model_name)
    epochs200_model_path = os.path.join(os.getcwd(), "saved_models", epochs200_model_name)

    model1 = load_model(regular_model_path)
    model2 = load_model(dropout_model_path)
    model3 = load_model(large_batch_model_path)
    model4 = load_model(batch_norm_model_path)
    model5 = load_model(epochs25_model_path)
    model6 = load_model(epochs100_model_path)
    model7 = load_model(epochs150_model_path)
    model8 = load_model(epochs200_model_path)

    shift = []

    print("epsilon: {0}, filter size: {1}, image batch: {2}, rate: {3}".format(epsilon, filter_size, image_name, rate))

    time_start = time.time()
    print("shift perturbation")
    x_test_shift_per = add_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    shift.append(
        robust_check_num_return(model1, x_test, y_test, x_test_shift_per, regular_model_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model2, x_test, y_test, x_test_shift_per, dropout_model_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model3, x_test, y_test, x_test_shift_per, large_batch_model_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model4, x_test, y_test, x_test_shift_per, batch_norm_model_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model5, x_test, y_test, x_test_shift_per, epochs25_model_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model6, x_test, y_test, x_test_shift_per, epochs100_model_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model7, x_test, y_test, x_test_shift_per, epochs150_model_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model8, x_test, y_test, x_test_shift_per, epochs200_model_name, "shift perturbation"))

    random_shift = []

    time_start = time.time()
    print("random shift perturbation")
    x_test_random_shift_per = add_random_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding random shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    random_shift.append(robust_check_num_return(model1, x_test, y_test, x_test_random_shift_per, regular_model_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model2, x_test, y_test, x_test_random_shift_per, dropout_model_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model3, x_test, y_test, x_test_random_shift_per, large_batch_model_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model4, x_test, y_test, x_test_random_shift_per, batch_norm_model_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model5, x_test, y_test, x_test_random_shift_per, epochs25_model_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model6, x_test, y_test, x_test_random_shift_per, epochs100_model_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model7, x_test, y_test, x_test_random_shift_per, epochs150_model_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model8, x_test, y_test, x_test_random_shift_per, epochs200_model_name,
                                                "random shift perturbation"))
    print("\n")

    random_rate = []

    time_start = time.time()
    print("random rate perturbation")
    x_test_random_rate_per = add_random_rate_per(x_test, epsilon=epsilon, rate=rate)
    time_end = time.time()
    print('time cost for adding random shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    random_rate.append(robust_check_num_return(model1, x_test, y_test, x_test_random_rate_per, regular_model_name,
                                               "random rate perturbation"))
    random_rate.append(robust_check_num_return(model2, x_test, y_test, x_test_random_rate_per, dropout_model_name,
                                               "random rate perturbation"))
    random_rate.append(robust_check_num_return(model3, x_test, y_test, x_test_random_rate_per, large_batch_model_name,
                                               "random rate perturbation"))
    random_rate.append(robust_check_num_return(model4, x_test, y_test, x_test_random_rate_per, batch_norm_model_name,
                                               "random rate perturbation"))
    random_rate.append(robust_check_num_return(model5, x_test, y_test, x_test_random_rate_per, epochs25_model_name,
                                               "random rate perturbation"))
    random_rate.append(robust_check_num_return(model6, x_test, y_test, x_test_random_rate_per, epochs100_model_name,
                                               "random rate perturbation"))
    random_rate.append(robust_check_num_return(model7, x_test, y_test, x_test_random_rate_per, epochs150_model_name,
                                               "random rate perturbation"))
    random_rate.append(robust_check_num_return(model8, x_test, y_test, x_test_random_rate_per, epochs200_model_name,
                                               "random rate perturbation"))
    print("===================================================================================================")
    return shift, random_shift, random_rate


if __name__ == '__main__':
    test_batch = []
    data_batch = []
    epsilons = []
    for epsilon in epsilons:
        test_batch.append(do_test_return(epsilon, "test_batch"))
        # data_batch.append(do_test_return(epsilon, "data_batch_2"))
    print(test_batch)
    print(data_batch)