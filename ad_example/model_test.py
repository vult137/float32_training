from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from utils.utils import unpickle, robust_check_num_return, check_robustness
from utils.image_processing import add_shift_per, add_random_shift_per, add_random_rate_per
from utils.FGSM_perturbation import add_FGSM_per

import os
import time


def FGSM_test(x_test, y_test, model, model_name, epsilon):
    x_test_FGSM_per = add_FGSM_per(x_test, y_test, model, epsilon)
    return robust_check_num_return(model, x_test, y_test, x_test_FGSM_per, model_name, "FGSM perturbation")


def do_FGSM_test(epsilon, test_data):
    classes = 10

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

    start = time.time()

    FGSM = [FGSM_test(x_test, y_test, model1, regular_model_name, epsilon),
            FGSM_test(x_test, y_test, model2, dropout_model_name, epsilon),
            FGSM_test(x_test, y_test, model3, large_batch_model_name, epsilon),
            FGSM_test(x_test, y_test, model4, batch_norm_model_name, epsilon),
            FGSM_test(x_test, y_test, model5, epochs25_model_name, epsilon),
            FGSM_test(x_test, y_test, model6, epochs100_model_name, epsilon),
            FGSM_test(x_test, y_test, model7, epochs150_model_name, epsilon),
            FGSM_test(x_test, y_test, model8, epochs200_model_name, epsilon)]

    end = time.time()

    print("time cost for all FGSM adding: {}".format(end - start))

    return FGSM


def do_test_return(epsilon, test_data):
    whole_start = time.time()
    classes = 10
    filter_size = (25, 25)
    image_name = test_data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)

    if image_name == "data_batch":
        x_test = x_train[0:10000]
        y_test = y_train[0:10000]
    elif image_name == "test_batch":
        pass
    else:
        raise ValueError

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

    print("epsilon: {0}, filter size: {1}, image batch: {2}".format(epsilon, filter_size, image_name))

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
    print("===================================================================================================")

    whole_end = time.time()
    print('time cost for whole test return function of epsilon {0} is {1}s'
          .format(epsilon, round(whole_end - whole_start, 3)))

    return {"shift": shift, "random_shift": random_shift}


def do_rate_test_return(epsilon, rate, test_data):
    classes = 10
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
    print("\n")


if __name__ == '__main__':
    test_batch = {}
    data_batch = {}
    epsilons = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    for e in epsilons:
        data_batch[e] = do_test_return(e, "data_batch")
    print(test_batch)
    print(data_batch)

    # epsilons = [0.001, 0.003, 0.008, 0.1]
    # res = {}
    # for e in epsilons:
    #     res[e] = do_FGSM_test(e, "test_batch")
    # print(res)
