from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from utils.utils import unpickle, robust_check_num_return, check_robustness
from utils.image_processing import add_shift_per, add_random_shift_per, add_random_rate_per
import os
import time


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

    model1_name = "dropout_rate_005.h5"
    model2_name = "dropout_rate_010.h5"
    model3_name = "dropout_rate_015.h5"
    model4_name = "dropout_rate_020.h5"
    model5_name = "dropout_rate_025.h5"
    model6_name = "dropout_rate_030.h5"
    model7_name = "dropout_rate_035.h5"
    model8_name = "dropout_rate_040.h5"

    model1_path = os.path.join(os.getcwd(), "saved_models", model1_name)
    model2_path = os.path.join(os.getcwd(), "saved_models", model2_name)
    model3_path = os.path.join(os.getcwd(), "saved_models", model3_name)
    model4_path = os.path.join(os.getcwd(), "saved_models", model4_name)
    model5_path = os.path.join(os.getcwd(), "saved_models", model5_name)
    model6_path = os.path.join(os.getcwd(), "saved_models", model6_name)
    model7_path = os.path.join(os.getcwd(), "saved_models", model7_name)
    model8_path = os.path.join(os.getcwd(), "saved_models", model8_name)

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)
    model3 = load_model(model3_path)
    model4 = load_model(model4_path)
    model5 = load_model(model5_path)

    model6 = load_model(model6_path)
    model7 = load_model(model7_path)
    model8 = load_model(model8_path)

    shift = []

    print("epsilon: {0}, filter size: {1}, image batch: {2}".format(epsilon, filter_size, image_name))

    time_start = time.time()
    print("shift perturbation")
    x_test_shift_per = add_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    shift.append(
        robust_check_num_return(model1, x_test, y_test, x_test_shift_per, model1_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model2, x_test, y_test, x_test_shift_per, model2_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model3, x_test, y_test, x_test_shift_per, model3_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model4, x_test, y_test, x_test_shift_per, model4_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model5, x_test, y_test, x_test_shift_per, model5_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model6, x_test, y_test, x_test_shift_per, model6_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model7, x_test, y_test, x_test_shift_per, model7_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model8, x_test, y_test, x_test_shift_per, model8_name, "shift perturbation"))

    random_shift = []

    time_start = time.time()
    print("random shift perturbation")
    x_test_random_shift_per = add_random_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding random shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    random_shift.append(robust_check_num_return(model1, x_test, y_test, x_test_random_shift_per, model1_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model2, x_test, y_test, x_test_random_shift_per, model2_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model3, x_test, y_test, x_test_random_shift_per, model3_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model4, x_test, y_test, x_test_random_shift_per, model4_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model5, x_test, y_test, x_test_random_shift_per, model5_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model6, x_test, y_test, x_test_random_shift_per, model6_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model7, x_test, y_test, x_test_random_shift_per, model7_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model8, x_test, y_test, x_test_random_shift_per, model8_name,
                                                "random shift perturbation"))

    print("\n")
    whole_end = time.time()
    print('time cost for whole test return function of epsilon {0} is {1}s'
          .format(epsilon, round(whole_end - whole_start, 3)))
    print("===================================================================================================")
    return {"shift": shift, "random_shift": random_shift}


if __name__ == '__main__':
    test_batch = {}
    data_batch = {}
    epsilons = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    for e in epsilons:
        data_batch[e] = do_test_return(e, "data_batch")
        test_batch[e] = do_test_return(e, "test_batch")
    print(test_batch)
    print(data_batch)
