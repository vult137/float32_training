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

    regular1_name = "regular_25_epochs.h5"
    regular2_name = "regular_50_epochs.h5"
    regular3_name = "regular_100_epochs.h5"
    regular4_name = "regular_150_epochs.h5"
    regular5_name = "regular_200_epochs.h5"
    dropout1_name = "dropout_25_epochs.h5"
    dropout2_name = "dropout_50_epochs.h5"
    dropout3_name = "dropout_100_epochs.h5"
    dropout4_name = "dropout_150_epochs.h5"
    dropout5_name = "dropout_200_epochs.h5"

    regular1_path = os.path.join(os.getcwd(), "saved_models", regular1_name)
    regular2_path = os.path.join(os.getcwd(), "saved_models", regular2_name)
    regular3_path = os.path.join(os.getcwd(), "saved_models", regular3_name)
    regular4_path = os.path.join(os.getcwd(), "saved_models", regular4_name)
    regular5_path = os.path.join(os.getcwd(), "saved_models", regular5_name)
    dropout1_path = os.path.join(os.getcwd(), "saved_models", dropout1_name)
    dropout2_path = os.path.join(os.getcwd(), "saved_models", dropout2_name)
    dropout3_path = os.path.join(os.getcwd(), "saved_models", dropout3_name)
    dropout4_path = os.path.join(os.getcwd(), "saved_models", dropout4_name)
    dropout5_path = os.path.join(os.getcwd(), "saved_models", dropout5_name)

    model1 = load_model(regular1_path)
    model2 = load_model(regular2_path)
    model3 = load_model(regular3_path)
    model4 = load_model(regular4_path)
    model5 = load_model(regular5_path)

    model6 = load_model(dropout1_path)
    model7 = load_model(dropout2_path)
    model8 = load_model(dropout3_path)
    model9 = load_model(dropout4_path)
    model10 = load_model(dropout5_path)

    shift = []

    print("epsilon: {0}, filter size: {1}, image batch: {2}".format(epsilon, filter_size, image_name))

    time_start = time.time()
    print("shift perturbation")
    x_test_shift_per = add_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    shift.append(
        robust_check_num_return(model1, x_test, y_test, x_test_shift_per, regular1_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model2, x_test, y_test, x_test_shift_per, regular2_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model3, x_test, y_test, x_test_shift_per, regular3_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model4, x_test, y_test, x_test_shift_per, regular4_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model5, x_test, y_test, x_test_shift_per, regular5_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model6, x_test, y_test, x_test_shift_per, dropout1_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model7, x_test, y_test, x_test_shift_per, dropout2_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model8, x_test, y_test, x_test_shift_per, dropout3_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model9, x_test, y_test, x_test_shift_per, dropout4_name, "shift perturbation"))
    shift.append(
        robust_check_num_return(model10, x_test, y_test, x_test_shift_per, dropout5_name, "shift perturbation"))

    random_shift = []

    time_start = time.time()
    print("random shift perturbation")
    x_test_random_shift_per = add_random_shift_per(x_test, epsilon=epsilon)
    time_end = time.time()
    print('time cost for adding random shift perturbation is', round(time_end - time_start, 3), 's')
    print("\n")

    random_shift.append(robust_check_num_return(model1, x_test, y_test, x_test_random_shift_per, regular1_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model2, x_test, y_test, x_test_random_shift_per, regular2_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model3, x_test, y_test, x_test_random_shift_per, regular3_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model4, x_test, y_test, x_test_random_shift_per, regular4_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model5, x_test, y_test, x_test_random_shift_per, regular5_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model6, x_test, y_test, x_test_random_shift_per, dropout1_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model7, x_test, y_test, x_test_random_shift_per, dropout2_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model8, x_test, y_test, x_test_random_shift_per, dropout3_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model9, x_test, y_test, x_test_random_shift_per, dropout4_name,
                                                "random shift perturbation"))
    random_shift.append(robust_check_num_return(model10, x_test, y_test, x_test_random_shift_per, dropout5_name,
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
        # data_batch[e] = do_test_return(e, "data_batch")
        test_batch[e] = do_test_return(e, "test_batch")
    print(test_batch)
    print(data_batch)

    # epsilons = [0.007, 0.009]
    # res = {}
    # for e in epsilons:
    #     res[e] = do_FGSM_test(e, "test_batch")
    # print(res)
