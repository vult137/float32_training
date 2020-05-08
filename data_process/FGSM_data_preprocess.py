import os
import matplotlib.pyplot as plt
import numpy as np

FGSM_test_batch = {
    0.001: [(0.702, 0.653, 0.0698), (0.78, 0.712, 0.08718), (0.668, 0.633, 0.0524), (0.69, 0.508, 0.26377),
            (0.731, 0.678, 0.0725), (0.725, 0.664, 0.08414), (0.711, 0.652, 0.08298), (0.731, 0.67, 0.08345)],
    0.002: [(0.702, 0.598, 0.14815), (0.78, 0.642, 0.17692), (0.668, 0.598, 0.10479), (0.69, 0.354, 0.48696),
            (0.731, 0.624, 0.14637), (0.725, 0.602, 0.16966), (0.711, 0.586, 0.17581), (0.731, 0.609, 0.16689)],
    0.003: [(0.702, 0.548, 0.21937), (0.78, 0.581, 0.25513), (0.668, 0.565, 0.15419), (0.69, 0.233, 0.66232),
            (0.731, 0.571, 0.21888), (0.725, 0.545, 0.24828), (0.711, 0.533, 0.25035), (0.731, 0.548, 0.25034)],
    0.004: [(0.702, 0.495, 0.29487), (0.78, 0.527, 0.32436), (0.668, 0.534, 0.2006), (0.69, 0.148, 0.78551),
            (0.731, 0.522, 0.28591), (0.725, 0.493, 0.32), (0.711, 0.478, 0.32771), (0.731, 0.488, 0.33242)],
    0.005: [(0.702, 0.446, 0.36467), (0.78, 0.476, 0.38974), (0.668, 0.497, 0.25599), (0.69, 0.094, 0.86377),
            (0.731, 0.473, 0.35294), (0.725, 0.436, 0.39862), (0.711, 0.43, 0.39522), (0.731, 0.432, 0.40903)],
    0.006: [(0.702, 0.394, 0.43875), (0.78, 0.43, 0.44872), (0.668, 0.463, 0.30689), (0.69, 0.058, 0.91594),
            (0.731, 0.423, 0.42134), (0.725, 0.386, 0.46759), (0.711, 0.382, 0.46273), (0.731, 0.388, 0.46922)],
    0.007: [(0.702, 0.353, 0.49715), (0.78, 0.391, 0.49872), (0.668, 0.432, 0.35329), (0.69, 0.034, 0.95072),
            (0.731, 0.38, 0.48016), (0.725, 0.34, 0.53103), (0.711, 0.341, 0.52039), (0.731, 0.357, 0.51163)],
    0.008: [(0.702, 0.317, 0.54843), (0.78, 0.356, 0.54359), (0.668, 0.402, 0.3982), (0.69, 0.02, 0.97101),
            (0.731, 0.341, 0.53352), (0.725, 0.295, 0.5931), (0.711, 0.299, 0.57947), (0.731, 0.341, 0.53352)],
    0.009: [(0.702, 0.279, 0.60256), (0.78, 0.325, 0.58333), (0.668, 0.374, 0.44012), (0.69, 0.012, 0.98261),
            (0.731, 0.303, 0.5855), (0.725, 0.259, 0.64276), (0.711, 0.263, 0.6301), (0.731, 0.334, 0.54309)],
    0.010: [(0.702, 0.246, 0.64957), (0.78, 0.298, 0.61795), (0.668, 0.345, 0.48353), (0.69, 0.007, 0.98986),
            (0.731, 0.269, 0.63201), (0.725, 0.23, 0.68276), (0.711, 0.232, 0.6737), (0.731, 0.33, 0.54856)]
}


def get_model_name(m_id):
    if m_id == 0:
        return "regular_model"
    if m_id == 1:
        return "dropout_model"
    if m_id == 2:
        return "large_batch_model"
    if m_id == 3:
        return "batch_norm_model"
    if m_id == 4:
        return "25_epochs_model"
    if m_id == 5:
        return "100_epochs_model"
    if m_id == 6:
        return "150_epochs_model"
    if m_id == 7:
        return "200_epochs_model"


def draw_acc_after_drop_rate(epsilons, y1, y2, ylim, per_name, model_name, dataset_name):
    X = [int(e * 1000) for e in epsilons]
    plt.ylim(ylim)
    plt.xticks(X, epsilons)
    plt.bar(X, y1, label="accuracy after per")
    plt.plot(X, y2, 'r', label="drop rate after per")
    plt.legend(loc="upper left")
    pic_title = "acc and drop rate of {0} after \n {1} on {2}".format(model_name, per_name, dataset_name)
    pic_title = pic_title.replace("_", " ")
    plt.title(pic_title)
    store_title = "{0} {1} {2}".format(model_name, per_name, dataset_name)
    store_title = store_title.replace("_model", "").replace(" perturbation", "").replace(" ", "_")
    save_folder = os.path.join(os.getcwd(), "FGSM_analyze_result")
    save_path = os.path.join(save_folder, store_title)
    plt.savefig(save_path)
    plt.show()


def draw_drop_rate_compare(X, y1, y2, ylim, per_name, model_name_1, model_name_2, dataset_name):
    X = [int(e * 1000) for e in epsilons]
    plt.ylim(ylim)
    plt.xticks(X, epsilons)
    plt.plot(X, y1, label=model_name_1.replace("_", " "))
    plt.plot(X, y2, "r", label=model_name_2.replace("_", " "))
    plt.legend(loc="upper left")
    pic_title = "{0} and {1} drop rate".format(model_name_1, model_name_2).replace("_", " ")
    plt.title(pic_title)
    store_title = "compare {0} and {1} on {2} on {3}"\
        .format(model_name_1, model_name_2, per_name, dataset_name)\
        .replace(" ", "_").replace("_model", "").replace("_perturbation", "")
    save_folder = os.path.join(os.getcwd(), "FGSM_analyze_compare")
    save_path = os.path.join(save_folder, store_title)
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    epsilons = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
    model_num = 8
    epsilon_num = 10
    acc_before = []
    for i in range(model_num):
        acc_before.append(FGSM_test_batch[0.01][i][0])

    # acc_after_shift
    #   key: model name
    #   value: an array
    #     x_axis: epsilon according to epsilons
    #     y_axis: accuracy of this model(key) after adding FGSM perturbation on epsilon(x_axis)
    acc_after = {}
    for model_id in range(model_num):
        model_temp = []
        for epsilon in epsilons:
            model_temp.append(FGSM_test_batch[epsilon][model_id][1])
        acc_after[get_model_name(model_id)] = model_temp

    # drop_rate_shift
    #   key: model name
    #   value: an array
    #     x_axis: epsilon according to epsilons
    #     y_axis: drop rate of the model(key) after adding FGSM perturbation on epsilon(x_axis)
    drop_rate = {}
    for model_id in range(model_num):
        model_temp = []
        for epsilon in epsilons:
            model_temp.append(FGSM_test_batch[epsilon][model_id][2])
        drop_rate[get_model_name(model_id)] = model_temp

    # for i in range(model_num):
    #     model_name = get_model_name(i)
    #     draw_acc_after_drop_rate(epsilons, acc_after[model_name], drop_rate[model_name], (0, 1),
    #                              "FGSM perturbation", model_name, "test_batch")

    # for i in range(model_num):
    #     if i == 1:
    #         continue
    #     model_name = get_model_name(i)
    #     draw_drop_rate_compare(epsilons, drop_rate["dropout_model"], drop_rate[model_name], (0, 1),
    #                            "FGSM perturbation", "dropout_model", model_name, "test_batch")


    X = [int(e * 1000) for e in epsilons]
    ylim = (0, 1)
    plt.ylim(ylim)
    y1 = drop_rate["dropout_model"]
    y2 = drop_rate["regular_model"]
    y3 = drop_rate["large_batch_model"]
    y4 = drop_rate["200_epochs_model"]
    y5 = drop_rate["batch_norm_model"]
    plt.xticks(X, epsilons)
    plt.plot(X, y1, label="dropout_model".replace("_", " "))
    plt.plot(X, y2, "r", label="regular_model".replace("_", " "))
    plt.plot(X, y3, "g", label="large_batch_model".replace("_", " "))
    plt.plot(X, y4, "purple", label="200_epochs_model".replace("_", " "))
    plt.plot(X, y5, "orange", label="batch_norm_model".replace("_", " "))

    plt.legend(loc="upper left")
    pic_title = "FGSM perturbation\n drop rate compare"
    plt.title(pic_title)
    store_title = "super compare FGSM"
    save_folder = os.path.join(os.getcwd(), "FGSM_analyze_compare")
    save_path = os.path.join(save_folder, store_title)
    plt.savefig(save_path)
    plt.show()
