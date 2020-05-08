import os
import matplotlib.pyplot as plt
import numpy as np

data_size_test_batch = {
    0.02: {
        'shift': [(0.602, 0.599, 0.00498), (0.641, 0.641, 0.0), (0.682, 0.68, 0.00293), (0.701, 0.701, 0.0),
                  (0.726, 0.726, 0.0), (0.656, 0.653, 0.00457), (0.735, 0.735, 0.0), (0.756, 0.756, 0.0),
                  (0.776, 0.777, -0.00129), (0.785, 0.787, -0.00255)],
        'random_shift': [(0.602, 0.603, -0.00166), (0.641, 0.639, 0.00312), (0.682, 0.681, 0.00147),
                         (0.701, 0.701, 0.0),
                         (0.726, 0.725, 0.00138), (0.656, 0.656, 0.0), (0.735, 0.733, 0.00272), (0.756, 0.756, 0.0),
                         (0.776, 0.774, 0.00258), (0.785, 0.782, 0.00382)]},
    0.04: {
        'shift': [(0.602, 0.596, 0.00997), (0.641, 0.641, 0.0), (0.682, 0.68, 0.00293), (0.701, 0.7, 0.00143),
                  (0.726, 0.724, 0.00275), (0.656, 0.652, 0.0061), (0.735, 0.735, 0.0), (0.756, 0.756, 0.0),
                  (0.776, 0.778, -0.00258), (0.785, 0.787, -0.00255)],
        'random_shift': [(0.602, 0.603, -0.00166), (0.641, 0.64, 0.00156), (0.682, 0.676, 0.0088),
                         (0.701, 0.698, 0.00428),
                         (0.726, 0.72, 0.00826), (0.656, 0.658, -0.00305), (0.735, 0.731, 0.00544),
                         (0.756, 0.754, 0.00265),
                         (0.776, 0.765, 0.01418), (0.785, 0.762, 0.0293)]},
    0.06: {
        'shift': [(0.602, 0.59, 0.01993), (0.641, 0.64, 0.00156), (0.682, 0.677, 0.00733), (0.701, 0.699, 0.00285),
                  (0.726, 0.722, 0.00551), (0.656, 0.647, 0.01372), (0.735, 0.735, 0.0), (0.756, 0.755, 0.00132),
                  (0.776, 0.778, -0.00258), (0.785, 0.787, -0.00255)],
        'random_shift': [(0.602, 0.599, 0.00498), (0.641, 0.635, 0.00936), (0.682, 0.672, 0.01466),
                         (0.701, 0.693, 0.01141),
                         (0.726, 0.713, 0.01791), (0.656, 0.655, 0.00152), (0.735, 0.714, 0.02857),
                         (0.756, 0.74, 0.02116),
                         (0.776, 0.738, 0.04897), (0.785, 0.717, 0.08662)]},
    0.08: {
        'shift': [(0.602, 0.584, 0.0299), (0.641, 0.637, 0.00624), (0.682, 0.674, 0.01173), (0.701, 0.695, 0.00856),
                  (0.726, 0.72, 0.00826), (0.656, 0.643, 0.01982), (0.735, 0.734, 0.00136), (0.756, 0.753, 0.00397),
                  (0.776, 0.78, -0.00515), (0.785, 0.785, 0.0)],
        'random_shift': [(0.602, 0.599, 0.00498), (0.641, 0.633, 0.01248), (0.682, 0.663, 0.02786),
                         (0.701, 0.687, 0.01997),
                         (0.726, 0.704, 0.0303), (0.656, 0.65, 0.00915), (0.735, 0.685, 0.06803),
                         (0.756, 0.711, 0.05952),
                         (0.776, 0.702, 0.09536), (0.785, 0.656, 0.16433)]},
    0.1: {
        'shift': [(0.602, 0.58, 0.03654), (0.641, 0.632, 0.01404), (0.682, 0.673, 0.0132), (0.701, 0.692, 0.01284),
                  (0.726, 0.717, 0.0124), (0.656, 0.635, 0.03201), (0.735, 0.732, 0.00408), (0.756, 0.752, 0.00529),
                  (0.776, 0.781, -0.00644), (0.785, 0.784, 0.00127)],
        'random_shift': [(0.602, 0.595, 0.01163), (0.641, 0.621, 0.0312), (0.682, 0.651, 0.04545),
                         (0.701, 0.67, 0.04422),
                         (0.726, 0.687, 0.05372), (0.656, 0.635, 0.03201), (0.735, 0.647, 0.11973),
                         (0.756, 0.671, 0.11243),
                         (0.776, 0.652, 0.15979), (0.785, 0.588, 0.25096)]},
    0.12: {
        'shift': [(0.602, 0.571, 0.0515), (0.641, 0.627, 0.02184), (0.682, 0.669, 0.01906), (0.701, 0.689, 0.01712),
                  (0.726, 0.714, 0.01653), (0.656, 0.628, 0.04268), (0.735, 0.73, 0.0068), (0.756, 0.748, 0.01058),
                  (0.776, 0.777, -0.00129), (0.785, 0.781, 0.0051)],
        'random_shift': [(0.602, 0.588, 0.02326), (0.641, 0.61, 0.04836), (0.682, 0.633, 0.07185),
                         (0.701, 0.653, 0.06847),
                         (0.726, 0.673, 0.073), (0.656, 0.614, 0.06402), (0.735, 0.594, 0.19184),
                         (0.756, 0.62, 0.17989),
                         (0.776, 0.598, 0.22938), (0.785, 0.52, 0.33758)]},
    0.14: {
        'shift': [(0.602, 0.562, 0.06645), (0.641, 0.62, 0.03276), (0.682, 0.664, 0.02639), (0.701, 0.686, 0.0214),
                  (0.726, 0.71, 0.02204), (0.656, 0.622, 0.05183), (0.735, 0.726, 0.01224), (0.756, 0.745, 0.01455),
                  (0.776, 0.775, 0.00129), (0.785, 0.779, 0.00764)],
        'random_shift': [(0.602, 0.581, 0.03488), (0.641, 0.595, 0.07176), (0.682, 0.613, 0.10117),
                         (0.701, 0.639, 0.08845),
                         (0.726, 0.648, 0.10744), (0.656, 0.587, 0.10518), (0.735, 0.549, 0.25306),
                         (0.756, 0.562, 0.25661),
                         (0.776, 0.542, 0.30155), (0.785, 0.464, 0.40892)]},
    0.16: {
        'shift': [(0.602, 0.552, 0.08306), (0.641, 0.613, 0.04368), (0.682, 0.658, 0.03519), (0.701, 0.683, 0.02568),
                  (0.726, 0.706, 0.02755), (0.656, 0.613, 0.06555), (0.735, 0.722, 0.01769), (0.756, 0.74, 0.02116),
                  (0.776, 0.772, 0.00515), (0.785, 0.776, 0.01146)],
        'random_shift': [(0.602, 0.575, 0.04485), (0.641, 0.571, 0.1092), (0.682, 0.593, 0.1305),
                         (0.701, 0.619, 0.11698),
                         (0.726, 0.621, 0.14463), (0.656, 0.557, 0.15091), (0.735, 0.494, 0.32789),
                         (0.756, 0.508, 0.32804),
                         (0.776, 0.486, 0.37371), (0.785, 0.417, 0.46879)]},
    0.18: {
        'shift': [(0.602, 0.543, 0.09801), (0.641, 0.606, 0.0546), (0.682, 0.652, 0.04399), (0.701, 0.679, 0.03138),
                  (0.726, 0.703, 0.03168), (0.656, 0.604, 0.07927), (0.735, 0.718, 0.02313), (0.756, 0.734, 0.0291),
                  (0.776, 0.77, 0.00773), (0.785, 0.771, 0.01783)],
        'random_shift': [(0.602, 0.559, 0.07143), (0.641, 0.56, 0.12637), (0.682, 0.576, 0.15543),
                         (0.701, 0.595, 0.15121),
                         (0.726, 0.591, 0.18595), (0.656, 0.52, 0.20732), (0.735, 0.446, 0.3932),
                         (0.756, 0.457, 0.3955),
                         (0.776, 0.442, 0.43041), (0.785, 0.38, 0.51592)]},
    0.2: {
        'shift': [(0.602, 0.535, 0.1113), (0.641, 0.599, 0.06552), (0.682, 0.648, 0.04985), (0.701, 0.674, 0.03852),
                  (0.726, 0.7, 0.03581), (0.656, 0.593, 0.09604), (0.735, 0.712, 0.03129), (0.756, 0.731, 0.03307),
                  (0.776, 0.766, 0.01289), (0.785, 0.767, 0.02293)],
        'random_shift': [(0.602, 0.554, 0.07973), (0.641, 0.538, 0.16069), (0.682, 0.551, 0.19208),
                         (0.701, 0.576, 0.17832),
                         (0.726, 0.561, 0.22727), (0.656, 0.49, 0.25305), (0.735, 0.4, 0.45578),
                         (0.756, 0.412, 0.45503),
                         (0.776, 0.4, 0.48454), (0.785, 0.346, 0.55924)]}}


def get_model_name(m_id):
    if m_id == 0:
        return "regular_1"
    if m_id == 1:
        return "regular_2"
    if m_id == 2:
        return "regular_3"
    if m_id == 3:
        return "regular_4"
    if m_id == 4:
        return "regular_5"
    if m_id == 5:
        return "dropout_1"
    if m_id == 6:
        return "dropout_2"
    if m_id == 7:
        return "dropout_3"
    if m_id == 8:
        return "dropout_4"
    if m_id == 9:
        return "dropout_5"


def draw_acc_after_drop_rate(epsilons, y1, y2, ylim, per_name, model_name, dataset_name):
    X = [int(e * 100) for e in epsilons]
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
    save_folder = os.path.join(os.getcwd(), "data_size_analyze_result")
    save_path = os.path.join(save_folder, store_title)
    plt.savefig(save_path)
    plt.show()


def draw_drop_rate_compare(X, y1, y2, ylim, per_name, model_name_1, model_name_2, dataset_name):
    X = [int(e * 100) for e in epsilons]
    plt.ylim(ylim)
    plt.xticks(X, epsilons)
    plt.plot(X, y1, label=model_name_1.replace("_", " "))
    plt.plot(X, y2, "r", label=model_name_2.replace("_", " "))
    plt.legend(loc="upper left")
    pic_title = "{0} and {1} drop rate".format(model_name_1, model_name_2).replace("_", " ")
    plt.title(pic_title)
    store_title = "compare {0} and {1} on {2} on {3}" \
        .format(model_name_1, model_name_2, per_name, dataset_name) \
        .replace(" ", "_").replace("_model", "").replace("_perturbation", "")
    save_folder = os.path.join(os.getcwd(), "data_size_analyze_compare")
    save_path = os.path.join(save_folder, store_title)
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    epsilons = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

    model_num = 10
    epsilon_num = 10

    # acc_before:
    #   x_axis: different models
    #   y_axis: the model(x_axis) accuracy on test batch before adding any perturbations
    acc_before = []
    for i in range(model_num):
        acc_before.append(data_size_test_batch[0.02]["shift"][i][0])

    # acc_after_shift
    #   key: model name
    #   value: an array
    #     x_axis: epsilon according to epsilons
    #     y_axis: accuracy of this model(key) after adding shift perturbation on epsilon(x_axis)
    acc_after_shift = {}
    for model_id in range(model_num):
        model_temp = []
        for epsilon in epsilons:
            model_temp.append(data_size_test_batch[epsilon]["shift"][model_id][1])
        acc_after_shift[get_model_name(model_id)] = model_temp

    # acc_after_rand
    #   key: model name
    #   value: an array
    #     x_axis: epsilon according to epsilons
    #     y_axis: accuracy of this model(key) after adding stochastic perturbation on epsilon(x_axis)
    acc_after_rand = {}
    for model_id in range(model_num):
        model_temp = []
        for epsilon in epsilons:
            model_temp.append(data_size_test_batch[epsilon]["random_shift"][model_id][1])
        acc_after_rand[get_model_name(model_id)] = model_temp

    # drop_rate_shift
    #   key: model name
    #   value: an array
    #     x_axis: epsilon according to epsilons
    #     y_axis: drop rate of the model(key) after adding shift perturbation on epsilon(x_axis)
    drop_rate_shift = {}
    for model_id in range(model_num):
        model_temp = []
        for epsilon in epsilons:
            model_temp.append(data_size_test_batch[epsilon]["shift"][model_id][2])
        drop_rate_shift[get_model_name(model_id)] = model_temp

    # drop_rate_rand
    #   key: model name
    #   value: an array
    #     x_axis: epsilon according to epsilons
    #     y_axis: drop rate of the model(key) after adding stochastic perturbation on epsilon(x_axis)
    drop_rate_rand = {}
    for model_id in range(model_num):
        model_temp = []
        for epsilon in epsilons:
            model_temp.append(data_size_test_batch[epsilon]["random_shift"][model_id][2])
        drop_rate_rand[get_model_name(model_id)] = model_temp

    print(acc_before)
    print(acc_after_rand)
    print(acc_after_shift)
    print(drop_rate_rand)
    print(drop_rate_shift)

    # for i in range(model_num):
    #     model_name = get_model_name(i)
    #     draw_acc_after_drop_rate(epsilons, acc_after_shift[model_name], drop_rate_shift[model_name], (0, 1),
    #                              "shift perturbation", model_name, "test_batch")

    # for i in range(model_num):
    #     model_name = get_model_name(i)
    #     draw_acc_after_drop_rate(epsilons, acc_after_rand[model_name], drop_rate_rand[model_name], (0, 1),
    #                              "stochastic perturbation", model_name, "test_batch")

    # X = [int(e * 100) for e in epsilons]
    # ylim = (0, 0.6)
    # plt.ylim(ylim)
    # y1 = drop_rate_rand[get_model_name(5)]
    # y2 = drop_rate_rand[get_model_name(6)]
    # y3 = drop_rate_rand[get_model_name(7)]
    # y4 = drop_rate_rand[get_model_name(8)]
    # y5 = drop_rate_rand[get_model_name(9)]
    # plt.xticks(X, epsilons)
    # plt.plot(X, y1, label=get_model_name(5).replace("_", " "))
    # plt.plot(X, y2, "r", label=get_model_name(6).replace("_", " "))
    # plt.plot(X, y3, "g", label=get_model_name(7).replace("_", " "))
    # plt.plot(X, y4, "yellow", label=get_model_name(8).replace("_", " "))
    # plt.plot(X, y5, "purple", label=get_model_name(9).replace("_", " "))
    # plt.legend(loc="upper left")
    # pic_title = "stochastic perturbation drop rate compare dropout"
    # plt.title(pic_title)
    # store_title = "super compare stochastic dropout"
    # save_folder = os.path.join(os.getcwd(), "data_size_analyze_compare")
    # save_path = os.path.join(save_folder, store_title)
    # plt.savefig(save_path)
    # plt.show()
    #
    # X = [int(e * 100) for e in epsilons]
    # ylim = (0, 0.6)
    # plt.ylim(ylim)
    # y1 = drop_rate_rand[get_model_name(0)]
    # y2 = drop_rate_rand[get_model_name(1)]
    # y3 = drop_rate_rand[get_model_name(2)]
    # y4 = drop_rate_rand[get_model_name(3)]
    # y5 = drop_rate_rand[get_model_name(4)]
    # plt.xticks(X, epsilons)
    # plt.plot(X, y1, label=get_model_name(0).replace("_", " "))
    # plt.plot(X, y2, "r", label=get_model_name(1).replace("_", " "))
    # plt.plot(X, y3, "g", label=get_model_name(2).replace("_", " "))
    # plt.plot(X, y4, "yellow", label=get_model_name(3).replace("_", " "))
    # plt.plot(X, y5, "purple", label=get_model_name(4).replace("_", " "))
    # plt.legend(loc="upper left")
    # pic_title = "stochastic perturbation drop rate compare regular"
    # plt.title(pic_title)
    # store_title = "super compare stochastic regular"
    # save_folder = os.path.join(os.getcwd(), "data_size_analyze_compare")
    # save_path = os.path.join(save_folder, store_title)
    # plt.savefig(save_path)
    # plt.show()
    #
    # print("========================================================")
    #
    # X = [int(e * 100) for e in epsilons]
    # ylim = (0, 0.2)
    # plt.ylim(ylim)
    # y1 = drop_rate_shift[get_model_name(5)]
    # y2 = drop_rate_shift[get_model_name(6)]
    # y3 = drop_rate_shift[get_model_name(7)]
    # y4 = drop_rate_shift[get_model_name(8)]
    # y5 = drop_rate_shift[get_model_name(9)]
    # plt.xticks(X, epsilons)
    # plt.plot(X, y1, label=get_model_name(5).replace("_", " "))
    # plt.plot(X, y2, "r", label=get_model_name(6).replace("_", " "))
    # plt.plot(X, y3, "g", label=get_model_name(7).replace("_", " "))
    # plt.plot(X, y4, "yellow", label=get_model_name(8).replace("_", " "))
    # plt.plot(X, y5, "purple", label=get_model_name(9).replace("_", " "))
    # plt.legend(loc="upper left")
    # pic_title = "shift perturbation drop rate compare dropout"
    # plt.title(pic_title)
    # store_title = "super compare shift dropout"
    # save_folder = os.path.join(os.getcwd(), "data_size_analyze_compare")
    # save_path = os.path.join(save_folder, store_title)
    # plt.savefig(save_path)
    # plt.show()
    #
    # X = [int(e * 100) for e in epsilons]
    # ylim = (0, 0.2)
    # plt.ylim(ylim)
    # y1 = drop_rate_shift[get_model_name(0)]
    # y2 = drop_rate_shift[get_model_name(1)]
    # y3 = drop_rate_shift[get_model_name(2)]
    # y4 = drop_rate_shift[get_model_name(3)]
    # y5 = drop_rate_shift[get_model_name(4)]
    # plt.xticks(X, epsilons)
    # plt.plot(X, y1, label=get_model_name(0).replace("_", " "))
    # plt.plot(X, y2, "r", label=get_model_name(1).replace("_", " "))
    # plt.plot(X, y3, "g", label=get_model_name(2).replace("_", " "))
    # plt.plot(X, y4, "yellow", label=get_model_name(3).replace("_", " "))
    # plt.plot(X, y5, "purple", label=get_model_name(4).replace("_", " "))
    # plt.legend(loc="upper left")
    # pic_title = "shift perturbation drop rate compare regular"
    # plt.title(pic_title)
    # store_title = "super compare shift regular"
    # save_folder = os.path.join(os.getcwd(), "data_size_analyze_compare")
    # save_path = os.path.join(save_folder, store_title)
    # plt.savefig(save_path)
    # plt.show()

    print("========================================================")


    x = np.arange(5)
    y1 = acc_before[0:5]
    y2 = acc_before[5:10]
    total_width, n = 0.8, 2
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, y1, width=width, label="regular")
    plt.bar(x + width, y2, width=width, label="dropout")
    plt.xticks(x, (10000, 20000, 30000, 40000, 50000))
    plt.legend()
    plt.savefig("data_size_acc_before")
    plt.show()
