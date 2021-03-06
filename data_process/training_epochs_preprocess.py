import os
import matplotlib.pyplot as plt
import  numpy as np


training_epochs_test_batch = {0.02: {
    'shift': [(0.731, 0.73, 0.00137), (0.702, 0.7, 0.00285), (0.725, 0.725, 0.0), (0.711, 0.71, 0.00141),
              (0.731, 0.73, 0.00137), (0.757, 0.756, 0.00132), (0.78, 0.781, -0.00128), (0.798, 0.798, 0.0),
              (0.759, 0.757, 0.00264), (0.66, 0.659, 0.00152)],
    'random_shift': [(0.731, 0.731, 0.0), (0.702, 0.702, 0.0), (0.725, 0.723, 0.00276), (0.711, 0.708, 0.00422),
                     (0.731, 0.729, 0.00274), (0.757, 0.757, 0.0), (0.78, 0.778, 0.00256), (0.798, 0.796, 0.00251),
                     (0.759, 0.754, 0.00659), (0.66, 0.608, 0.07879)]}, 0.04: {
    'shift': [(0.731, 0.729, 0.00274), (0.702, 0.698, 0.0057), (0.725, 0.725, 0.0), (0.711, 0.708, 0.00422),
              (0.731, 0.73, 0.00137), (0.757, 0.755, 0.00264), (0.78, 0.779, 0.00128), (0.798, 0.798, 0.0),
              (0.759, 0.755, 0.00527), (0.66, 0.655, 0.00758)],
    'random_shift': [(0.731, 0.729, 0.00274), (0.702, 0.701, 0.00142), (0.725, 0.718, 0.00966), (0.711, 0.706, 0.00703),
                     (0.731, 0.724, 0.00958), (0.757, 0.751, 0.00793), (0.78, 0.768, 0.01538), (0.798, 0.767, 0.03885),
                     (0.759, 0.73, 0.03821), (0.66, 0.511, 0.22576)]}, 0.06: {
    'shift': [(0.731, 0.724, 0.00958), (0.702, 0.695, 0.00997), (0.725, 0.724, 0.00138), (0.711, 0.704, 0.00985),
              (0.731, 0.726, 0.00684), (0.757, 0.752, 0.00661), (0.78, 0.776, 0.00513), (0.798, 0.798, 0.0),
              (0.759, 0.751, 0.01054), (0.66, 0.652, 0.01212)],
    'random_shift': [(0.731, 0.72, 0.01505), (0.702, 0.696, 0.00855), (0.725, 0.71, 0.02069), (0.711, 0.693, 0.02532),
                     (0.731, 0.715, 0.02189), (0.757, 0.734, 0.03038), (0.78, 0.733, 0.06026), (0.798, 0.714, 0.10526),
                     (0.759, 0.675, 0.11067), (0.66, 0.42, 0.36364)]}, 0.08: {
    'shift': [(0.731, 0.721, 0.01368), (0.702, 0.693, 0.01282), (0.725, 0.722, 0.00414), (0.711, 0.697, 0.01969),
              (0.731, 0.723, 0.01094), (0.757, 0.75, 0.00925), (0.78, 0.773, 0.00897), (0.798, 0.797, 0.00125),
              (0.759, 0.749, 0.01318), (0.66, 0.648, 0.01818)],
    'random_shift': [(0.731, 0.713, 0.02462), (0.702, 0.688, 0.01994), (0.725, 0.693, 0.04414), (0.711, 0.682, 0.04079),
                     (0.731, 0.704, 0.03694), (0.757, 0.7, 0.0753), (0.78, 0.685, 0.12179), (0.798, 0.624, 0.21805),
                     (0.759, 0.601, 0.20817), (0.66, 0.35, 0.4697)]}, 0.1: {
    'shift': [(0.731, 0.718, 0.01778), (0.702, 0.689, 0.01852), (0.725, 0.721, 0.00552), (0.711, 0.693, 0.02532),
              (0.731, 0.721, 0.01368), (0.757, 0.748, 0.01189), (0.78, 0.769, 0.0141), (0.798, 0.795, 0.00376),
              (0.759, 0.748, 0.01449), (0.66, 0.643, 0.02576)],
    'random_shift': [(0.731, 0.698, 0.04514), (0.702, 0.677, 0.03561), (0.725, 0.673, 0.07172), (0.711, 0.662, 0.06892),
                     (0.731, 0.677, 0.07387), (0.757, 0.656, 0.13342), (0.78, 0.616, 0.21026), (0.798, 0.52, 0.34837),
                     (0.759, 0.522, 0.31225), (0.66, 0.314, 0.52424)]}, 0.12: {
    'shift': [(0.731, 0.712, 0.02599), (0.702, 0.684, 0.02564), (0.725, 0.716, 0.01241), (0.711, 0.686, 0.03516),
              (0.731, 0.718, 0.01778), (0.757, 0.744, 0.01717), (0.78, 0.767, 0.01667), (0.798, 0.794, 0.00501),
              (0.759, 0.746, 0.01713), (0.66, 0.638, 0.03333)],
    'random_shift': [(0.731, 0.676, 0.07524), (0.702, 0.652, 0.07123), (0.725, 0.646, 0.10897), (0.711, 0.632, 0.11111),
                     (0.731, 0.65, 0.11081), (0.757, 0.619, 0.1823), (0.78, 0.549, 0.29615), (0.798, 0.418, 0.47619),
                     (0.759, 0.447, 0.41107), (0.66, 0.284, 0.5697)]}, 0.14: {
    'shift': [(0.731, 0.712, 0.02599), (0.702, 0.679, 0.03276), (0.725, 0.712, 0.01793), (0.711, 0.683, 0.03938),
              (0.731, 0.714, 0.02326), (0.757, 0.741, 0.02114), (0.78, 0.763, 0.02179), (0.798, 0.791, 0.00877),
              (0.759, 0.745, 0.01845), (0.66, 0.632, 0.04242)],
    'random_shift': [(0.731, 0.649, 0.11218), (0.702, 0.633, 0.09829), (0.725, 0.626, 0.13655), (0.711, 0.619, 0.1294),
                     (0.731, 0.623, 0.14774), (0.757, 0.568, 0.24967), (0.78, 0.485, 0.37821), (0.798, 0.346, 0.56642),
                     (0.759, 0.385, 0.49275), (0.66, 0.258, 0.60909)]}, 0.16: {
    'shift': [(0.731, 0.706, 0.0342), (0.702, 0.673, 0.04131), (0.725, 0.707, 0.02483), (0.711, 0.676, 0.04923),
              (0.731, 0.712, 0.02599), (0.757, 0.736, 0.02774), (0.78, 0.76, 0.02564), (0.798, 0.787, 0.01378),
              (0.759, 0.741, 0.02372), (0.66, 0.626, 0.05152)],
    'random_shift': [(0.731, 0.619, 0.15321), (0.702, 0.613, 0.12678), (0.725, 0.6, 0.17241), (0.711, 0.595, 0.16315),
                     (0.731, 0.592, 0.19015), (0.757, 0.522, 0.31044), (0.78, 0.428, 0.45128), (0.798, 0.297, 0.62782),
                     (0.759, 0.34, 0.55204), (0.66, 0.23, 0.65152)]}, 0.18: {
    'shift': [(0.731, 0.701, 0.04104), (0.702, 0.666, 0.05128), (0.725, 0.703, 0.03034), (0.711, 0.669, 0.05907),
              (0.731, 0.706, 0.0342), (0.757, 0.732, 0.03303), (0.78, 0.756, 0.03077), (0.798, 0.783, 0.0188),
              (0.759, 0.738, 0.02767), (0.66, 0.62, 0.06061)],
    'random_shift': [(0.731, 0.59, 0.19289), (0.702, 0.585, 0.16667), (0.725, 0.571, 0.21241), (0.711, 0.562, 0.20956),
                     (0.731, 0.56, 0.23393), (0.757, 0.48, 0.36592), (0.78, 0.381, 0.51154), (0.798, 0.251, 0.68546),
                     (0.759, 0.292, 0.61528), (0.66, 0.201, 0.69545)]}, 0.2: {
    'shift': [(0.731, 0.697, 0.04651), (0.702, 0.658, 0.06268), (0.725, 0.698, 0.03724), (0.711, 0.662, 0.06892),
              (0.731, 0.7, 0.04241), (0.757, 0.727, 0.03963), (0.78, 0.754, 0.03333), (0.798, 0.782, 0.02005),
              (0.759, 0.737, 0.02899), (0.66, 0.612, 0.07273)],
    'random_shift': [(0.731, 0.558, 0.23666), (0.702, 0.563, 0.19801), (0.725, 0.539, 0.25655), (0.711, 0.54, 0.24051),
                     (0.731, 0.527, 0.27907), (0.757, 0.44, 0.41876), (0.78, 0.343, 0.56026), (0.798, 0.228, 0.71429),
                     (0.759, 0.265, 0.65086), (0.66, 0.171, 0.74091)]}}


def get_model_name(m_id):
    if m_id == 0:
        return "regular_25"
    if m_id == 1:
        return "regular_50"
    if m_id == 2:
        return "regular_100"
    if m_id == 3:
        return "regular_150"
    if m_id == 4:
        return "regular_200"
    if m_id == 5:
        return "dropout_25"
    if m_id == 6:
        return "dropout_50"
    if m_id == 7:
        return "dropout_100"
    if m_id == 8:
        return "dropout_150"
    if m_id == 9:
        return "dropout_200"


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
    save_folder = os.path.join(os.getcwd(), "training_epochs_analyze_result")
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
    save_folder = os.path.join(os.getcwd(), "training_epochs_analyze_compare")
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
        acc_before.append(training_epochs_test_batch[0.02]["shift"][i][0])

    # acc_after_shift
    #   key: model name
    #   value: an array
    #     x_axis: epsilon according to epsilons
    #     y_axis: accuracy of this model(key) after adding shift perturbation on epsilon(x_axis)
    acc_after_shift = {}
    for model_id in range(model_num):
        model_temp = []
        for epsilon in epsilons:
            model_temp.append(training_epochs_test_batch[epsilon]["shift"][model_id][1])
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
            model_temp.append(training_epochs_test_batch[epsilon]["random_shift"][model_id][1])
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
            model_temp.append(training_epochs_test_batch[epsilon]["shift"][model_id][2])
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
            model_temp.append(training_epochs_test_batch[epsilon]["random_shift"][model_id][2])
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
    #
    # for i in range(model_num):
    #     model_name = get_model_name(i)
    #     draw_acc_after_drop_rate(epsilons, acc_after_rand[model_name], drop_rate_rand[model_name], (0, 1),
    #                              "stochastic perturbation", model_name, "test_batch")

    # X = [int(e * 100) for e in epsilons]
    # ylim = (0, 0.8)
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
    # pic_title = "drop rate compare"
    # plt.title(pic_title)
    # store_title = "super compare"
    # save_folder = os.path.join(os.getcwd(), "training_epochs_analyze_compare")
    # save_path = os.path.join(save_folder, store_title)
    # plt.savefig(save_path)
    # plt.show()

    # X = [int(e * 100) for e in epsilons]
    # ylim = (0, 0.8)
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
    # save_folder = os.path.join(os.getcwd(), "training_epochs_analyze_compare")
    # save_path = os.path.join(save_folder, store_title)
    # plt.savefig(save_path)
    # plt.show()
    #
    # X = [int(e * 100) for e in epsilons]
    # ylim = (0, 0.8)
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
    # save_folder = os.path.join(os.getcwd(), "training_epochs_analyze_compare")
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
    # save_folder = os.path.join(os.getcwd(), "training_epochs_analyze_compare")
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
    # save_folder = os.path.join(os.getcwd(), "training_epochs_analyze_compare")
    # save_path = os.path.join(save_folder, store_title)
    # plt.savefig(save_path)
    # plt.show()

    x = np.arange(5)
    y1 = acc_before[0:5]
    y2 = acc_before[5:10]
    total_width, n = 0.8, 2
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, y1, width=width, label="regular")
    plt.bar(x + width, y2, width=width, label="dropout")
    plt.xticks(x, (25, 50, 100, 150, 200))
    plt.legend()
    plt.savefig("training epochs_acc_before")
    plt.show()
