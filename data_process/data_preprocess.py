import os
import matplotlib.pyplot as plt
import numpy as np

regular_test_batch = {
    0.02:
        {'shift': [(0.702, 0.7, 0.00285), (0.78, 0.781, -0.00128), (0.668, 0.67, -0.00299), (0.69, 0.688, 0.0029),
                   (0.731, 0.73, 0.00137), (0.725, 0.725, 0.0), (0.711, 0.71, 0.00141), (0.731, 0.73, 0.00137)],
         'random_shift': [(0.702, 0.702, 0.0), (0.78, 0.778, 0.00256), (0.668, 0.665, 0.00449), (0.69, 0.688, 0.0029),
                          (0.731, 0.731, 0.0), (0.725, 0.724, 0.00138), (0.711, 0.709, 0.00281),
                          (0.731, 0.732, -0.00137)]},
    0.04:
        {'shift': [(0.702, 0.698, 0.0057), (0.78, 0.779, 0.00128), (0.668, 0.67, -0.00299), (0.69, 0.686, 0.0058),
                   (0.731, 0.729, 0.00274), (0.725, 0.725, 0.0), (0.711, 0.708, 0.00422), (0.731, 0.73, 0.00137)],
         'random_shift': [(0.702, 0.702, 0.0), (0.78, 0.766, 0.01795), (0.668, 0.652, 0.02395), (0.69, 0.682, 0.01159),
                          (0.731, 0.726, 0.00684), (0.725, 0.718, 0.00966), (0.711, 0.702, 0.01266),
                          (0.731, 0.724, 0.00958)]},
    0.06:
        {'shift': [(0.702, 0.695, 0.00997), (0.78, 0.776, 0.00513), (0.668, 0.671, -0.00449), (0.69, 0.68, 0.01449),
                   (0.731, 0.724, 0.00958), (0.725, 0.724, 0.00138), (0.711, 0.704, 0.00985), (0.731, 0.726, 0.00684)],
         'random_shift': [(0.702, 0.696, 0.00855), (0.78, 0.734, 0.05897), (0.668, 0.636, 0.0479),
                          (0.69, 0.654, 0.05217), (0.731, 0.723, 0.01094), (0.725, 0.708, 0.02345),
                          (0.711, 0.695, 0.0225), (0.731, 0.714, 0.02326)]},
    0.08:
        {'shift': [(0.702, 0.693, 0.01282), (0.78, 0.773, 0.00897), (0.668, 0.671, -0.00449), (0.69, 0.673, 0.02464),
                   (0.731, 0.721, 0.01368), (0.725, 0.722, 0.00414), (0.711, 0.697, 0.01969), (0.731, 0.723, 0.01094)],
         'random_shift': [(0.702, 0.684, 0.02564), (0.78, 0.683, 0.12436), (0.668, 0.615, 0.07934),
                          (0.69, 0.622, 0.09855), (0.731, 0.708, 0.03146), (0.725, 0.691, 0.0469),
                          (0.711, 0.68, 0.0436), (0.731, 0.7, 0.04241)]},
    0.1:
        {'shift': [(0.702, 0.689, 0.01852), (0.78, 0.769, 0.0141), (0.668, 0.669, -0.0015), (0.69, 0.667, 0.03333),
                   (0.731, 0.718, 0.01778), (0.725, 0.721, 0.00552), (0.711, 0.693, 0.02532), (0.731, 0.721, 0.01368)],
         'random_shift': [(0.702, 0.672, 0.04274), (0.78, 0.615, 0.21154), (0.668, 0.581, 0.13024),
                          (0.69, 0.572, 0.17101), (0.731, 0.693, 0.05198), (0.725, 0.672, 0.0731),
                          (0.711, 0.661, 0.07032), (0.731, 0.68, 0.06977)]},
    0.12:
        {'shift': [(0.702, 0.684, 0.02564), (0.78, 0.767, 0.01667), (0.668, 0.669, -0.0015), (0.69, 0.656, 0.04928),
                   (0.731, 0.712, 0.02599), (0.725, 0.716, 0.01241), (0.711, 0.686, 0.03516), (0.731, 0.718, 0.01778)],
         'random_shift': [(0.702, 0.653, 0.0698), (0.78, 0.545, 0.30128), (0.668, 0.545, 0.18413),
                          (0.69, 0.519, 0.24783), (0.731, 0.678, 0.0725), (0.725, 0.649, 0.10483),
                          (0.711, 0.639, 0.10127), (0.731, 0.652, 0.10807)]},
    0.14:
        {'shift': [(0.702, 0.679, 0.03276), (0.78, 0.763, 0.02179), (0.668, 0.667, 0.0015), (0.69, 0.648, 0.06087),
                   (0.731, 0.712, 0.02599), (0.725, 0.712, 0.01793), (0.711, 0.683, 0.03938), (0.731, 0.714, 0.02326)],
         'random_shift': [(0.702, 0.633, 0.09829), (0.78, 0.482, 0.38205), (0.668, 0.505, 0.24401),
                          (0.69, 0.462, 0.33043), (0.731, 0.648, 0.11354), (0.725, 0.623, 0.14069),
                          (0.711, 0.616, 0.13361), (0.731, 0.616, 0.15732)]},
    0.16:
        {'shift': [(0.702, 0.673, 0.04131), (0.78, 0.76, 0.02564), (0.668, 0.664, 0.00599), (0.69, 0.638, 0.07536),
                   (0.731, 0.706, 0.0342), (0.725, 0.707, 0.02483), (0.711, 0.676, 0.04923), (0.731, 0.712, 0.02599)],
         'random_shift': [(0.702, 0.61, 0.13105), (0.78, 0.424, 0.45641), (0.668, 0.47, 0.29641), (0.69, 0.414, 0.4),
                          (0.731, 0.62, 0.15185), (0.725, 0.598, 0.17517), (0.711, 0.592, 0.16737),
                          (0.731, 0.589, 0.19425)]},
    0.18:
        {'shift': [(0.702, 0.666, 0.05128), (0.78, 0.756, 0.03077), (0.668, 0.66, 0.01198), (0.69, 0.626, 0.09275),
                   (0.731, 0.701, 0.04104), (0.725, 0.703, 0.03034), (0.711, 0.669, 0.05907), (0.731, 0.706, 0.0342)],
         'random_shift': [(0.702, 0.584, 0.16809), (0.78, 0.38, 0.51282), (0.668, 0.43, 0.35629),
                          (0.69, 0.368, 0.46667), (0.731, 0.59, 0.19289), (0.725, 0.571, 0.21241),
                          (0.711, 0.564, 0.20675), (0.731, 0.558, 0.23666)]},
    0.2:
        {'shift': [(0.702, 0.658, 0.06268), (0.78, 0.754, 0.03333), (0.668, 0.655, 0.01946), (0.69, 0.615, 0.1087),
                   (0.731, 0.697, 0.04651), (0.725, 0.698, 0.03724), (0.711, 0.662, 0.06892), (0.731, 0.7, 0.04241)],
         'random_shift': [(0.702, 0.561, 0.20085), (0.78, 0.345, 0.55769), (0.668, 0.399, 0.40269),
                          (0.69, 0.341, 0.5058), (0.731, 0.553, 0.2435), (0.725, 0.543, 0.25103),
                          (0.711, 0.538, 0.24332), (0.731, 0.524, 0.28317)]}
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
    save_folder = os.path.join(os.getcwd(), "analyze_result")
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
    store_title = "compare {0} and {1} on {2} on {3}"\
        .format(model_name_1, model_name_2, per_name, dataset_name)\
        .replace(" ", "_").replace("_model", "").replace("_perturbation", "")
    save_folder = os.path.join(os.getcwd(), "analyze_compare")
    save_path = os.path.join(save_folder, store_title)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    epsilons = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

    model_num = 8
    epsilon_num = 10

    # acc_before:
    #   x_axis: different models
    #   y_axis: the model(x_axis) accuracy on test batch before adding any perturbations
    acc_before = []
    for i in range(model_num):
        acc_before.append(regular_test_batch[0.02]["shift"][i][0])

    # acc_after_shift
    #   key: model name
    #   value: an array
    #     x_axis: epsilon according to epsilons
    #     y_axis: accuracy of this model(key) after adding shift perturbation on epsilon(x_axis)
    acc_after_shift = {}
    for model_id in range(model_num):
        model_temp = []
        for epsilon in epsilons:
            model_temp.append(regular_test_batch[epsilon]["shift"][model_id][1])
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
            model_temp.append(regular_test_batch[epsilon]["random_shift"][model_id][1])
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
            model_temp.append(regular_test_batch[epsilon]["shift"][model_id][2])
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
            model_temp.append(regular_test_batch[epsilon]["random_shift"][model_id][2])
        drop_rate_rand[get_model_name(model_id)] = model_temp

    # for i in range(model_num):
    #     model_name = get_model_name(i)
    #     draw_acc_after_drop_rate(epsilons, acc_after_shift[model_name], drop_rate_shift[model_name], (0, 1),
    #                              "shift perturbation", model_name, "test_batch")
    #     draw_acc_after_drop_rate(epsilons, acc_after_rand[model_name], drop_rate_rand[model_name], (0, 1),
    #                              "stochastic perturbation", model_name, "test_batch")

    # draw_drop_rate_compare(epsilons, drop_rate_rand["dropout_model"], drop_rate_rand["regular_model"], (0, 0.8),
    #                        "stochastic perturbation", "dropout_model", "regular_model", "test_batch")

    # for i in range(model_num):
    #     if i == 1:
    #         continue
    #     model_name = get_model_name(i)
    #     draw_drop_rate_compare(epsilons, drop_rate_rand["dropout_model"], drop_rate_rand[model_name], (0, 0.8),
    #                            "stochastic perturbation", "dropout_model", model_name, "test_batch")
    #     draw_drop_rate_compare(epsilons, drop_rate_shift["dropout_model"], drop_rate_shift[model_name], (0, 0.2),
    #                            "shift perturbation", "dropout_model", model_name, "test_batch")

    X = [int(e * 100) for e in epsilons]
    ylim = (0, 0.8)
    plt.ylim(ylim)
    y1 = drop_rate_rand["dropout_model"]
    y2 = drop_rate_rand["regular_model"]
    y3 = drop_rate_rand["large_batch_model"]
    y4 = drop_rate_rand["200_epochs_model"]
    y5 = drop_rate_rand["batch_norm_model"]
    plt.xticks(X, epsilons)
    plt.plot(X, y1, label="dropout_model".replace("_", " "))
    plt.plot(X, y2, "r", label="regular_model".replace("_", " "))
    plt.plot(X, y3, "g", label="large_batch_model".replace("_", " "))
    plt.plot(X, y4, "purple", label="200_epochs_model".replace("_", " "))
    plt.plot(X, y5, "orange", label="batch_norm_model".replace("_", " "))

    plt.legend(loc="upper left")
    pic_title = "stochastic perturbation\n drop rate compare"
    plt.title(pic_title)
    store_title = "super compare stochastic"
    save_folder = os.path.join(os.getcwd(), "analyze_compare")
    save_path = os.path.join(save_folder, store_title)
    plt.savefig(save_path)
    plt.show()

    X = [int(e * 100) for e in epsilons]
    ylim = (0, 0.4)
    plt.ylim(ylim)
    y1 = drop_rate_shift["dropout_model"]
    y2 = drop_rate_shift["regular_model"]
    y3 = drop_rate_shift["large_batch_model"]
    y4 = drop_rate_shift["200_epochs_model"]
    y5 = drop_rate_shift["batch_norm_model"]
    plt.xticks(X, epsilons)
    plt.plot(X, y1, label="dropout_model".replace("_", " "))
    plt.plot(X, y2, "r", label="regular_model".replace("_", " "))
    plt.plot(X, y3, "g", label="large_batch_model".replace("_", " "))
    plt.plot(X, y4, "purple", label="200_epochs_model".replace("_", " "))
    plt.plot(X, y5, "orange", label="batch_norm_model".replace("_", " "))
    plt.legend(loc="upper left")
    pic_title = "shift perturbation\n drop rate compare"
    plt.title(pic_title)
    store_title = "super compare shift"
    save_folder = os.path.join(os.getcwd(), "analyze_compare")
    save_path = os.path.join(save_folder, store_title)
    plt.savefig(save_path)
    plt.show()