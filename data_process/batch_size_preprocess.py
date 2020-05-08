import os
import matplotlib.pyplot as plt
import numpy as np

batch_size_test_batch = {0.02: {
    'shift': [(0.732, 0.73, 0.00273), (0.708, 0.709, -0.00141), (0.708, 0.706, 0.00282), (0.68, 0.677, 0.00441),
              (0.677, 0.679, -0.00295), (0.782, 0.783, -0.00128), (0.769, 0.768, 0.0013), (0.774, 0.773, 0.00129),
              (0.718, 0.716, 0.00279), (0.68, 0.679, 0.00147)],
    'random_shift': [(0.732, 0.73, 0.00273), (0.708, 0.706, 0.00282), (0.708, 0.705, 0.00424), (0.68, 0.682, -0.00294),
                     (0.677, 0.674, 0.00443), (0.782, 0.78, 0.00256), (0.769, 0.76, 0.0117), (0.774, 0.771, 0.00388),
                     (0.718, 0.716, 0.00279), (0.68, 0.677, 0.00441)]}, 0.04: {
    'shift': [(0.732, 0.729, 0.0041), (0.708, 0.71, -0.00282), (0.708, 0.706, 0.00282), (0.68, 0.671, 0.01324),
              (0.677, 0.678, -0.00148), (0.782, 0.782, 0.0), (0.769, 0.768, 0.0013), (0.774, 0.772, 0.00258),
              (0.718, 0.715, 0.00418), (0.68, 0.675, 0.00735)],
    'random_shift': [(0.732, 0.727, 0.00683), (0.708, 0.702, 0.00847), (0.708, 0.705, 0.00424), (0.68, 0.684, -0.00588),
                     (0.677, 0.666, 0.01625), (0.782, 0.758, 0.03069), (0.769, 0.725, 0.05722), (0.774, 0.767, 0.00904),
                     (0.718, 0.706, 0.01671), (0.68, 0.677, 0.00441)]}, 0.06: {
    'shift': [(0.732, 0.726, 0.0082), (0.708, 0.71, -0.00282), (0.708, 0.704, 0.00565), (0.68, 0.664, 0.02353),
              (0.677, 0.677, 0.0), (0.782, 0.78, 0.00256), (0.769, 0.768, 0.0013), (0.774, 0.77, 0.00517),
              (0.718, 0.713, 0.00696), (0.68, 0.672, 0.01176)],
    'random_shift': [(0.732, 0.723, 0.0123), (0.708, 0.696, 0.01695), (0.708, 0.7, 0.0113), (0.68, 0.684, -0.00588),
                     (0.677, 0.654, 0.03397), (0.782, 0.72, 0.07928), (0.769, 0.652, 0.15215), (0.774, 0.749, 0.0323),
                     (0.718, 0.692, 0.03621), (0.68, 0.666, 0.02059)]}, 0.08: {
    'shift': [(0.732, 0.724, 0.01093), (0.708, 0.709, -0.00141), (0.708, 0.702, 0.00847), (0.68, 0.659, 0.03088),
              (0.677, 0.676, 0.00148), (0.782, 0.777, 0.00639), (0.769, 0.766, 0.0039), (0.774, 0.767, 0.00904),
              (0.718, 0.71, 0.01114), (0.68, 0.67, 0.01471)],
    'random_shift': [(0.732, 0.715, 0.02322), (0.708, 0.682, 0.03672), (0.708, 0.694, 0.01977), (0.68, 0.686, -0.00882),
                     (0.677, 0.638, 0.05761), (0.782, 0.658, 0.15857), (0.769, 0.559, 0.27308), (0.774, 0.717, 0.07364),
                     (0.718, 0.669, 0.06825), (0.68, 0.649, 0.04559)]}, 0.1: {
    'shift': [(0.732, 0.72, 0.01639), (0.708, 0.706, 0.00282), (0.708, 0.7, 0.0113), (0.68, 0.652, 0.04118),
              (0.677, 0.672, 0.00739), (0.782, 0.775, 0.00895), (0.769, 0.764, 0.0065), (0.774, 0.764, 0.01292),
              (0.718, 0.708, 0.01393), (0.68, 0.665, 0.02206)],
    'random_shift': [(0.732, 0.704, 0.03825), (0.708, 0.667, 0.05791), (0.708, 0.678, 0.04237), (0.68, 0.678, 0.00294),
                     (0.677, 0.616, 0.0901), (0.782, 0.573, 0.26726), (0.769, 0.455, 0.40832), (0.774, 0.677, 0.12532),
                     (0.718, 0.636, 0.11421), (0.68, 0.627, 0.07794)]}, 0.12: {
    'shift': [(0.732, 0.714, 0.02459), (0.708, 0.704, 0.00565), (0.708, 0.694, 0.01977), (0.68, 0.645, 0.05147),
              (0.677, 0.67, 0.01034), (0.782, 0.773, 0.01151), (0.769, 0.761, 0.0104), (0.774, 0.76, 0.01809),
              (0.718, 0.704, 0.0195), (0.68, 0.661, 0.02794)],
    'random_shift': [(0.732, 0.693, 0.05328), (0.708, 0.646, 0.08757), (0.708, 0.666, 0.05932), (0.68, 0.671, 0.01324),
                     (0.677, 0.584, 0.13737), (0.782, 0.496, 0.36573), (0.769, 0.362, 0.52926), (0.774, 0.626, 0.19121),
                     (0.718, 0.603, 0.16017), (0.68, 0.593, 0.12794)]}, 0.14: {
    'shift': [(0.732, 0.708, 0.03279), (0.708, 0.702, 0.00847), (0.708, 0.689, 0.02684), (0.68, 0.636, 0.06471),
              (0.677, 0.67, 0.01034), (0.782, 0.77, 0.01535), (0.769, 0.758, 0.0143), (0.774, 0.753, 0.02713),
              (0.718, 0.701, 0.02368), (0.68, 0.655, 0.03676)],
    'random_shift': [(0.732, 0.674, 0.07923), (0.708, 0.616, 0.12994), (0.708, 0.65, 0.08192), (0.68, 0.651, 0.04265),
                     (0.677, 0.556, 0.17873), (0.782, 0.423, 0.45908), (0.769, 0.292, 0.62029), (0.774, 0.576, 0.25581),
                     (0.718, 0.566, 0.2117), (0.68, 0.552, 0.18824)]}, 0.16: {
    'shift': [(0.732, 0.703, 0.03962), (0.708, 0.698, 0.01412), (0.708, 0.683, 0.03531), (0.68, 0.627, 0.07794),
              (0.677, 0.666, 0.01625), (0.782, 0.764, 0.02302), (0.769, 0.755, 0.01821), (0.774, 0.75, 0.03101),
              (0.718, 0.697, 0.02925), (0.68, 0.649, 0.04559)],
    'random_shift': [(0.732, 0.649, 0.11339), (0.708, 0.588, 0.16949), (0.708, 0.628, 0.11299), (0.68, 0.629, 0.075),
                     (0.677, 0.527, 0.22157), (0.782, 0.362, 0.53708), (0.769, 0.236, 0.69311), (0.774, 0.528, 0.31783),
                     (0.718, 0.531, 0.26045), (0.68, 0.516, 0.24118)]}, 0.18: {
    'shift': [(0.732, 0.697, 0.04781), (0.708, 0.695, 0.01836), (0.708, 0.677, 0.04379), (0.68, 0.614, 0.09706),
              (0.677, 0.663, 0.02068), (0.782, 0.758, 0.03069), (0.769, 0.75, 0.02471), (0.774, 0.745, 0.03747),
              (0.718, 0.693, 0.03482), (0.68, 0.642, 0.05588)],
    'random_shift': [(0.732, 0.626, 0.14481), (0.708, 0.557, 0.21328), (0.708, 0.6, 0.15254), (0.68, 0.602, 0.11471),
                     (0.677, 0.494, 0.27031), (0.782, 0.309, 0.60486), (0.769, 0.192, 0.75033), (0.774, 0.478, 0.38243),
                     (0.718, 0.494, 0.31198), (0.68, 0.484, 0.28824)]}, 0.2: {
    'shift': [(0.732, 0.692, 0.05464), (0.708, 0.691, 0.02401), (0.708, 0.671, 0.05226), (0.68, 0.602, 0.11471),
              (0.677, 0.658, 0.02806), (0.782, 0.752, 0.03836), (0.769, 0.747, 0.02861), (0.774, 0.74, 0.04393),
              (0.718, 0.687, 0.04318), (0.68, 0.635, 0.06618)],
    'random_shift': [(0.732, 0.594, 0.18852), (0.708, 0.516, 0.27119), (0.708, 0.574, 0.18927), (0.68, 0.563, 0.17206),
                     (0.677, 0.463, 0.3161), (0.782, 0.268, 0.65729), (0.769, 0.165, 0.78544), (0.774, 0.43, 0.44444),
                     (0.718, 0.458, 0.36212), (0.68, 0.447, 0.34265)]}}


def get_model_name(m_id):
    if m_id == 0:
        return "regular_32"
    if m_id == 1:
        return "regular_64"
    if m_id == 2:
        return "regular_128"
    if m_id == 3:
        return "regular_256"
    if m_id == 4:
        return "regular_512"
    if m_id == 5:
        return "dropout_32"
    if m_id == 6:
        return "dropout_64"
    if m_id == 7:
        return "dropout_128"
    if m_id == 8:
        return "dropout_256"
    if m_id == 9:
        return "dropout_512"


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
    save_folder = os.path.join(os.getcwd(), "batch_size_analyze_result")
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
    save_folder = os.path.join(os.getcwd(), "batch_size_analyze_compare")
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
        acc_before.append(batch_size_test_batch[0.02]["shift"][i][0])

    # acc_after_shift
    #   key: model name
    #   value: an array
    #     x_axis: epsilon according to epsilons
    #     y_axis: accuracy of this model(key) after adding shift perturbation on epsilon(x_axis)
    acc_after_shift = {}
    for model_id in range(model_num):
        model_temp = []
        for epsilon in epsilons:
            model_temp.append(batch_size_test_batch[epsilon]["shift"][model_id][1])
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
            model_temp.append(batch_size_test_batch[epsilon]["random_shift"][model_id][1])
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
            model_temp.append(batch_size_test_batch[epsilon]["shift"][model_id][2])
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
            model_temp.append(batch_size_test_batch[epsilon]["random_shift"][model_id][2])
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
    # pic_title = "stochastic perturbation drop rate compare dropout"
    # plt.title(pic_title)
    # store_title = "super compare stochastic dropout"
    # save_folder = os.path.join(os.getcwd(), "batch_size_analyze_compare")
    # save_path = os.path.join(save_folder, store_title)
    # plt.savefig(save_path)
    # plt.show()
    #
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
    # save_folder = os.path.join(os.getcwd(), "batch_size_analyze_compare")
    # save_path = os.path.join(save_folder, store_title)
    # plt.savefig(save_path)
    # plt.show()
    #
    #
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
    # save_folder = os.path.join(os.getcwd(), "batch_size_analyze_compare")
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
    # save_folder = os.path.join(os.getcwd(), "batch_size_analyze_compare")
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
    plt.xticks(x, (32, 64, 128, 256, 512))
    plt.legend()
    plt.savefig("batch_size_acc_before")
    plt.show()
