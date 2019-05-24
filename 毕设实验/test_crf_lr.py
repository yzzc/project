from tools import load_data
from relative_density import relative_density
from generateCompleteTree import CRFNFL_LR
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def kernel_LR(train, test=None):
    """kernel-hinge accuracy"""

    model = LogisticRegression()

    model.fit(train[:, 1:], train[:, 0])
    acc = model.score(test[:, 1:], test[:, 0])
    # acc = acc * 100
    return acc


def main_train(key, noise, divide_num=5):
    """在指定噪声率下，训练，并返回其精度、标准差"""

    temp_acc = []
    print(key, "数据集 ", "噪声率为：", noise, sep="")
    for i in range(divide_num):
        train, test = load_data(key, 0.30, noise)

        acc0 = 0

        acc1 = CRFNFL_LR(train, test)         # 调用随机森林
        # acc1 = 0                             # 调试RD阈值

        acc2 = 0                             # 测试随机森林算法阈值

        temp_acc.append([acc0, acc1, acc2])
        print("klr:", acc0, "crf:", acc1, "rd:", acc2)

    mean_acc = np.mean(temp_acc, axis=0)
    round_acc = np.round(mean_acc, 4)

    std_acc = np.std(temp_acc, axis=0)
    round_accstd = np.round(std_acc, 4)

    print("KLR:%.4lf" % (round_acc[0]), "CRF:%.4lf" % (round_acc[1]),
          "RD:%.4lf" % (round_acc[2]), "KLR:%.4lf" % (round_accstd[0]), "CRF:%.4lf" % (round_accstd[1]),
          "RD:%.4lf" % (round_accstd[2]))
    print()

    return mean_acc, round_acc, round_accstd


def main_LR():
    import warnings
    warnings.filterwarnings("ignore")

    keys = ["breastcancer", "diabetes", "german", "heart", "image", "thyroid"]
    noises = [(0.2, 0.2)]

    acc_all = []
    acc_all1 = []
    acc_all2 = []
    for key in keys:
        for noise in noises:
            mean_acc, round_acc, std_acc = main_train(key, noise)
            acc_all.append(round_acc)
            acc_all1.append(std_acc)
            acc_all2.append(mean_acc)

    mean = np.round(np.mean(acc_all2, axis=0), 4)

    # acc_all.append(list(mean))
    # temp_all = pd.DataFrame(acc_all, columns=["KLR", "CRF", "RD"])
    # temp_all.to_excel(r"C:\Users\ASUS\Desktop\毕设实验\record data/kernel_LR.xlsx", index=False)
    # 
    # temp_all1 = pd.DataFrame(acc_all1, columns=["KLR", "CRF", "RD"])
    # temp_all1.to_excel(r"C:\Users\ASUS\Desktop\毕设实验\record data/kernel_LR_std.xlsx", index=False)

    print("KSVM_mean:", mean[0], "CRF_mean:", mean[1], "RD_mean:", mean[2])
    print()


if __name__ == "__main__":
    main_LR()

