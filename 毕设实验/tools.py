from sklearn.model_selection import train_test_split
import numpy as np


def cal_sample_weight(dataset, noise_rate):
    """
    Add the rollover noise.

    """

    samples_num = dataset.shape[0]
    positive_num = np.count_nonzero(dataset[:, 0] == 1)
    negative_num = np.count_nonzero(dataset[:, 0] == -1)

    m = int(positive_num * noise_rate[0])
    n = int(negative_num * noise_rate[1])
    noise_index_list = []        # Record all subscripts of the noise

    p_index = 0
    n_index = 0
    np.random.seed()

    while True:
        rand_index = int(np.random.uniform(0, samples_num))

        if rand_index in noise_index_list:
            continue

        if dataset[rand_index, 0] == 1 and p_index < m:
            dataset[rand_index, 0] = -1
            p_index += 1
            noise_index_list.append(rand_index)

        elif dataset[rand_index, 0] == -1 and n_index < n:
            dataset[rand_index, 0] = 1
            n_index += 1
            noise_index_list.append(rand_index)

        if p_index >= m and n_index >= n:
            break
    return dataset




def load_data(key, test_rate=0.3, noise_rate=(0.2, 0.2)):
    """
    Import data set and add the rollover noise.

    """

    import scipy.io as sio
    path = r"C:\Users\ASUS\Desktop\毕设实验"
    data = sio.loadmat(path+"/dataset.mat")
    data = data[key]
    train, test = train_test_split(data, test_size=test_rate)

    if noise_rate[0] != 0 or noise_rate[1] != 0:
        train = cal_sample_weight(train, noise_rate)
    return train, test


