import time
from pathlib import Path

import numpy as np



def lp_dist(type_distance: int, p1: (float, float, float), p2: (float, float, float)) -> float:
    """
    :param type_distance: type of li distance (1, 2, -1) where -1 is infinity.
    :param p1: point 1
    :param p2: point 2
    :return: li between p1 and p2 according to type_distance.
    """

    if type_distance == 1:
        return np.linalg.norm(np.array(p1) - np.array(p2), ord=1)
    elif type_distance == 2:
        return np.linalg.norm(np.array(p1) - np.array(p2), ord=2)
    elif type_distance == -1:
        return np.linalg.norm(np.array(p1) - np.array(p2), ord=np.inf)
    else:
        raise ValueError("type_distance must be in [1, 2, -1]")



def split_data(points, labels):
    """
    :param points: points of data set
    :param labels: labels of data set
    :return: train and test data set (50% train and 50% test) and labels
    """

    n = len(points)
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:n // 2], indices[n // 2:]
    x_train, y_train = [points[i] for i in train_idx], [labels[i] for i in train_idx]
    x_test, y_test = [points[i] for i in test_idx], [labels[i] for i in test_idx]
    return x_train, y_train, x_test, y_test



def build_net(points, labels, type_distance):
    """
    :param points: points of data
    :param labels: labels of data
    :param type_distance: type of lp distance
    :return: list of tuples (point, label) sorted by distance to point according to type_distance.
    """

    type_1_points = [points[i] for i in range(len(points)) if labels[i] == 1]
    type_2_points = [points[i] for i in range(len(points)) if labels[i] == 2]
    epsilon = np.min([lp_dist(type_distance, p1, p2) for p1 in type_1_points for p2 in type_2_points])
    T = []
    L = []
    for i in range(len(points)):
        if len(T) == 0:
            T.append(points[i])
            L.append(labels[i])
        else:
            distances = [lp_dist(type_distance, points[i], point_t) for point_t in T]
            min_distance = np.min(distances)
            if min_distance > epsilon:
                T.append(points[i])
                L.append(labels[i])
    return T, L



def knn(type_distance: int, k: int, p: (float, float, float), points: [(float, float, float)], labels) -> int:
    """
    :param type_distance: type of li distance (1, 2, -1) where -1 is infinity.
    :param k: number of neighbors
    :param p: point to predict
    :param points: points of data
    :param labels: labels of data
    :return: prediction for p (1 or 2) according to k nearest neighbors algorithm with type_distance li distance.
    """

    distances = [(lp_dist(type_distance, p, points[i]), labels[i]) for i in range(len(points))]
    distances.sort()
    count_1, count_2 = 0, 0
    for pair in distances[:k]:
        if pair[1] == 1:
            count_1 += 1
        else:
            count_2 += 1
    return 1 if count_1 > count_2 else 2



def read_date(file_name: str) -> ([(float, float, float)], [int]):
    """
    :param file_name: path to file
    :return: list of points and list of labels
    """

    data_folder = Path("./data")
    file_to_open = data_folder / file_name
    pi = []
    li = []
    with open(file_to_open, 'r') as file:
        for line in file:
            line = line.split(',')
            pi.append((float(line[0]), float(line[1]), float(line[2])))
            li.append(int(line[3]))
    return pi, li



def write_to_file(file_name: str, data: str):
    """
    :param file_name: path to file
    :param data: data to write
    """

    with open(file_name, 'w') as file:
        file.write(data)



if __name__ == '__main__':
    ks = [1, 3, 5, 7, 9]
    lps = [1, 2, -1]
    data_name = 'haberman.data'
    points, labels = read_date(data_name)
    res = "Summary of results for haberman.data with 100 runs for k in [1, 3, 5, 7, 9] and lp in [1, 2, INF]:\n"
    logs = 'Output of KNN.py\n\n|    k    |     li     |     iteration     |      correct train      |      correct test      |\n\n'
    time_took = time.time()
    for k in ks:
        for li in lps:
            empirical_error = []
            test_error = []
            li_ = 'INF' if li == -1 else li
            for i in range(100):
                train_data, train_labels, test_data, test_labels = split_data(points, labels)
                epsilon_net, net_labels = build_net(train_data, train_labels, li)
                correct_test, correct_train = 0, 0
                for j in range(len(train_data)):
                    pred = knn(li, k, train_data[j], epsilon_net, net_labels)
                    if pred == train_labels[j]:
                        correct_train += 1
                for j in range(len(test_data)):
                    pred = knn(li, k, test_data[j], epsilon_net, net_labels)
                    if pred == test_labels[j]:
                        correct_test += 1

                logs += f'   k = {k},    li = {li_},      iteration = {i},       correct train = {correct_train},       correct test = {correct_test}\n'
                empirical_error.append(correct_train / len(train_data))
                test_error.append(correct_test / len(test_data))

            average_emp = round(1 - np.mean(empirical_error), 5)
            average_test = round(1 - np.mean(test_error), 5)
            print(f"For {k} neighbours and {li_} type distance we got {average_emp} empirical error and {average_test} test error, the difference is {round(abs(average_emp - average_test), 5)}.")
            res += f"For {k} neighbours and {li_} type distance we got {average_emp} empirical error and {average_test} test error, the difference is {round(abs(average_emp - average_test), 5)}.\n"

    time_took = time.time() - time_took
    print(f"it took {time_took // 60} minutes and {round(time_took % 60, 2)} seconds")
    write_to_file(f'logs{data_name.split(".")[0].capitalize()}.txt', logs)
    write_to_file(f'results{data_name.split(".")[0].capitalize()}.txt', res)
