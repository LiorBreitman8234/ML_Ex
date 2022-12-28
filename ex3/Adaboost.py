import numpy as np
import pandas as pd


def split(data, labels):
    indexes = np.random.randint(0, len(data) - 1, len(data) // 2)
    train_data = [data[i] for i in indexes]
    test_data = [data[i] for i in range(len(data)) if i not in indexes]
    train_label = [labels[i] for i in indexes]
    test_label = [labels[i] for i in range(len(data)) if i not in indexes]
    return train_data, train_label, test_data, test_label


def createLines(data):
    dict_lines = {}
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j and 'ij' not in dict_lines.keys() and 'ji' not in dict_lines.keys():
                point_1 = data[i]
                point_2 = data[j]
                # m = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
                # c = point_2[1] - m * point_2[0]
                dict_lines[f"{i}{j}"] = (point_1, point_2)
    return dict_lines


class Adaboost:
    def __init__(self, num_iterations=50):
        self.num_iterations = num_iterations

    def __call__(self, data, labels):
        lines = createLines(data)
        print(len(lines))
        for i in range(self.num_iterations):
            x_train, y_train, x_test, y_test = split(data, labels)
            break

