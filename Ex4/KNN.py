from pathlib import Path

import numpy as np

# calculate needed distance 
def lp_dist(type_distance: int, p1: (float, float, float), p2: (float, float, float)) -> float:
    """
    :param type_distance: type of li
    :param p1: point 1
    :param p2: point 2
    :return: li between p1 and p2
    """
    if type_distance == 1:
        return np.linalg.norm(np.array(p1) - np.array(p2), ord=1)
    elif type_distance == 2:
        return np.linalg.norm(np.array(p1) - np.array(p2), ord=2)
    elif type_distance == -1:
        return np.linalg.norm(np.array(p1) - np.array(p2), ord=np.inf)
    else:
        raise ValueError("type_distance must be in [1, 2, -1]")

#This function splits the data to train and test
def split_data(points,labels):
    n = len(points)
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:n // 2], indices[n // 2:]
    x_train, y_train = [points[i] for i in train_idx], [labels[i] for i in train_idx]
    x_test, y_test = [points[i] for i in test_idx], [labels[i] for i in test_idx]
    return x_train, y_train, x_test, y_test

#This function builds the epsilon-net needed to make guesses
def build_net(points,labels,type_distance):
    """
    param points: points of data
    param labels: labels of data
    param type_distance: type of lp distance
    """
    type_1_points = [points[i] for i in range(len(points)) if labels[i] == 1]
    type_2_points = [points[i] for i in range(len(points)) if labels[i] == 2]
    epsilon = np.min([lp_dist(type_distance,p1,p2) for p1 in type_1_points for p2 in type_2_points])
    T = []
    L = []
    for i in range(len(points)):
        if len(T) == 0:
            T.append(points[i])
            L.append(labels[i])
        else:
            distances = [lp_dist(type_distance,points[i],point_t) for point_t in T]
            min_distance = np.min(distances)
            if min_distance > epsilon:
                T.append(points[i])
                L.append(labels[i])
    return T,L
        


# make prediction for a point
def knn(type_distance: int, k: int, p: (float, float, float), points: [(float, float, float)], labels) -> int:
    """
    :param type_distance: type of li
    :param k: number of nearest neighbors
    :param p: point
    :param data: list of points
    :return: label of p
    """
    distances = [(lp_dist(type_distance,p,points[i]),labels[i]) for i in range(len(points))]

    return np.bincount(np.array(distances).argsort()[:k]).argmax()


# reads ./data/haberman.data and returns a list of points and a list of labels
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


if __name__ == '__main__':
    ks = [1, 3, 5, 7, 9]
    lps = [1, 2, -1]
    points, labels = read_date('haberman.data')
    results = open('results.txt','w')
    for k in ks:
        for li in lps:
            empirical_error = []
            test_error = []
            for i in range(100):
                train_data, train_labels,test_data, test_labels = split_data(points,labels)
                epsilon_net,net_labels = build_net(train_data,train_labels,li)
                correct_test,correct_train = 0,0
                for j in range(len(train_data)):
                    pred = knn(li,k,train_data[j],epsilon_net)
                    if pred == train_labels[j]:
                        correct_train += 1
                for j in range(len(test_data)):
                    pred = knn(li,k,test_data[j],epsilon_net)
                    if pred == test_labels[j]:
                        correct_test += 1
                print(f"correct in train: {correct_train}, not correct: {len(train_data) - correct_train}")
                print(f"correct in test: {correct_test}, not correct: {len(test_data) - correct_test}")
                empirical_error.append(correct_train/len(train_data))
                test_error.append(correct_test/len(test_data))        
            average_emp = np.mean(empirical_error)
            average_test = np.mean(test_error)
            print(f"for {k} neighbours and {li} type distance we got {round(average_emp,5)} empirical error and {round(average_test,5)} test error")
            results.write(f"for {k} neighbours and {li} type distance we got {round(average_emp,5)} empirical error and {round(average_test,5)} test error\n")
    results.close()
