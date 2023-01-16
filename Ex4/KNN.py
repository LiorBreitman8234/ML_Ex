import numpy as np


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


def knn(type_distance: int, k: int, p: (float, float, float), data: [(float, float, float)]) -> int:
    """
    :param type_distance: type of li
    :param k: number of nearest neighbors
    :param p: point
    :param data: list of points
    :return: label of p
    """
    distances = []
    for point in data:
        distances.append(lp_dist(type_distance, p, point))
    distances = np.array(distances)
    labels = np.array([point[2] for point in data])
    sorted_distances = np.argsort(distances)
    labels = labels[sorted_distances]
    labels = labels[:k]
    return np.argmax(np.bincount(labels))


# reads ./data/haberman.data and returns a list of points and a list of labels
def read_date(path: str) -> ([(float, float, float)], [int]):
    """
    :param path: path to file
    :return: list of points and list of labels
    """
    p = []
    l = []
    with open(path, 'r') as file:
        for line in file:
            line = line.split(',')
            p.append((float(line[0]), float(line[1]), float(line[2])))
            l.append(int(line[3]))
    return p, l


if __name__ == '__main__':
    ks = [1, 3, 5, 7, 9]
    lps = [1, 2, -1]
    points, labels = read_date('./data/haberman.data')
    for k in ks:
        for li in lps:
            correct = 0
            for i in range(len(points)):
                if knn(li, k, points[i], points[:i] + points[i + 1:]) == labels[i]:
                    correct += 1
            print(f'k = {k}, li = {li}, accuracy = {correct / len(points)}')