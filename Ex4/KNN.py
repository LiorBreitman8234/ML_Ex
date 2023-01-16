import numpy as np


def lp_dist(type_distance: int, p1: (float, float, float), p2: (float, float, float)) -> float:
    """
    :param type_distance: type of distance
    :param p1: point 1
    :param p2: point 2
    :return: distance between p1 and p2
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
    :param type_distance: type of distance
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


if __name__ == '__main__':
    # test
    data = [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1), (3, 1, 2), (3, 2, 2), (4, 1, 2), (4, 2, 2)]
    print(knn(1, 3, (1, 1, 0), data))
    print(knn(2, 3, (1, 1, 0), data))
    print(knn(-1, 3, (1, 1, 0), data))