import numpy as np
from pathlib import Path

"""
Steps:
preprocess data:
    0.1 create dictionary of all the possible lines where the key is the string of start end point of the 
        line and the value is the line.
    0.2 create a list of all the starting weights for each line.
    0.3 create a dictionary of all the possible points where the key is the index and the value is the point.
    0.4 create a list of all the starting weights for each point.
    
1. Set up starting weights for each line and point.
2. For 1 to n=50 iterations:
    a. Split the data into 50% training and 50% testing.
    b. For 1 to k=8 rules:
        b.1. Compute the weights error for each rule(line).
        b.2. Choose the rule with the lowest error.
        b.3. Compute the alpha for the rule.
        b.4. Update the weights for each point.

"""

# key: id of a point, value: the point
POINTS = {int: (float, float)}

# key: line between two points as string "p1p2", value tuple of the line.
LINES = {str: ((float, float), (float, float))}

# the index in this list matches the point in @POINTS with the same key.
LABELS = [int]

# the index in this list matches the point in @POINTS with the same key.
POINT_WEIGHTS = np.zeros(100).astype(np.float64)



def fit(points, labels, num_iterations=50):

    """
    This function will run the adaboost algorithm on the given data.
    :param points: list of points
    :param labels: list of labels
    :param num_iterations: number of iterations
    :return: list of rules, list of alphas
    """

    global POINTS, LINES, POINT_WEIGHTS, LABELS

    # create dictionary of all the possible lines where the key is the string of start end point of the
    LINES = createLines(points)

    # create a dictionary where the key is an index and the value is the point.
    POINTS = {i: points[i] for i in range(len(points))}
    LABELS = labels

    # create a list of all the starting weights for each line
    POINT_WEIGHTS = np.full(shape=100, fill_value=0.01).astype(np.float64)

    rules = []
    alphas = []
    total_train_errors = []
    total_test_errors = []
    LOGS = 'Data of each iteration:\n\n'
    break_line = '#######################################################################################################################################'

    for i in range(num_iterations):
        # split the data into 50% training and 50% testing.
        x_train, y_train, x_test, y_test = split()
        # one run of adaboost each iteration
        train_errors, test_errors = one_adaboost(x_train, y_train, x_test, y_test, rules, alphas)

        print(f'\n{break_line}')
        print(f"iteration {i + 1}: train_err = {train_errors}, test_err = {test_errors}")
        print(f'{break_line}')

        LOGS += f"\n{break_line}\niteration {i + 1}: train_err = {train_errors}, test_err = {test_errors}\n{break_line}\n"

        total_train_errors.append(train_errors)
        total_test_errors.append(test_errors)

    avg_train_err, avg_test_err = calculate_avg_error(total_train_errors, total_test_errors, num_iterations)
    # print("\n\ntrain_errors: ", total_train_errors)

    return rules, alphas, avg_train_err, avg_test_err, LOGS



def one_adaboost(x_train, y_train, x_test, y_test, rules, alphas):

    """
    This function will run one iteration of the adaboost algorithm, each iteration will run 8 rules.
    :param x_train: list of training points
    :param y_train: list of training labels
    :param x_test: list of testing points
    :param y_test: list of testing labels
    :param rules: list of rules (lines)
    :param alphas: list of alphas (weights)
    :return: train error, test error
    """

    global POINT_WEIGHTS, POINTS, LINES
    train_errors = []
    test_errors = []

    for i in range(8):
        min_err = 1
        best_line = None
        for line in LINES.values():
            # line = (points[int(line[0])], points[int(line[1])])
            err = compute_line_error(x_train, y_train, line)
            if err < min_err:
                min_err = err
                best_line = line

        rules.append(best_line)
        alphas.append(compute_alpha(min_err))
        # print(f'iteration -> {i} best line: {best_line}, alpha: {alphas[-1]}')

        update_point_weights(x_train, y_train, rules[-1], alphas[-1])
        train_err = sum([1 if cumulative_Hx(rules, alphas, POINTS[x_train[i]]) != y_train[i] else 0
                        for i in range(len(x_train))]) / len(x_train)
        test_err = sum([1 if cumulative_Hx(rules, alphas, POINTS[x_test[i]]) != y_test[i] else 0
                        for i in range(len(x_test))]) / len(x_test)

        train_errors.append(train_err)
        test_errors.append(test_err)

    return train_errors, test_errors



def createLines(data):

    """
    This function will create a dictionary of all the possible lines where the key is the string
    of start end point of the line and the value is the line.
    :param data:
    :return: dictionary of all the possible lines
    """
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



# function to compute the error of a hypothesis
def compute_line_error(x_train, y_train, line):

    """
    This function will compute the error of a line,
    the error is the sum of the weights of the points that are misclassified.
    :param x_train: list of training points
    :param y_train: list of training labels
    :param line: the line
    :return: the error of the line
    """

    global POINT_WEIGHTS, POINTS
    n = len(x_train)
    err = 0
    # line = (x_train[int(line[0])], x_train[int(line[1])])
    for i in range(n):
        point = POINTS[x_train[i]]
        if point in line:
            continue

        if hx(line, point) != y_train[i]:
            err += POINT_WEIGHTS[x_train[i]]

    return err



# function to determine the label of a point
def hx(line: ((float, float), (float, float)), p3: (float, float)):

    """
    :param line: ((x1, y1), (x2, y2))
    :param p3: (x3, y3)
    :return: 1 if p3 is on the right side of the line, -1 if p3 is on the left side of the line
    """

    # print(f'Line: {line}, type: {type(line)}')
    p1, p2 = line
    mat = np.array([[1, 1, 1], [p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]]])
    return -1 if np.linalg.det(mat) > 0 else 1



def compute_alpha(err):

    """
    This function will compute the alpha of the best rule.
    :param err: the error of the best rule
    :return: the alpha of the best rule
    """

    if err == 0:
        return 1
    return 0.5 * np.log((1 - err) / err)



def update_point_weights(x_train, y_train, line, alpha):

    """
    This function will update the weights of the points according to the new rule.
    :param x_train: list of training points
    :param y_train: list of training labels
    :param line: the line
    :param alpha: the alpha of the line
    :return:   None
    """

    if line is None:
        print('line is None')
        raise Exception
    global POINT_WEIGHTS, POINTS
    n = len(x_train)
    sum_weights = 0
    for i in range(n):
        # line = (points[int(line[0])], points[int(line[1])])
        point = POINTS[x_train[i]]
        if point in line:
            continue

        POINT_WEIGHTS[x_train[i]] *= np.exp(-alpha * hx(line, point) * y_train[i])
        sum_weights += POINT_WEIGHTS[x_train[i]]

    for i in range(n):
        POINT_WEIGHTS[x_train[i]] /= sum_weights



def cumulative_Hx(lines, alphas, p3):

    """
    This function will determine the label of a point according to at most 8 rules.
    :param lines: list of lines
    :param alphas: list of alphas
    :param p3: the point
    :return: the label of the point
    """

    label = 1 if np.sum([alphas[i] * hx(lines[i], p3) for i in range(len(lines))]) > 0 else -1
    return label



def split():

    """
    This function will split the data into 50% training and 50% testing randomly.
    :return: x_train, y_train, x_test, y_test
    """

    global POINTS, LABELS
    n = len(POINTS)
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:n // 2], indices[n // 2:]
    x_train, y_train = [i for i in train_idx], [LABELS[i] for i in train_idx]
    x_test, y_test = [i for i in test_idx], [LABELS[i] for i in test_idx]
    return x_train, y_train, x_test, y_test



def calculate_avg_error(train_errors, test_errors, num_of_runs):

    """
        This function will calculate the average error of the training and testing sets of line.
        :param train_errors: list of training errors
        :param test_errors: list of testing errors
        :param num_of_runs: number of runs
        :return: average training error, average testing error
        """

    # sum_train = np.zeros(len(train_errors[0]))
    # sum_test = np.zeros(len(test_errors[0]))
    sum_train = np.sum(np.array(train_errors), axis=0)
    sum_test = np.sum(np.array(test_errors), axis=0)

    train_err = sum_train / num_of_runs
    test_err = sum_test / num_of_runs

    return train_err, test_err



def read_data(file_name):

    """
    This function will read the data from the file, and return the points and labels.
    :param file_name: the name of the file
    :return: points, labels
    """

    data_folder = Path("/Ex3/data/")
    file_to_open = data_folder / file_name

    points = []
    labels = []
    with open(file_to_open, 'r') as data:
        rows = data.readlines()

        for li in rows:
            li = li.strip()
            row = li.split(" ")
            point = (float(row[0]), float(row[1]))
            if point in points:
                continue
            points.append(point)
            labels.append(1 if int(row[2]) == 1 else -1)
    return points, labels



def write_to_file(file_name, rules, alphas):

    """
    This function will write the rules and alphas to a file, so we can verify the results.
    :param file_name: the name of the file
    :param rules: the rules
    :param alphas: the alphas
    :return:
    """

    data_folder = Path("/Ex3/data/")
    file_to_open = data_folder / f'{file_name}_output.txt'

    global POINTS, LABELS
    with open(file_to_open, 'w') as data:
        for i in range(len(POINTS)):
            point = POINTS[i]
            label = cumulative_Hx(rules, alphas, point)
            data.write(f'{point[0]} {point[1]} {LABELS[i]} {label}\n')



def write_logs(file_name, log):

    """
    This function will write the logs to a file.
    :param file_name: the name of the file
    :param log: the logs
    :return:
    """

    data_folder = Path("/Ex3/")
    file_to_open = data_folder / f'{file_name}_logs.txt'

    with open(file_to_open, 'w') as data:
        for l in log:
            data.write(f'{l}')



if __name__ == '__main__':
    X, y = read_data("squares.txt")
    r, a, final_train_err, final_test_err, logs = fit(X, y, 50)
    write_to_file("after_train_squares", r, a)
    write_logs("data", logs)
    print("\nFinal train avg error ->", final_train_err)
    print("Final test avg error ->", final_test_err)
