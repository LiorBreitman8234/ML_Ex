# This is a sample Python script.
from Winnow import Winnow


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def create_data_list(name) -> (list, list):
    rows, targets = [], []
    file = open(name, 'r')
    lines = file.readlines()
    for line in lines:
        digits = line.split()
        digits = list(map(int, digits))
        label = digits[len(digits) - 1]
        row = digits[:-1]
        rows.append(row)
        targets.append(label)
    return rows, targets


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data, labels = create_data_list('/home/bravo8234/Desktop/study/Machine_learning/Ex3/winnow_vectors.txt')
    algo = Winnow(len(data[0]))
    algo(data, labels)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
