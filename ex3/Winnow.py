import numpy as np
import pandas as pd


class Winnow:

    def __init__(self, amount_features):
        self.n = amount_features
        self.weights = np.ones(amount_features)

    def __call__(self, data, labels):
        num_mistakes = 0
        while True:
            print(f"current weights: {self.weights}")
            num_mistake = 0
            for i in range(len(data)):
                row = data[i]
                print(f"{i}th row: {row} ,label: {labels[i]}")
                row_sum = np.sum(np.array([self.weights[j] * row[j] for j in range(self.n)]))
                print(f"sum: {row_sum}, threshold: {self.n}")
                if row_sum > self.n:
                    if labels[i] == 0:
                        num_mistake += 1
                        for k in range(self.n):
                            if row[k] == 1:
                                self.weights[k] = 0
                        num_mistakes += 1
                        break
                else:
                    if labels[i] == 1:
                        num_mistake += 1
                        for k in range(self.n):
                            if row[k] == 1:
                                self.weights[k] = 2 * self.weights[k]
                        num_mistakes += 1
                        break
            if num_mistake == 0:
                print(f"the algo made: {num_mistakes} mistakes")
                print(f"final weights: {self.weights}")
                break
