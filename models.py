import numpy as np
import pandas as pd

from utils import diff


class Node:
    def fit(self, df, target_name):
        pass

    def predict(self, df):
        pass


class DecisionTreeClassifier:
    def __init__(self):
        self.root = Node()

    def fit(self, df, target_name):
        pass

    def predict(self, df):
        pass


def metrics(y, y_pred):
    TP = np.sum(np.logical_and(y_pred == True, y == True))
    TN = np.sum(np.logical_and(y_pred == False, y == False))
    FP = np.sum(np.logical_and(y_pred == True, y == False))
    FN = np.sum(np.logical_and(y_pred == False, y == True))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return accuracy, precision, recall, f1_score


def best_split(df, target_name):
    splits = dict({})
    y = df[target_name].values

    for col_name in diff(df.columns, [target_name]):
        vals = pd.unique(df[col_name])
        m = 0
        val_max = 0

        for val in vals:
            accuracy, precision, recall, f1_score = metrics(y, (df[col_name] >= val).values)
            if m < f1_score:
                m = f1_score
                val_max = val

        splits[col_name] = val_max

    return splits


def e_metrics(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += np.square(x1[i] - x2[i])

    return np.sqrt(distance)


def knn(x_train, y_train, x_test, k):
    answers = []

    for x in x_test:
        test_distances = []

        for i in range(len(x_train)):
            # расчет расстояния от классифицируемого объекта до
            # объекта обучающей выборки
            distance = e_metrics(x, x_train[i])

            # Записываем в список значение расстояния и ответа на объекте обучающей выборки
            test_distances.append((distance, y_train[i]))

        # создаем словарь со всеми возможными классами
        classes = {class_item: 0 for class_item in set(y_train)}

        # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
        for d in sorted(test_distances)[0:k]:
            classes[d[1]] += 1

        # Записываем в список ответов наиболее часто встречающийся класс
        answers.append(sorted(classes, key=classes.get)[-1])
    return answers
