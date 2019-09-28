import numpy
from pathlib import Path
import Transformations

WINEDIR = Path.cwd() / "wine" / "winequality-red.csv"
CANCERDIR = Path.cwd() / "cancer" / "breast-cancer-wisconsin.data"


def process_wine(filename):
    red_wine = numpy.genfromtxt(filename, delimiter=";", skip_header=True)

    # deletes rows containing NaN values
    red_wine = red_wine[~numpy.isnan(red_wine).any(axis=1)]

    # classifies wines into 1 and 0
    for row in red_wine:
        if row[-1] > 5:
            row[-1] = 1
        else:
            row[-1] = 0

    # removes outliers from selected columns
    list_del = {3, 4, 9}
    for i in list_del:
        li = Transformations.find_outliers(red_wine[:, i], 1.8)
        for j in li:
            red_wine = numpy.delete(red_wine, j, axis=0)

    # adds features
    # y = red_wine[:, -1]
    # x = red_wine[:, 0:-1]
    # x = Transformations.add_feature(x, [1, 4])
    # red_wine = numpy.concatenate((x, y.reshape(y.shape[0], 1)), axis=1)

    # selects features
    # red_wine = Transformations.select_feature(red_wine, [8, 3])

    # normalizes each column
    for i in range(red_wine.shape[1]):
        red_wine[:, i] = normalize(red_wine[:, i])
    return red_wine


def process_cancer(filename):
    breast_cancer = numpy.genfromtxt(filename, delimiter=",")[:, 1:]
    # deletes rows containing NaN values
    breast_cancer = breast_cancer[~numpy.isnan(breast_cancer).any(axis=1)]

    # classifies wines into 1 and 0
    for row in breast_cancer:
        if row[-1] > 2:
            row[-1] = 1
        else:
            row[-1] = 0

    # normalizes each column
    for i in range(breast_cancer.shape[1]):
        breast_cancer[:, i] = normalize(breast_cancer[:, i])

    return breast_cancer


def normalize(column):
    # uses the min-max normalization
    max = 0
    min = float("inf")
    for i in column:
        if i > max:
            max = i
        if i < min:
            min = i
    if not max == min:
        for i in range(column.shape[0]):
            column[i] = (column[i] - min) / (max - min)
    else:
        if max != 0:
            column[i] = column[i] / max
    return column
