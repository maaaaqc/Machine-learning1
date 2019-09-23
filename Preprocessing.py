import numpy
from pathlib import Path

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
    # normalizes each column
    for i in range(red_wine.shape[1]):
        red_wine[:,i] = normalize(red_wine[:,i])
    # red_wine = select_feature_wine(red_wine)
    return red_wine


def process_cancer(filename):
    breast_cancer = numpy.genfromtxt(filename, delimiter=",", skip_header=False)[:,1:]
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
        breast_cancer[:,i] = normalize(breast_cancer[:,i])
    #breast_cancer = select_feature_wine(breast_cancer)
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


def select_feature_wine(dataset):
    to_del = [4, 1]
    for i in to_del:
        dataset = numpy.delete(dataset, i, axis=1)
    return dataset


def select_feature_cancer(dataset):
    to_del = [8, 7, 3, 0]
    for i in to_del:
        dataset = numpy.delete(dataset, i, axis=1)
    return dataset