import numpy
from pathlib import Path

WINEDIR = Path.cwd() / "wine" / "winequality-red.csv"
CANCERDIR = Path.cwd() / "cancer" / "breast-cancer-wisconsin.data"

def process_wine():
    red_wine = numpy.genfromtxt(str(WINEDIR), delimiter=";", skip_header=True)
    # deletes rows containing NaN values
    index = 0
    for row in red_wine:
        if numpy.isnan(row).any():
            red_wine = numpy.delete(red_wine, index, 0)
        else:
            index += 1
    # classifies wines into 1 and 0
    for row in red_wine:
        if row[-1] > 5:
            row[-1] = 1
        else: 
            row[-1] = 0
    # normalizes each column
    for i in range(red_wine.shape[1]):
        red_wine[:,i] = normalize(red_wine[:,i])
    return red_wine


def process_cancer():
    breast_cancer = numpy.genfromtxt(str(CANCERDIR), delimiter=",", skip_header=False)[:,1:-1]
    # deletes rows containing NaN values
    index = 0
    for row in breast_cancer:
        if numpy.isnan(row).any():
            breast_cancer = numpy.delete(breast_cancer, index, 0)
        else:
            index += 1
    # classifies wines into 1 and 0
    for row in breast_cancer:
        if row[-1] > 2:
            row[-1] = 1
        else: 
            row[-1] = 0
    # normalizes each column
    for i in range(breast_cancer.shape[1]-1):
        breast_cancer[:,i] = normalize(breast_cancer[:,i])
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
    return column


if __name__ == "__main__":
    process_wine()