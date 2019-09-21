import numpy
from pathlib import Path

WINEDIR = Path.cwd() / "wine" / "winequality-red.csv"
CANCERDIR = Path.cwd() / "wine" / "breast-cancer-wisconsin"

def process_wine():
    red_wine = numpy.genfromtxt(str(WINEDIR), delimiter=";", skip_header=True)
    index = 0
    for row in red_wine:
        if numpy.isnan(row).any():
            red_wine = numpy.delete(red_wine, index, 0)
        else:
            index += 1
    for row in red_wine:
        if row[-1] > 5:
            row[-1] = 1
        else: 
            row[-1] = 0
    return red_wine

def process_cancer():
    breast_cancer = numpy.genfromtxt(str(CANCERDIR), delimiter=",", skip_header=True)
    index = 0
    for row in breast_cancer:
        if numpy.isnan(row).any():
            breast_cancer = numpy.delete(breast_cancer, index, 0)
        else:
            index += 1
    return breast_cancer