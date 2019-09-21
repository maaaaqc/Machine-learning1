import numpy
from pathlib import Path

WINEDIR = Path.cwd() / "wine" / "winequality-red.csv"

def process_data():
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