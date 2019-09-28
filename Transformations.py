import numpy


def select_feature(dataset, indices):
    for i in indices:
        dataset = numpy.delete(dataset, i, axis=1)
    return dataset


def add_feature(dataset, indices):
    vector = numpy.full(dataset.shape[0], 1)
    for i in indices:
        vector = vector * numpy.array(dataset[:, i])
    dataset = numpy.concatenate((dataset, vector.reshape(vector.shape[0], 1)), axis=1)
    return dataset


def find_outliers(x, outlierConstant):
    resultList = []
    upper_quartile = numpy.percentile(x, 75)
    lower_quartile = numpy.percentile(x, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    for i in range(x.shape[0]):
        if x[i] <= quartileSet[0] or x[i] >= quartileSet[1]:
            resultList.append(i)
    resultList.reverse()
    return resultList
