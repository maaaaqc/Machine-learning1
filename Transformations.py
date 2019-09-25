import numpy


def select_feature(dataset, indices):
    for i in indices:
        dataset = numpy.delete(dataset, i, axis=1)
    return dataset


def add_feature(dataset, indices):
    vector = numpy.full((dataset.shape[0], 1), 1)
    for i in indices:
        for j in vector:
            vector[j] = vector[j] * dataset[j, i]
    dataset = numpy.concatenate((dataset, vector), axis=1)
    return dataset
