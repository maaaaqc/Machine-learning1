import numpy


class LDA:
    def __init__(self, data):
        self.y = data[:, -1]
        self.x = data[:, 0:-1]
        n1 = numpy.count_nonzero(self.y)
        n0 = self.y.shape[0] - n1
        self.n = [n0, n1]
        self.initialize()

    def initialize(self):
        self.u = [self.mean(0), self.mean(1)]
        # stores the inverse of the covariance matrix
        self.cvinv = numpy.linalg.inv(self.cov(self.x, self.u, self.n))
        # stores the constant part w0
        w0 = numpy.log(self.n[1]/self.n[0]) - \
            0.5 * numpy.dot(numpy.dot(self.u[1].transpose(), self.cvinv), self.u[1]) + \
            0.5 * numpy.dot(numpy.dot(self.u[0].transpose(), self.cvinv), self.u[0])
        self.w0 = w0

    def fit(self, cvinv, mean):
        dif = numpy.subtract(mean[1], mean[0])
        w = numpy.dot(cvinv, dif)
        return w

    def mean(self, clss):
        # calculates the mean vector of the given class
        mu = numpy.full((self.x.shape[1], 1), 0)
        for i in range(self.x.shape[0]):
            if self.y[i] == clss:
                mu = numpy.add(mu, self.x[i].reshape(self.x.shape[1], 1))
        mean = mu / self.n[clss]
        return mean

    def cov(self, x, u, n):
        # calculates the covariance matrix of the data set
        c = numpy.full((x.shape[1], x.shape[1]), 0)
        div = (n[0] + n[1] - 2)
        for k in [0, 1]:
            for i in range(x.shape[0]):
                if self.y[i] == k:
                    a = numpy.subtract(x[i].reshape(x.shape[1], 1), u[k])
                    c = numpy.add(c, numpy.divide(numpy.outer(a, a), div))
        return c

    def predict(self, val_set, w):
        result = []
        for i in range(val_set.shape[0]):
            result.append(numpy.dot(val_set[i].transpose(), w) + self.w0)
            if result[i] > 0:
                result[i] = 1
            else:
                result[i] = 0
        return result
