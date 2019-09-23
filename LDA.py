import Preprocessing
import numpy
import math

class LDA:
    def __init__(self, data):
        self.y = data[:,-1]
        self.x = data[:,0:-1]
        n1 = numpy.count_nonzero(self.y)
        n0 = self.y.shape[0] - n1
        self.n = [n0, n1]
        self.initialize()


    def initialize(self):
        self.u = [self.mean(0), self.mean(1)]
        self.cvinv = numpy.linalg.inv(self.cov())
        w0 = math.log(self.n[1]/self.n[0]) - \
            0.5 * numpy.matmul(numpy.matmul(self.u[1].transpose(), self.cvinv), self.u[1]) + \
            0.5 * numpy.matmul(numpy.matmul(self.u[0].transpose(), self.cvinv), self.u[0])
        self.w0 = w0


    def fit(self, cvinv, mean):
        w = numpy.matmul(cvinv, numpy.subtract(mean[1], mean[0]))
        return w


    def mean(self, clss):
        mu = numpy.full((self.x.shape[1],1), 0)
        for i in range(self.x.shape[0]):
            if self.y[i] == clss:
                mu = numpy.add(mu, self.x[i].reshape(self.x.shape[1],1))
        return mu / self.n[clss]


    def cov(self):
        c = numpy.full((self.x.shape[1], self.x.shape[1]), 0)
        for k in range(2):
            for i in range(self.x.shape[0]):
                if self.y[i] == k:
                    a = numpy.subtract(self.x[i].reshape(self.x.shape[1],1), self.u[k])
                    c = numpy.add(c, numpy.outer(a, a))
        return c / (self.n[0] + self.n[1] - 2)


    def predict(self, val_set, w):
        r = numpy.full((val_set.shape[0], 1), 0)
        for i in range(val_set.shape[0]):
            r[i] = numpy.matmul(val_set[i].transpose(), w) + self.w0
            if r[i] > 0:
                r[i] = 1
            else:
                r[i] = 0
        return r