import math
import numpy

class LogReg:
    def __init__(self, data):
        self.y = data[:,-1]
        self.x = data[:,0:-1]

    def fit(self, rate, ites):
        w = numpy.full((self.x.shape[1], 1), 1)
        for i in range(ites):
            w_old = w
            for j in range(self.x.shape[0]):
                sigma = self.sigma(numpy.matmul(w_old.transpose(), self.x[j].transpose())[0])
                w = numpy.add(w, rate * ((self.y[j]-sigma) * self.x[j].reshape(self.x.shape[1],1)))
        return w

    def sigma(self, ratio):
        if ratio < 0:
            return 1 - 1 / (1 + math.exp(ratio))
        else:
            return 1 / (1 + math.exp(-ratio))

    def predict(self, val_set, w):
        r = numpy.full((val_set.shape[0], 1), 0)
        for i in range(val_set.shape[0]):
            r[i] = numpy.matmul(w.transpose(), val_set[i])
            if self.sigma(r[i]) > 0.5:
                r[i] = 1
            else:
                r[i] = 0
        return r
