import math
import numpy

class LogReg:
    def __init__(self, data):
        self.y = data[:,-1]
        self.x = data[:,0:-1]


    def fit(self, rate, ites):
        w = numpy.full((self.x.shape[1], 1), 1)
        for i in range(ites):
            # stores the value of wk
            w_old = w
            for j in range(self.x.shape[0]):
                #calculates the sigma function result
                sigma = self.sigma(numpy.matmul(w_old.transpose(), self.x[j].transpose())[0])
                # updates w
                w = numpy.add(w, rate * ((self.y[j]-sigma) * self.x[j].reshape(self.x.shape[1],1)))
        return w


    def sigma(self, ratio):
        # 1-1/(1+math.exp(ratio)) is equal to 1/(1+math.exp(-ratio))
        # wrote it this way to avoid overflow
        if ratio < 0:
            return 1 - 1 / (1 + math.exp(ratio))
        else:
            return 1 / (1 + math.exp(-ratio))


    def predict(self, val_set, w):
        result = numpy.full((val_set.shape[0], 1), 0)
        for i in range(val_set.shape[0]):
            result[i] = numpy.matmul(w.transpose(), val_set[i])
            # calculates estimated P(y=1|x) and classifies with boundary 0.5
            if self.sigma(result[i]) > 0.5:
                result[i] = 1
            else:
                result[i] = 0
        return result
