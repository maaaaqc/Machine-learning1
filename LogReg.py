import math
import numpy

class LogReg:
    def __init__(self, data):
        self.y = data[:,-1]
        w0 = numpy.full((data.shape[0], 1), 1)
        self.x = numpy.concatenate((w0, data[:,0:-1]), axis=1)


    def fit(self, start_rate, end_rate, ites):
        w = numpy.full((self.x.shape[1], 1), 1)
        for i in range(ites):
            # stores the value of wk
            w_old = w
            for j in range(self.x.shape[0]):
                #calculates the sigma function result
                sigma = self.sigma(numpy.dot(w_old.transpose(), self.x[j].transpose())[0])
                # updates w
                w = numpy.add(w, start_rate * ((self.y[j]-sigma) * self.x[j].reshape(self.x.shape[1],1)))
        for i in range(ites):
            # stores the value of wk
            w_old = w
            for j in range(self.x.shape[0]):
                #calculates the sigma function result
                sigma = self.sigma(numpy.dot(w_old.transpose(), self.x[j].transpose())[0])
                # updates w
                w = numpy.add(w, end_rate * ((self.y[j]-sigma) * self.x[j].reshape(self.x.shape[1],1)))
        return w


    def sigma(self, ratio):
        # 1-1/(1+math.exp(ratio)) is equal to 1/(1+math.exp(-ratio))
        # wrote it this way to avoid overflow
        if ratio < 0:
            return 1 - 1 / (1 + math.exp(ratio))
        else:
            return 1 / (1 + math.exp(-ratio))


    def predict(self, val_set, w):
        w0 = numpy.full((val_set.shape[0], 1), 1)
        val_set= numpy.concatenate((w0, val_set), axis=1)
        result = numpy.full((val_set.shape[0], 1), 0)
        for i in range(val_set.shape[0]):
            result[i] = numpy.dot(w.transpose(), val_set[i])
            # calculates estimated P(y=1|x) and classifies with boundary 0.5
            if self.sigma(result[i]) > 0.5:
                result[i] = 1
            else:
                result[i] = 0
        return result
