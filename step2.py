import step1
import math
import numpy

class LogReg:
    def __init__(self, data):
        self.y = data[:,-1]
        self.x = data[:,0:-1]

    def fit(self, x, y, rate, ites):
        w = numpy.full((self.x.shape[1], 1), 1)
        for i in range(ites):
            w_old = w
            for j in range(x.shape[0]):
                sigma = self.sigma(numpy.matmul(w_old.transpose(), x[j].transpose())[0])
                w = numpy.add(w, rate * ((y[j]-sigma) * x[j].reshape(x.shape[1],1)))
        return w

    def sigma(self, ratio):
        if ratio < 0:
            return 1 - 1 / (1 + math.exp(ratio))
        else:
            return 1 / (1 + math.exp(-ratio))

    # def predict(data):
    #    for :

if __name__ == "__main__":
    l1 = LogReg(step1.process_wine())
    w  = l1.fit(l1.x, l1.y, 0.01, 1000)
    print(w)

