import step1
import math
import numpy

class LogReg:
    def __init__(self, data, rate, ites):
        self.rate = rate
        self.ites = ites
        self.y = data[:,-1]
        self.x = data[:,0:-1]
        self.w = numpy.full((self.x.shape[1], 1), 1)

    def fit(self, x, y, w):
        for i in range(self.ites):
            for j in range(x.shape[0]):
                xv = x[j].reshape(x.shape[1],1)
                ratio = numpy.matmul(w.transpose(), xv)[0][0]
                dif = numpy.full((1, 1), y[j]-self.sigma(ratio))
                wd = numpy.matmul(xv, dif)
                w = numpy.add(w, wd)
                print(w)
        self.w = w
        return w

    def sigma(self, ratio):
        return 1 // (1 + math.exp(-ratio))

if __name__ == "__main__":
    l1 = LogReg(step1.process_wine(),1/100,2)
    print(l1.fit(l1.x, l1.y, l1.w))

