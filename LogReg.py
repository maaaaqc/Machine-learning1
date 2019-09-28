import math
import numpy


class LogReg:
    def __init__(self, data):
        self.y = data[:, -1]
        w0 = numpy.full((data.shape[0], 1), 1)
        # concatenates w0 onto x
        self.x = numpy.concatenate((w0, data[:, 0:-1]), axis=1)

    def fit(self, rate, ites):
        # generates a srandom start from standard normal distribution
        w = numpy.random.randn(self.x.shape[1], 1)
        w = self.grad_descent(rate, ites, w)
        w = self.grad_descent(rate/20, ites, w)
        return w

    def grad_descent(self, rate, ites, w):
        for i in range(ites):
            # stores the value of wk
            w_old = w
            for j in range(self.x.shape[0]):
                # calculates the sigma function result
                sigma = self.sigma(numpy.dot(w_old.transpose(), self.x[j].transpose())[0])
                # updates w
                w = numpy.add(w, rate * ((self.y[j]-sigma) * self.x[j].reshape(self.x.shape[1], 1)))
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
        val_set = numpy.concatenate((w0, val_set), axis=1)
        # val_set = self.add_dimension(val_set)
        result = []
        for i in range(val_set.shape[0]):
            result.append(numpy.dot(w.transpose(), val_set[i]))
            # calculates estimated P(y=1|x) and classifies with boundary 0.5
            if self.sigma(result[i]) > 0.5:
                result[i] = 1
            else:
                result[i] = 0
        return result

    # some helper funcitons used for analysis
    def mean(self):
        # calculates the mean vector of X's
        mu = numpy.full(self.x.shape[1], 0)
        for i in range(self.x.shape[0]):
            mu = numpy.add(mu, self.x[i])
        mean = mu / self.x.shape[0]
        return mean

    def corr(self):
        # calculate the correlation between each feature X and Y
        ux = self.mean()
        uy = 0
        num = numpy.full(self.x.shape[1], 0)
        denum0 = numpy.full(self.x.shape[1], 0)
        denum1 = 0
        for i in range(self.y.shape[0]):
            uy = uy + self.y[i]
        uy = uy / self.y.shape[0]
        for i in range(self.x.shape[0]):
            num = numpy.add(num, (self.x[i] - ux) * (self.y[i] - uy))
            denum0 = numpy.add(denum0, numpy.power(self.x[i], 2))
            denum1 = denum1 + (self.y[i] - uy) ** 2
        denum = numpy.sqrt(denum0 * denum1)
        corr = numpy.divide(num, denum)
        return corr
