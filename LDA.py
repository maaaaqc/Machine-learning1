#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import step1
import step2
import numpy
import math

class LDA:
    
    def __init__(self, data):
        self.y = data[:,-1]
        self.x = data[:,0:-1]
        self.N1 = numpy.count_nonzero(self.y)
        self.N0 = self.y.shape[0] - self.N1
        
    def w0(self):
        w0 = math.log(self.N1 / self.N0) - 0.5 * numpy.matmul(numpy.matmul(numpy.transpose(self.mean(1)), self.cov()), self.mean(0))
        return w0
    
    def fit(self):
         w = numpy.matmul(numpy.linalg.inv(self.cov()), numpy.subtract(self.mean(1), self.mean(0)))
         return w
         
    def mean(self, clss):
        mu = numpy.full((self.x.shape[1], 1), 0)
        for i in range(self.x.shape[0]):
            if self.y[i] == clss:
                mu = numpy.add(mu, self.x[i])
        print(mu/self.N0)
        return mu / self.N0
         
    def cov(self):
        c = numpy.full((self.x.shape[1], self.x.shape[1]), 0)
        for k in [0,1]:
            for i in range(self.x.shape[0]):
                if self.y[i] == k:
                    a = self.x[i] - self.mean(k)
                    #print(a)
                    #print(numpy.matmul(numpy.transpose(a), a))
                    c = numpy.add(c, numpy.matmul(numpy.transpose(a), a))
        return c / (self.N0 + self.N1 - 2)
    
    def predict(self, val_set, w):
        r = numpy.full((val_set.shape[0], 1), 0)
        for i in range(1):
            print(w)
            #print(self.w0())
            r[i] = numpy.matmul(numpy.transpose(w), val_set[i]) + self.w0()
            if r[i] > 0:
                r[i] = 1
            else:
                r[i] = 0
        return r
    
def kfold(data, k):
    numpy.random.shuffle(data)
    groups = numpy.array_split(data, k, axis=0)
    for i in range(1):
        val_set = groups[i][:,0:-1]
        true_val = groups[i][:,-1]
        train_set = numpy.concatenate(groups[:i] + groups[i+1:], axis=0)
        model = LDA(train_set)
        w = model.fit()
        r = model.predict(val_set, w)
        print(step2.get_accuracy(r, true_val))
                    
if __name__ == "__main__":
    kfold(step1.process_wine(), 5)
    kfold(step1.process_cancer(), 5)