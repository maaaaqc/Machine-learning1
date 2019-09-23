#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import Preprocessing
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
        mu = numpy.full((self.x.shape[1],1), 0)
        for i in range(self.x.shape[0]):
            if self.y[i] == clss:
                mu = numpy.add(mu, self.x[i].reshape(self.x.shape[1],1))
        return mu / self.N0
         
    def cov(self):
        c = numpy.full((self.x.shape[1], self.x.shape[1]), 0)
        for k in [0,1]:
            for i in range(self.x.shape[0]):
                if self.y[i] == k:
                    a = numpy.subtract(self.x[i].reshape(self.x.shape[1],1), self.mean(k))
                    c = numpy.add(c, numpy.matmul(a, numpy.transpose(a)))
        return c / (self.N0 + self.N1 - 2)
    
    def predict(self, val_set, w):
        r = numpy.full((val_set.shape[0], 1), 0)
        for i in range(val_set.shape[0]):
            #print(w)
            #print(self.w0())
            r[i] = numpy.matmul(numpy.transpose(w), val_set[i]) + self.w0()
            if r[i] > 0:
                r[i] = 1
            else:
                r[i] = 0
        return r
    
def kfold(data, k):
    # shuffles the data and group them into k groups
    numpy.random.shuffle(data)
    groups = numpy.array_split(data, k, axis=0)
    # averages the acuracy of k predictions
    acc = 0
    for i in range(k):
        val_set = groups[i][:,0:-1]
        true_val = groups[i][:,-1]
        train_set = numpy.concatenate(groups[:i] + groups[i+1:], axis=0)
        # build a model using the training set
        model = LDA(train_set)
        w = model.fit()
        r = model.predict(val_set, w)
        acc += evaluate_acc(r, true_val)
    acc /= k
    return acc
        
def evaluate_acc(pred, fact):
    # counts the number of successful predictions
    count = 0
    for i in range(pred.shape[0]):
        if pred[i] == fact[i]:
            count += 1
    # returns the success rate
    return float(count)/float(pred.shape[0])
                    
if __name__ == "__main__":
    kfold(Preprocessing.process_wine(), 5)
    kfold(Preprocessing.process_cancer(), 5)