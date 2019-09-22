import Preprocessing
import numpy
from LogReg import LogReg

def kfold(data, k):
    numpy.random.shuffle(data)
    groups = numpy.array_split(data, k, axis=0)
    for i in range(k):
        val_set = groups[i][:,0:-1]
        true_val = groups[i][:,-1]
        train_set = numpy.concatenate(groups[:i] + groups[i+1:], axis=0)
        model = LogReg(train_set)
        w = model.fit(0.005, 1000)
        r = model.predict(val_set, w)
        print(get_accuracy(r, true_val))

def get_accuracy(pred, fact):
    count = 0
    for i in range(pred.shape[0]):
        if pred[i] == fact[i]:
            count += 1
    return float(count)/float(pred.shape[0])

if __name__ == "__main__":
    kfold(Preprocessing.process_wine(), 5)
    kfold(Preprocessing.process_cancer(), 5)