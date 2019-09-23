import Preprocessing
import numpy
from LogReg import LogReg

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
        model = LogReg(train_set)
        w = model.fit(0.005, 1000)
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
    print(kfold(Preprocessing.process_wine(), 5))
    print(kfold(Preprocessing.process_cancer(), 5))