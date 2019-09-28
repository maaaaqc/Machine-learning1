import numpy
import time
import Preprocessing
from LogReg import LogReg
from LDA import LDA
from enum import Enum


class Model(Enum):
    logreg = 1
    lda = 2


def kfold(data, k, m):
    # shuffles the data and group them into k groups
    numpy.random.shuffle(data)
    groups = numpy.array_split(data, k, axis=0)
    # averages the acuracy of k predictions
    acc = 0
    t = 0
    for i in range(k):
        val_set = groups[i][:, 0:-1]
        true_val = groups[i][:, -1]
        train_set = numpy.concatenate(groups[:i] + groups[i+1:], axis=0)
        # timing starts here
        start = time.time()
        # build a model using the training set
        if m == Model.logreg:
            model = LogReg(train_set)
            w = model.fit(0.1, 0.005, 100)
        elif m == Model.lda:
            model = LDA(train_set)
            w = model.fit(model.cvinv, model.u)
        r = model.predict(val_set, w)
        # timing ends here
        end = time.time()
        t += (end - start)*1000
        acc += evaluate_acc(r, true_val)
    acc /= k
    t /= k
    return [acc, t]


def evaluate_acc(pred, fact):
    # counts the number of successful predictions
    count = 0
    for i in range(len(pred)):
        if pred[i] == fact[i]:
            count += 1
    # returns the success rate
    return count/len(pred)


if __name__ == "__main__":
    wine_path = str(Preprocessing.WINEDIR)
    cancer_path = str(Preprocessing.CANCERDIR)
    # prints the average accuracy of 5-fold experiments
    for i in range(10):
        logreg_wine = kfold(Preprocessing.process_wine(wine_path), 5, Model.logreg)
        print("Red wine quality prediction accuracy using LogReg: {:.2%}".format(logreg_wine[0]))
    # print("Red wine quality prediction time cost using LogReg: {:.5} miliseconds".format(logreg_wine[1]))
    # lda_wine = kfold(Preprocessing.process_wine(wine_path), 5, Model.lda)
    # print("Red wine quality prediction accuracy using LDA: {:.2%}".format(lda_wine[0]))
    # print("Red wine quality prediction time cost using LDA: {:.5} miliseconds".format(lda_wine[1]))
    # logreg_cancer = kfold(Preprocessing.process_cancer(cancer_path), 5, Model.logreg)
    # print("Tumor classification prediction accuracy using LogReg: {:.2%}".format(logreg_cancer[0]))
    # print("Tumor classification prediction time cost using LogReg: {:.5} miliseconds".format(logreg_cancer[1]))
    # lda_cancer = kfold(Preprocessing.process_cancer(cancer_path), 5, Model.lda)
    # print("Tumor classification prediction accuracy using LDA: {:.2%}".format(lda_cancer[0]))
    # print("Tumor classification prediction time cost using LDA: {:.5} miliseconds".format(lda_cancer[1]))
