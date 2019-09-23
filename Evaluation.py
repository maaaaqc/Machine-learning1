import Preprocessing
import numpy
from LogReg import LogReg
from LDA import LDA

def kfold_logreg(data, k):
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
        w = model.fit(0.05, 0.005, 500)
        r = model.predict(val_set, w)
        acc += evaluate_acc(r, true_val)
    acc /= k
    return acc

def kfold_lda(data, k):
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
        w = model.fit(model.cvinv, model.u)
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
    return count/pred.shape[0]


if __name__ == "__main__":
    wine_path = str(Preprocessing.WINEDIR)
    cancer_path = str(Preprocessing.CANCERDIR)
    print("Red wine quality prediction accuracy using LogReg: {:.2%}".format(kfold_logreg(Preprocessing.process_wine(wine_path), 5)))
    print("Tumor classification prediction accuracy using LogReg: {:.2%}".format(kfold_logreg(Preprocessing.process_cancer(cancer_path), 5)))
    print("Red wine quality prediction accuracy using LDA: {:.2%}".format(kfold_lda(Preprocessing.process_wine(wine_path), 5)))
    print("Tumor classification prediction accuracy using LDA: {:.2%}".format(kfold_lda(Preprocessing.process_cancer(cancer_path), 5)))