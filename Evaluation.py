import numpy
import Preprocessing
from LogReg import LogReg
from LDA import LDA
from enum import Enum
from argparse import ArgumentParser


wine_path = str(Preprocessing.WINEDIR)
cancer_path = str(Preprocessing.CANCERDIR)


class Model(Enum):
    logreg = 1
    lda = 2


def kfold(data, k, m):
    # shuffles the data and group them into k groups
    numpy.random.shuffle(data)
    groups = numpy.array_split(data, k, axis=0)
    # averages the acuracy of k predictions
    acc = 0
    for i in range(k):
        val_set = groups[i][:, 0:-1]
        true_val = groups[i][:, -1]
        train_set = numpy.concatenate(groups[:i] + groups[i+1:], axis=0)
        # build a model using the training set
        if m == Model.logreg:
            model = LogReg(train_set)
            w = model.fit(0.1, 100)
        elif m == Model.lda:
            model = LDA(train_set)
            w = model.fit(model.cvinv, model.u)
        r = model.predict(val_set, w)
        acc += evaluate_acc(r, true_val)
    acc /= k
    return acc


def evaluate_acc(pred, fact):
    # counts the number of successful predictions
    count = 0
    for i in range(len(pred)):
        if pred[i] == fact[i]:
            count += 1
    # returns the success rate
    return count/len(pred)


def logreg_wine():
    logreg_wine = kfold(Preprocessing.process_wine(wine_path), 5, Model.logreg)
    print("Red wine quality prediction accuracy using LogReg: {:.2%}".format(logreg_wine))


def logreg_cancer():
    logreg_cancer = kfold(Preprocessing.process_cancer(cancer_path), 5, Model.logreg)
    print("Tumor classification prediction accuracy using LogReg: {:.2%}".format(logreg_cancer))


def lda_wine():
    lda_wine = kfold(Preprocessing.process_wine(wine_path), 5, Model.lda)
    print("Red wine quality prediction accuracy using LDA: {:.2%}".format(lda_wine))


def lda_cancer():
    lda_cancer = kfold(Preprocessing.process_cancer(cancer_path), 5, Model.lda)
    print("Tumor classification prediction accuracy using LDA: {:.2%}".format(lda_cancer))


if __name__ == "__main__":
    wine_path = str(Preprocessing.WINEDIR)
    cancer_path = str(Preprocessing.CANCERDIR)

    parser = ArgumentParser()
    parser.add_argument("-l", "--lda", help="use the lda model to run with")
    parser.add_argument("-r", "--regression", help="use the regression model to run with")
    parser.add_argument("-w", "--wine", help="run the model on wine dataset")
    parser.add_argument("-c", "--cancer", help="run the model on cancer dataset")
    args = parser.parse_args()

    if not args.lda and not args.regression and not args.wine and not args.cancer:
        logreg_wine()
        lda_wine()
        logreg_cancer()
        lda_cancer()

    if not args.lda and not args.regression:
        if args.wine:
            logreg_wine()
            lda_wine()
        if args.cancer:
            logreg_cancer()
            lda_cancer()

    if args.lda:
        if not args.wine and not args.cancer:
            lda_wine()
            lda_cancer()
        if args.wine:
            lda_wine()
        if args.cancer:
            lda_cancer()

    if args.regression:
        if not args.wine and not args.cancer:
            logreg_wine()
            logreg_cancer()
        if args.wine:
            logreg_wine()
        if args.cancer:
            logreg_cancer()
