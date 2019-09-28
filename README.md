## Machine Learning Group Project 1
* Group memebers: Qingchuan Ma, Jiaqi Peng, Qingyue Ma
* Features: Logistic Regression, Linear Discriminative Analysis
* Main functions: 
```console
LogReg.fit() # computes the w vector for regression
LogReg.predict() # computes the estimated class value
Evaluation.evaluate_acc() # outputs the accuracy of the model
```
* To see complete results for prediction accuracy:
```console
python Evaluation.py
```
* To see results for prediction accuracy with Logistic Regression:
```console
python Evaluation.py -regression
```
* To see results for prediction accuracy with LDA:
```console
python Evaluation.py -lda
```
* To see results for prediction accuracy on wine dataset:
```console
python Evaluation.py -wine
```
* To see results for prediction accuracy on wine dataset:
```console
python Evaluation.py -cancer
```
* Note: mixed use of multiple tags like "-lda -wine" is also supported