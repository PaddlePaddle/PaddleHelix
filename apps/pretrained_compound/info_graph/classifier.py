"""
Adapted from https://github.com/fanyun-sun/InfoGraph/blob/master/unsupervised/evaluate_embedding.py
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def logistic_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LogisticRegression(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LogisticRegression(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)


def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    ret = np.mean(accuracies)
    return ret


def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)


def eval_on_classifiers(embeddings, labels, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    # print(x.shape, y.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logreg_accuracies = [logistic_classify(x, y, search) for _ in range(1)]
        # print(logreg_accuracies)
        print('LogReg', np.mean(logreg_accuracies))

        svc_accuracies = [svc_classify(x, y, search) for _ in range(1)]
        # print(svc_accuracies)
        print('svc', np.mean(svc_accuracies))

        linearsvc_accuracies = [linearsvc_classify(x, y, search) for _ in range(1)]
        # print(linearsvc_accuracies)
        print('LinearSvc', np.mean(linearsvc_accuracies))

        randomforest_accuracies = [randomforest_classify(x, y, search) for _ in range(1)]
        # print(randomforest_accuracies)
        print('randomforest', np.mean(randomforest_accuracies))

        metrics = {
            'logreg_acc': np.mean(logreg_accuracies),
            'svc_acc': np.mean(svc_accuracies),
            'linearsvc_acc': np.mean(linearsvc_accuracies),
            'randomforest_acc': np.mean(randomforest_accuracies)
        }

        return metrics

if __name__ == '__main__':
    with open('emb_dir/epoch_10.pkl', 'rb') as f:
        data = pickle.load(f)

    eval_on_classifiers(data['emb'], data['y'], search=True)
