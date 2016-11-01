from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LeaveOneOut
from sklearn import tree

from sklearn.metrics import *

from numpy import std, array, concatenate

import csv

classifiers = {
    'random forest': RandomForestClassifier,
    'logistic regression': LogisticRegression,
    'linear SVM': LinearSVC,
#    'RBF kernel': SVC,
}

featuresfile = 'features.csv'

def classify():
    labels = []
    data = []
    RESULTS = {}

    with open(featuresfile) as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[1])
            vector = [float(x) for x in row[6:]]
            labels.append(label)
            data.append(vector)

            # XXX HACK: triple number of positive samples to correct for weighting
            # if label > 0:
            #     labels.append(label)
            #     labels.append(label)
            #     data.append(vector)
            #     data.append(vector)

    # pos = [x for x in labels if x > 0]
    # neg = [x for x in labels if x < 0]
    # print('(count) positive / negative:', len(pos), '/', len(neg))

    labels = [(lambda x: 1 if x > 0 else 0)(x) for x in labels]
    # print(labels)
    # print(data)
    data = array(data)
    labels = array(labels)

    for classifier_name in classifiers:
        classifier = classifiers[classifier_name]
        print('classifier: {}'.format(classifier_name))
        RESULTS[classifier] = {}

        # old cross-validation code
        # classifier = classifier()
        # predictions = cross_val_predict(classifier, data, labels, cv=10)
        #
        # cmat = confusion_matrix(labels, predictions)

        # new cross-validation code
        real_labels = []
        predicted_labels = []

        # cv_iterator = StratifiedKFold(labels, n_folds=3, random_state=0)
        cv_iterator = LeaveOneOut(len(data))
        for train_index, test_index in cv_iterator:
            train_data, test_data = data[train_index], data[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            classifier_instance = classifier()
            classifier_instance.fit(train_data, train_labels)
            test_results = classifier_instance.predict(test_data)

            real_labels.append(test_labels)
            predicted_labels.append(test_results)

        real_labels = concatenate(real_labels)
        predicted_labels = concatenate(predicted_labels)
        cmat = confusion_matrix(real_labels, predicted_labels)

        scaled = map(lambda r: map(lambda x: float(x) / len(data), r), cmat)

        print('confusion matrix:\n{}'.format(cmat))
        RESULTS[classifier][confusion_matrix] = cmat
        # print 'false negatives: {}'.format(cmat[1][0])
        # print 'false positives: {}'.format(cmat[0][1])
        # print('normalized confusion matrix:\n{}'.format(array(scaled)))
        # print 'false negative rate: {}'.format(scaled[1][0])
        # print 'false positive rate: {}'.format(scaled[0][1])
        # print ''

        # compute accuracy
        hit = 0
        miss = 0
        for real, predicted in zip(real_labels, predicted_labels):
            if real == predicted:
                hit += 1
            else:
                miss += 1
        print('accuracy: {}'.format(float(hit) / (hit+miss)))
        RESULTS[classifier][accuracy] = float(hit) / (hit+miss)

    # random forest weights
    classifier_instance = RandomForestClassifier()
    classifier_instance.fit(data, labels)
    print(classifier_instance.feature_importances_)
    RESULTS[classifier][random_forest_weights] = classifier_instance.feature_importances_

    # linear weights
    classifier_instance = LinearSVC()
    classifier_instance.fit(data, labels)
    print(classifier_instance.coef_)
    RESULTS[classifier][linear_weights] = classifier_instance.coef_

    return RESULTS

def main():
    classify()

if __name__ == '__main__':
    main()
