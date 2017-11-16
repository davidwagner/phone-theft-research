"""
Binary classifiers for phone theft and normal usage.
Run on theserver: ssh jasonxyliu@theserver.cs.berkeley.edu
Author: Jason Liu
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from numpy import std, array, concatenate
from sklearn.externals import joblib

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import IPython

classifiers = {
    'random_forest': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    # 'linear_svm': LinearSVC,
}


pos_data_file = '/Users/JasonLiu/research/security/phone_theft_detect/data/features_pos.csv'
neg_data_file = '/Users/JasonLiu/research/security/phone_theft_detect/data/features_neg.csv'

class_weight_lr = {0: 1.0, 1: 200.0} # 'balanced' {0: 1.0, 1: 1.0} {0: 1.0, 1: 2000.0}
class_weight_rf = {0: 1.0, 1: 5000.0}
class_weight_lsvm = {0: 1.0, 1: 1000.0}
print('class_weight for logistic_regression: ', class_weight_lr)
print('class_weight for random_forest: ', class_weight_rf)
print('class_weight for linear_svm: ', class_weight_lsvm)


# 1. read positvie and negative datasets.
pos_dmat = pd.read_csv(pos_data_file,
                    names=['filename','label','max0','mean0','std0','rms0','arc_len0','arc_len_std0',
                                              'max1','mean1','std1','rms1','arc_len1','arc_len_std1'],
                    dtype={'filename':str,'label':np.int32,'max0':np.float64,'mean0':np.float64,'std0':np.float64,'rms0':np.float64,'arc_len0':np.float64,'arc_len_std0':np.float64,
                                                           'max1':np.float64,'mean1':np.float64,'std1':np.float64,'rms1':np.float64,'arc_len1':np.float64,'arc_len_std1':np.float64,},
                    index_col=False)
pos_dmat = pos_dmat[1:]

neg_dmat = pd.read_csv(neg_data_file,
                    names=['filename','label','max0','mean0','std0','rms0','arc_len0','arc_len_std0',
                                              'max1','mean1','std1','rms1','arc_len1','arc_len_std1'],
                    dtype={'filename':str,'label':np.int32,'max0':np.float64,'mean0':np.float64,'std0':np.float64,'rms0':np.float64,'arc_len0':np.float64,'arc_len_std0':np.float64,
                                                           'max1':np.float64,'mean1':np.float64,'std1':np.float64,'rms1':np.float64,'arc_len1':np.float64,'arc_len_std1':np.float64,},
                    index_col=False)

# feature_stds = [dmat['max0'].std(), dmat['mean0'].std(), dmat['std0'].std(), dmat['rms0'].std(), dmat['arc_len0'].std(), dmat['arc_len_std0'].std(), 
#                 dmat['max1'].std(), dmat['mean1'].std(), dmat['std1'].std(), dmat['rms1'].std(), dmat['arc_len1'].std(), dmat['arc_len_std1'].std()] 

print('pos dmat shape {}'.format(pos_dmat.shape))
print('neg dmat shape {}'.format(neg_dmat.shape))

pos_X = []
pos_y = []

with open(pos_data_file) as f:
    reader = csv.reader(f)
    for row in reader:
        y = int(row[1])
        vector = [float(x) for x in row[2:]]
        pos_X.append(vector)
        pos_y.append(y)

pos_y = [(lambda x: 1 if x > 0 else 0)(x) for x in pos_y]
pos_X = array(pos_X)
pos_y = array(pos_y)


neg_X = []
neg_y = []

with open(neg_data_file) as f:
    reader = csv.reader(f)
    for row in reader:
        y = int(row[1])
        vector = [float(x) for x in row[2:]]
        neg_X.append(vector)
        neg_y.append(y)

neg_y = [(lambda x: 1 if x > 0 else 0)(x) for x in neg_y]
neg_X = array(neg_X)
neg_y = array(neg_y)



fpr = dict()
tpr = dict()
roc_auc = dict()

# split pos and neg to train and test set
pos_X_train, pos_X_test, pos_y_train, pos_y_test = train_test_split(pos_X, pos_y, test_size=0.3, random_state=0)
neg_X_train, neg_X_test, neg_y_train, neg_y_test = train_test_split(neg_X, neg_y, test_size=0.3, random_state=0)


print('pos_X_train shape {}'.format(pos_X_train.shape))
print('pos_y_train shape {}'.format(pos_y_train.shape))
print('neg_X_train shape {}'.format(neg_X_train.shape))
print('neg_y_train shape {}'.format(neg_y_train.shape))
print('pos_X_test shape {}'.format(pos_X_test.shape))
print('pos_y_test shape {}'.format(pos_y_test.shape))
print('neg_X_test shape {}'.format(neg_X_test.shape))
print('neg_y_test shape {}'.format(neg_y_test.shape))

# Concatenate pos, neg to form train, test sets.
X_train = np.concatenate((pos_X_train, neg_X_train), axis=0)
y_train = np.concatenate((pos_y_train, neg_y_train), axis=0)
X_test = np.concatenate((pos_X_test, neg_X_test), axis=0)
y_test = np.concatenate((pos_y_test, neg_y_test), axis=0)

print('X_train shape {}'.format(X_train.shape))
print('y_train shape {}'.format(y_train.shape))
print('X_test shape {}'.format(X_test.shape))
print('y_test shape {}'.format(y_test.shape))


for classifier_name in classifiers:
    classifier = classifiers[classifier_name]
    print('classifier: {}'.format(classifier_name))

    if classifier_name == 'random_forest':
        classifier_instance = classifier(n_estimators=1000, class_weight=class_weight_rf)
    elif classifier_name == 'logistic_regression':
        classifier_instance = classifier(class_weight=class_weight_lr)
    elif classifier_name == 'linear_svm':
        classifier_instance = classifier(class_weight=class_weight_lsvm)

    # 2.1 train classifiers on training set.
    clf = classifier_instance.fit(X_train, y_train)
    # filename = os.path.join('/Users/JasonLiu/research/security/phone_theft_detect/data/theft_classifiers_weights', classifier_name, '_weights.pkl')
    # joblib.dump(clf, filename)

    # 2.2 test classifiers on test set.
    y_predicted = classifier_instance.predict(X_test)

    cmat = confusion_matrix(y_test, y_predicted)
    scaled = map(lambda r: map(lambda x: float(x) / len(X_test), r), cmat)
    print('confusion matrix:\n{}'.format(cmat))

    # 3.2 compute accuracy
    hit = 0
    miss = 0
    for real, predicted in zip(y_test, y_predicted):
        if real == predicted:
            hit += 1
        else:
            miss += 1
    print('accuracy: {}'.format(float(hit) / (hit+miss)))


    if classifier_name == 'random_forest':
        y_scores = classifier_instance.predict_proba(X_test)[:, 1]
    elif classifier_name == 'logistic_regression':
        y_scores = classifier_instance.decision_function(X_test)

    fpr[classifier_name], tpr[classifier_name], _ = roc_curve(y_test, y_scores)
    roc_auc[classifier_name] = auc(fpr[classifier_name], tpr[classifier_name])


plt.figure()
lw = 2
plt.plot(fpr['logistic_regression'], tpr['logistic_regression'], color='red', #'ro',
         lw=lw, label='ROC Curve of Logistic Regression (area = %0.2f)' % roc_auc['logistic_regression'])
plt.xscale('log')
plt.plot(fpr['random_forest'], tpr['random_forest'], color='blue', #'bo',
         lw=lw, label='ROC curve of Random Forest (area = %0.2f)' % roc_auc['random_forest'])
plt.xscale('log')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Curves of Two classifiers', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.show()


# print('feature standard deviations:\n{}\n'.format(feature_stds))

# # random forest weights
# random_forest_classifier = RandomForestClassifier(n_estimators=1000, class_weight=class_weight_rf)
# random_forest_classifier.fit(data, labels)
# print('random forest feature importances:\n{}\n'.format(random_forest_classifier.feature_importances_))

# # logistic regression weights
# logistic_regression_classifier = LogisticRegression(class_weight=class_weight_lr)
# logistic_regression_classifier.fit(data, labels)
# logistic_regression_adjusted_coef = [coef*std for coef, std in zip(logistic_regression_classifier.coef_, feature_stds)]
# print('logistic regression feature importances:\n{}\n'.format(logistic_regression_adjusted_coef))

# # linear SVM weights
# linear_svm_classifier = LinearSVC(class_weight=class_weight_lsvm)
# linear_svm_classifier.fit(data, labels)
# linear_SVM_adjusted_coef = [coef*std for coef, std in zip(linear_svm_classifier.coef_, feature_stds)]
# print('linear SVM feature weights:\n{}\n'.format(linear_SVM_adjusted_coef))
