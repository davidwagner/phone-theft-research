"""
Calculate classifiers' detection rates (accuracy on only pos data),
from one phone model (Nexus 6P) after training on pos and neg data (Nexus 5X).
Binary classifiers for phone theft and normal usage.
Run on theserver: ssh jasonxyliu@theserver.cs.berkeley.edu
Author: Jason Liu
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from numpy import array, concatenate
from sklearn.externals import joblib

import csv
import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    classifiers = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'linear_svm': LinearSVC,
    }

    pos_data_file = '/Users/JasonLiu/research/security/phone_theft_detect/data/features_pos.csv'
    neg_data_file = '/Users/JasonLiu/research/security/phone_theft_detect/data/features_neg.csv'

    # 120 positive data points collected using Nexus 6P. Used to calculate detection rate.
    new_pos_data_file = '/Users/JasonLiu/research/security/phone_theft_detect/data/features_win_size_1_2_new_120_pos_6p.csv'

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

    new_pos_dmat = pd.read_csv(new_pos_data_file,
                        names=['filename','label','max0','mean0','std0','rms0','arc_len0','arc_len_std0',
                                                  'max1','mean1','std1','rms1','arc_len1','arc_len_std1'],
                        dtype={'filename':str,'label':np.int32,'max0':np.float64,'mean0':np.float64,'std0':np.float64,'rms0':np.float64,'arc_len0':np.float64,'arc_len_std0':np.float64,
                                                               'max1':np.float64,'mean1':np.float64,'std1':np.float64,'rms1':np.float64,'arc_len1':np.float64,'arc_len_std1':np.float64,},
                        index_col=False)
    new_pos_dmat = new_pos_dmat[1:]

    # feature_stds = [dmat['max0'].std(), dmat['mean0'].std(), dmat['std0'].std(), dmat['rms0'].std(), dmat['arc_len0'].std(), dmat['arc_len_std0'].std(),
    #                 dmat['max1'].std(), dmat['mean1'].std(), dmat['std1'].std(), dmat['rms1'].std(), dmat['arc_len1'].std(), dmat['arc_len_std1'].std()]

    print('pos dmat shape {}'.format(pos_dmat.shape))
    print('neg dmat shape {}'.format(neg_dmat.shape))
    print('new pos dmat shape {}'.format(new_pos_dmat.shape))

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

    new_pos_X = []
    new_pos_y = []
    with open(new_pos_data_file) as f:
        reader = csv.reader(f)
        for row in reader:
            y = int(row[1])
            vector = [float(x) for x in row[2:]]
            new_pos_X.append(vector)
            new_pos_y.append(y)

    new_pos_y = [(lambda x: 1 if x > 0 else 0)(x) for x in new_pos_y]
    new_pos_X = array(new_pos_X)
    new_pos_y = array(new_pos_y)


    # Concatenate pos, neg to form train, test sets.
    pos_X_train = pos_X
    pos_y_train = pos_y
    neg_X_train = neg_X
    neg_y_train = neg_y
    X_train = np.concatenate((pos_X_train, neg_X_train), axis=0)
    y_train = np.concatenate((pos_y_train, neg_y_train), axis=0)

    X_test = new_pos_X
    y_test = new_pos_y

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

        # 3.2 compute accuracy
        hit = 0
        miss = 0
        for real, predicted in zip(y_test, y_predicted):
            if real == predicted:
                hit += 1
            else:
                miss += 1
        print('accuracy: {}'.format(float(hit) / (hit+miss)))
