"""
Compute the coefficient and corresponding confidence interval of classifiers.
Author: Jason Liu
"""

from sklearn.ensemble import RandomForestClassifier # sklearn vs. statsmodel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from numpy import std, array, concatenate
from collections import defaultdict
from functools import partial

import csv
import numpy as np
import pandas as pd
import time


NUM_REPEATS = 16
classifiers = ['rf', 'lr']
features = ['max0','mean0','std0','rms0','arc_len0','arc_len_std0',
            'max1','mean1','std1','rms1','arc_len1','arc_len_std1']
class_weight_lr = {0: 1.0, 1: 200.0} # 'balanced' {0: 1.0, 1: 1.0} {0: 1.0, 1: 2000.0}
class_weight_rf = {0: 1.0, 1: 5000.0}
class_weight_lsvm = {0: 1.0, 1: 1000.0}
T_STAR = 1.833

def read_data(data_file):
    print('reading data')
    dmat = pd.read_csv(data_file,
                       names=['filename','label','max0','mean0','std0','rms0','arc_len0','arc_len_std0',
                                                 'max1','mean1','std1','rms1','arc_len1','arc_len_std1'],
                       dtype={'filename':str,'label':np.int32,'max0':np.float64,'mean0':np.float64,'std0':np.float64,'rms0':np.float64,'arc_len0':np.float64,'arc_len_std0':np.float64,
                                                              'max1':np.float64,'mean1':np.float64,'std1':np.float64,'rms1':np.float64,'arc_len1':np.float64,'arc_len_std1':np.float64,},
                       index_col=False)

    feature_stds = [dmat['max0'].std(), dmat['mean0'].std(), dmat['std0'].std(), dmat['rms0'].std(), dmat['arc_len0'].std(), dmat['arc_len_std0'].std(), 
                dmat['max1'].std(), dmat['mean1'].std(), dmat['std1'].std(), dmat['rms1'].std(), dmat['arc_len1'].std(), dmat['arc_len_std1'].std()] 

    trial_names = []
    labels = []
    data = []
    with open(data_file) as f:
        reader = csv.reader(f)
        for row in reader:
            trial_name = str(row[0])
            label = int(row[1])
            vector = [float(x) for x in row[2:]]
            trial_names.append(trial_name)
            labels.append(label)
            data.append(vector)

    labels = [(lambda x: 1 if x > 0 else 0)(x) for x in labels]
    trial_names = array(trial_names)
    data = array(data)
    labels = array(labels)

    print('reading data complete. X shape {}, y shape {}'.format(data.shape, labels.shape))
    return data, labels, feature_stds

def coef_conf_int(X, y, feature_stds):
    coefficients = defaultdict(partial(np.array))
    for repeat in range(NUM_REPEATS):
        print('repeat # {} start'.format(repeat))
        start_time = time.time()
        coef_row = coef(X, y, feature_stds)
        print('new row {}'.format(coef_row))
        print('repeat # {} took {}s'.format(repeat, time.time()-start_time))
        for c in classifiers:
            if repeat == 0:
                coefficients[c] = np.array(coef_row[c])
            else:
                # print('matrix shape {}, row shape {}'.format(coefficients[c].shape, coef_row[c].shape))
                coefficients[c] = np.vstack((coefficients[c], coef_row[c]))
        print(coefficients['rf'])
        print(coefficients['lr'])

    return conf_int(coefficients)

def coef(X, y, feature_stds):
    coef_row = defaultdict(lambda: list)

    # random forest weights
    random_forest_classifier = RandomForestClassifier(n_estimators=1000, class_weight=class_weight_rf)
    random_forest_classifier.fit(X, y)
    coef_row['rf'] = np.array(random_forest_classifier.feature_importances_)
    print('random forest feature importances:\n{}\n'.format(random_forest_classifier.feature_importances_))

    # logistic regression weights
    logistic_regression_classifier = LogisticRegression(class_weight=class_weight_lr)
    logistic_regression_classifier.fit(X, y)
    logistic_regression_adjusted_coef = [np.abs(coef*std) for coef, std in zip(logistic_regression_classifier.coef_, feature_stds)]
    coef_row['lr'] = logistic_regression_adjusted_coef
    print('logistic regression feature importances:\n{}\n'.format(logistic_regression_adjusted_coef))

    # # linear SVM weights
    # linear_svm_classifier = LinearSVC(class_weight=class_weight_lsvm)
    # linear_svm_classifier.fit(X, y)
    # linear_SVM_adjusted_coef = [np.abs(coef*std) for coef, std in zip(linear_svm_classifier.coef_, feature_stds)]
    # coef_row['lsvm'] = linear_SVM_adjusted_coef
    # print('linear SVM feature weights:\n{}\n'.format(linear_SVM_adjusted_coef))

    return coef_row

def conf_int(coefficients):
    mean = defaultdict(lambda: list)
    std = defaultdict(lambda: list)
    interval = defaultdict(lambda: list)
    for c in classifiers:
        mean[c] = np.mean(coefficients[c], axis=0)
        std[c] = np.std(coefficients[c], axis = 0)
        interval[c] = T_STAR / np.sqrt(NUM_REPEATS) * std[c]
    return mean, interval

def plot_bar_grasph(mean, interval):
    ind = np.arange(1, len(features)+1)
    width = 0.45

    # plot bar graph with error bar for random forest.
    plt.bar(ind, mean['rf'], width, yerr=interval['rf'], color='orange', ecolor='r', align='center')
    plt.title('Coefficients of Random Forest w/ error bars', fontsize=14, fontweight='bold')
    plt.ylabel('Coefficients', fontsize=14, fontweight='bold')
    plt.xticks(ind, features, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # plot bar graph with error bar for logistic regression.
    fig,(ax0,ax1) = plt.subplots(2, 1, sharex=True)
    fig.suptitle('Coefficients of Logistic Regression w/ error bars', fontsize=14, fontweight='bold')
    plt.ylabel('Coefficients', fontsize=14, fontweight='bold')
    plt.xticks(ind, features, fontweight='bold')
    
    ax0.bar(ind, mean['lr'], width, yerr=interval['lr'], color='b', ecolor='r', align='center')
    ax1.bar(ind, mean['lr'], width, yerr=interval['lr'], color='b', ecolor='r', align='center')
    
    ax0.set_ylim(20, 145)
    ax1.set_ylim(0, 2.8)

    ax0.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax0.xaxis.tick_top()
    ax0.tick_params(labeltop='off')
    ax1.xaxis.tick_bottom()

    d = .015
    kwargs = dict(transform=ax0.transAxes, color='k', clip_on=False)
    ax0.plot((-d,+d),(-d,+d), **kwargs)
    ax0.plot((1-d,1+d),(-d,+d), **kwargs)
    kwargs.update(transform=ax1.transAxes)
    ax1.plot((-d,+d),(1-d,1+d), **kwargs)
    ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

    plt.show()


if __name__ == '__main__':
    data_file = '/Users/JasonLiu/research/security/phone_theft_detect/data/features_win_size_1_2.csv'
    X, y, feature_stds = read_data(data_file)

    mean, interval = coef_conf_int(X, y, feature_stds)
    for c in classifiers:
        for i, feature in enumerate(features):
            print('95% confidence interval for classifier {}, feature {}: mean {}, ({}, {})'.\
                    format(c, feature, mean[c][i], mean[c][i]-interval[c][i], mean[c][i]+interval[c][i]))

    plot_bar_grasph(mean, interval)
