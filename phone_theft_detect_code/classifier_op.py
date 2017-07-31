"""
Binary classifiers for phone theft and normal usage.
Author: Jason Liu
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import *
from numpy import std, array, concatenate
from sklearn.externals import joblib

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython

classifiers = {
    'random_forest': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    # 'linear_svm': LinearSVC,
}

# feature_file = 'data/features_win_size_1_(2)1.csv'
feature_file = 'data/features_win_size_1_2.csv'
print('dataset: ', feature_file)

class_weight = {0: 1.0, 1: 200.0} # 'balanced' {0: 1.0, 1: 1.0} {0: 1.0, 1: 2000.0}
print('class_weight: ', class_weight)

# 1. read dataset.

# dmat = pd.read_csv(feature_file,
#                     names=['filename','label','max0','mean0','std0','rms0','arc_len0','arc_len_std0','mean_abs0',
#                                               'max1','mean1','std1','rms1','arc_len1','arc_len_std1','mean_abs1',
#                                               'max2','mean2','std2','rms2','arc_len2','arc_len_std2','mean_abs2',
#                                               'max3','mean3','std3','rms3','arc_len3','arc_len_std3','mean_abs3',
#                                               'max4','mean4','std4','rms4','arc_len4','arc_len_std4','mean_abs4',
#                                               'max5','mean5','std5','rms5','arc_len5','arc_len_std5','mean_abs5',
#                                               'max6','mean6','std6','rms6','arc_len6','arc_len_std6','mean_abs6'],
#                     dtype={'filename':str,'label':np.int32,'max0':np.float64,'mean0':np.float64,'std0':np.float64,'rms0':np.float64,'arc_len0':np.float64,'arc_len_std0':np.float64,'mean_abs0':np.float64,
#                                                            'max1':np.float64,'mean1':np.float64,'std1':np.float64,'rms1':np.float64,'arc_len1':np.float64,'arc_len_std1':np.float64,'mean_abs1':np.float64,
#                                                            'max2':np.float64,'mean2':np.float64,'std2':np.float64,'rms2':np.float64,'arc_len2':np.float64,'arc_len_std2':np.float64,'mean_abs2':np.float64,
#                                                            'max3':np.float64,'mean3':np.float64,'std3':np.float64,'rms3':np.float64,'arc_len3':np.float64,'arc_len_std3':np.float64,'mean_abs3':np.float64,
#                                                            'max4':np.float64,'mean4':np.float64,'std4':np.float64,'rms4':np.float64,'arc_len4':np.float64,'arc_len_std4':np.float64,'mean_abs4':np.float64,
#                                                            'max5':np.float64,'mean5':np.float64,'std5':np.float64,'rms5':np.float64,'arc_len5':np.float64,'arc_len_std5':np.float64,'mean_abs5':np.float64,
#                                                            'max6':np.float64,'mean6':np.float64,'std6':np.float64,'rms6':np.float64,'arc_len6':np.float64,'arc_len_std6':np.float64,'mean_abs6':np.float64},
#                     index_col=False)

# feature_stds = [dmat['max0'].std(), dmat['mean0'].std(), dmat['std0'].std(), dmat['rms0'].std(), dmat['arc_len0'].std(), dmat['arc_len_std0'].std(), dmat['mean_abs0'].std(), 
#                 dmat['max1'].std(), dmat['mean1'].std(), dmat['std1'].std(), dmat['rms1'].std(), dmat['arc_len1'].std(), dmat['arc_len_std1'].std(), dmat['mean_abs1'].std(), 
#                 dmat['max2'].std(), dmat['mean2'].std(), dmat['std2'].std(), dmat['rms2'].std(), dmat['arc_len2'].std(), dmat['arc_len_std2'].std(), dmat['mean_abs2'].std(),
#                 dmat['max3'].std(), dmat['mean3'].std(), dmat['std3'].std(), dmat['rms3'].std(), dmat['arc_len3'].std(), dmat['arc_len_std3'].std(), dmat['mean_abs3'].std(),
#                 dmat['max4'].std(), dmat['mean4'].std(), dmat['std4'].std(), dmat['rms4'].std(), dmat['arc_len4'].std(), dmat['arc_len_std4'].std(), dmat['mean_abs4'].std(),
#                 dmat['max5'].std(), dmat['mean5'].std(), dmat['std5'].std(), dmat['rms5'].std(), dmat['arc_len5'].std(), dmat['arc_len_std5'].std(), dmat['mean_abs5'].std(),
#                 dmat['max6'].std(), dmat['mean6'].std(), dmat['std6'].std(), dmat['rms6'].std(), dmat['arc_len6'].std(), dmat['arc_len_std6'].std(), dmat['mean_abs6'].std()]

dmat = pd.read_csv(feature_file,
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

with open(feature_file) as f:
    reader = csv.reader(f)
    for row in reader:
        trial_name = str(row[0])
        label = int(row[1])
        vector = [float(x) for x in row[2:]]
        # vector = [float(x) for x in [row[2], row[4]]] # only look at two important features.
        trial_names.append(trial_name)
        labels.append(label)
        data.append(vector)

# 2. train and cross-validate.
labels = [(lambda x: 1 if x > 0 else 0)(x) for x in labels]
trial_names = array(trial_names)
data = array(data)
labels = array(labels)

fpr = dict()
tpr = dict()
roc_auc = dict()

for classifier_name in classifiers:
    classifier = classifiers[classifier_name]
    print('classifier: {}'.format(classifier_name))

    real_labels = []
    predicted_labels = []
    test_trial_names = []

    ten_fold = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in ten_fold.split(data):
        train_trials, test_trials = trial_names[train_index], trial_names[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        train_data, test_data = data[train_index], data[test_index]

        if classifier_name == 'random_forest':
            classifier_instance = classifier(n_estimators=1000, class_weight={0: 1.0, 1: 5000.0})
        elif classifier_name == 'logistic_regression':
            classifier_instance = classifier(class_weight=class_weight)
        elif classifier_name == 'linear_svm':
            classifier_instance = classifier(class_weight=class_weight)

        # 2.1 train classifiers on training set.
        clf = classifier_instance.fit(train_data, train_labels)
        # filename = 'data/theft_classifiers_weights/' + classifier_name + '_weights.pkl'
        # joblib.dump(clf, filename)

        # 2.2 validate classifiers on validation set.
        test_results = classifier_instance.predict(test_data)

        real_labels.append(test_labels)
        predicted_labels.append(test_results)
        test_trial_names.append(test_trials)

    # 3.1 compute confusion matrices.
    real_labels = concatenate(real_labels)
    predicted_labels = concatenate(predicted_labels)
    test_trial_names = concatenate(test_trial_names)

    cmat = confusion_matrix(real_labels, predicted_labels)
    scaled = map(lambda r: map(lambda x: float(x) / len(data), r), cmat)
    print('confusion matrix:\n{}'.format(cmat))

    # 3.2 compute accuracy
    hit = 0
    miss = 0
    fn_trials = []
    for real, predicted, test_trial_name in zip(real_labels, predicted_labels, test_trial_names):
        if real == predicted:
            hit += 1
        else:
            miss += 1
            if predicted == 0:
                fn_trials.append(test_trial_name)
            # elif predicted == 1:
            #     print("FP trial: ", test_trial_name)
    print('accuracy: {}'.format(float(hit) / (hit+miss)))

    # fn_trials.sort()
    # for fn_trial in fn_trials:
    #     print("FN trial: ", fn_trial)

    fpr[classifier_name], tpr[classifier_name], _ = roc_curve(real_labels, predicted_labels)
    roc_auc[classifier_name] = auc(fpr[classifier_name], tpr[classifier_name])


plt.figure()
lw = 2
plt.plot(fpr['logistic_regression'], tpr['logistic_regression'], color='red',
         lw=lw, label='ROC Curve of Logistic Regression (area = %0.2f)' % roc_auc['logistic_regression'])
plt.plot(fpr['random_forest'], tpr['random_forest'], color='blue',
         lw=lw, label='ROC curve of Random Forest (area = %0.2f)' % roc_auc['random_forest'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Two classifiers')
plt.legend(loc="lower right")
plt.show()

IPython.embed()

print('feature standard deviations:\n{}\n'.format(feature_stds))

# random forest weights
random_forest_classifier = RandomForestClassifier(n_estimators=1000, class_weight=class_weight)
random_forest_classifier.fit(data, labels)
print('random forest feature importances:\n{}\n'.format(random_forest_classifier.feature_importances_))

# logistic regression weights
logistic_regression_classifier = LogisticRegression(class_weight=class_weight)
logistic_regression_classifier.fit(data, labels)
logistic_regression_adjusted_coef = [coef*std for coef, std in zip(logistic_regression_classifier.coef_, feature_stds)]
print('logistic regression feature importances:\n{}\n'.format(logistic_regression_adjusted_coef))

# linear SVM weights
linear_svm_classifier = LinearSVC(class_weight=class_weight)
linear_svm_classifier.fit(data, labels)
linear_SVM_adjusted_coef = [coef*std for coef, std in zip(linear_svm_classifier.coef_, feature_stds)]
print('linear SVM feature weights:\n{}\n'.format(linear_SVM_adjusted_coef))
