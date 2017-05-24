from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# from sklearn.cross_validation import cross_val_predict
# from sklearn.cross_validation import StratifiedKFold
# from sklearn.cross_validation import LeaveOneOut
# from sklearn.cross_validation import LeavePOut
# from sklearn.model_selection import LeaveOneOut
# from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold

from sklearn import tree

from sklearn.metrics import *

from numpy import std, array, concatenate
from sklearn.externals import joblib

import csv
# import cPickle



classifiers = {
    'random_forest': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    'linear_svm': LinearSVC,
    # 'svm_guassian': SVC,
    # 'svm_poly': SVC,
    # 'svm_sigmoid': SVC
}

featuresfile = 'data/features_win_size_1_1.csv'
# featuresfile = 'data/features.csv'

trial_names = []
labels = []
data = []

with open(featuresfile) as f:
    reader = csv.reader(f)
    for row in reader:
        trial_name = str(row[0])
        label = int(row[1])
        # vector = [float(x) for x in [row[2], row[4]]]
        vector = [float(x) for x in row[2:]]
        trial_names.append(trial_name)
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
trial_names = array(trial_names)
data = array(data)
labels = array(labels)

for classifier_name in classifiers:
    classifier = classifiers[classifier_name]
    print('classifier: {}'.format(classifier_name))

    # old cross-validation code
    # classifier = classifier()
    # predictions = cross_val_predict(classifier, data, labels, cv=10)
    #
    # cmat = confusion_matrix(labels, predictions)

    # new cross-validation code
    real_labels = []
    predicted_labels = []
    test_trial_names = []

    # cv_iterator = StratifiedKFold(labels, n_folds=3, random_state=0)
    # cv_iterator = LeaveOneOut(len(data))

    ten_fold = KFold(n_splits=10, shuffle=True)
    # loo = LeaveOneOut()
    # kf = KFold(n_splits=10)

    for train_index, test_index in ten_fold.split(data):
    # for train_index, test_index in loo.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        train_trials, test_trials = trial_names[train_index], trial_names[test_index]

        if classifier_name == 'random_forest':
            classifier_instance = classifier(n_estimators=1000, class_weight={0: 1.0, 1: 1.0}) # 'balanced' {0: 1.0, 1: 1.0}
        elif classifier_name == 'logistic_regression':
            classifier_instance = classifier(class_weight={0: 1.0, 1: 1.0})
        elif classifier_name == 'linear_svm':
            classifier_instance = classifier(class_weight={0: 1.0, 1: 1.0})
        # elif classifier_name == 'svm_guassian':
        #     classifier_instance = classifier(class_weight='balanced')
        # elif classifier_name == 'svm_poly':
        #     classifier_instance = classifier(kernel='poly', class_weight='balanced')
        # elif classifier_name == 'svm_sigmoid':
        #     classifier_instance = classifier(kernel='sigmoid', class_weight='balanced')
        # classifier_instance = classifier(class_weight='balanced')

        clf = classifier_instance.fit(train_data, train_labels)
        filename = 'data/theft_classifiers_weights/' + classifier_name + '_weights.pkl'
        joblib.dump(clf, filename)

        test_results = classifier_instance.predict(test_data)

        # test pretrained weight in pickle
        # clf = joblib.load(filename)
        # print(filename)
        # test_results = clf.predict(test_data)

        real_labels.append(test_labels)
        predicted_labels.append(test_results)
        test_trial_names.append(test_trials)

    real_labels = concatenate(real_labels)
    predicted_labels = concatenate(predicted_labels)
    test_trial_names = concatenate(test_trial_names)

    cmat = confusion_matrix(real_labels, predicted_labels)
    scaled = map(lambda r: map(lambda x: float(x) / len(data), r), cmat)
    print('confusion matrix:\n{}'.format(cmat))
    # print 'false negatives: {}'.format(cmat[1][0])
    # print 'false positives: {}'.format(cmat[0][1])
    # print('normalized confusion matrix:\n{}'.format(array(scaled)))
    # print 'false negative rate: {}'.format(scaled[1][0])
    # print 'false positive rate: {}'.format(scaled[0][1])
    # print ''

    # compute accuracy
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

    fn_trials.sort()
    for fn_trial in fn_trials:
        print("FN trial: ", fn_trial)

# random forest weights
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(data, labels)
print('feature importances: {}'.format(random_forest_classifier.feature_importances_))

# logistic regression weights
logistic_regression_classifier = LogisticRegression()
logistic_regression_classifier.fit(data, labels)
print('feature importances: {}'.format(logistic_regression_classifier.coef_))

# linear SVM weights
linear_svm_classifier = LinearSVC()
linear_svm_classifier.fit(data, labels)
print('feature weights: {}'.format(linear_svm_classifier.coef_))
