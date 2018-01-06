featurizer_op.py: generate training and validation data set stored in features.csv
classifier_op.py: take input data from features.csv, output confusion matrices and feature rankings for random forest, logistic regression, linear SVM.
cal_detect_rate.py: train 3 classifiers using pos & neg data collected on Nexus 6X then compute detection rate of the trained classifiers using pos data collected on Nexus 6P.
plot_hist.py: plot histograms of all features used in featurizer_op.py with pos and neg data in one historgram.
plotcsv.m: plot raw accelerometer data stored in a csv file.
