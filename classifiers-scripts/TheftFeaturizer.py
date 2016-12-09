import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from numpy import linalg as LA

ACC_THRESHOLD = 40

def data_to_windows(times, datas):
    '''
    Parameters
    ----------
    times: a flat list of times.
    datas: a flat list of corresponding data. For acc, data is norm.

    Partitions data into windows before and after indicator times. 
    For acc, indicator time is the time when acc norm exceeds 40; window size is 1s.
    from [t0, t1, t2, t3, t4, t5, ...], [n0, n1, n2, n3, n4, n5, ...] (indicator time is t3)
    to [[t0, t1, t2, t3], [t4, t5, ...], ...], [[n0, n1, n2, n3], [n4, n5, ...], ...]

    Returns
    -------
    window_times: a list of times of each window.
    window_datas: a list of correspoding data of each window. For acc, data is acc norm.
    '''
    window_times = []
    window_datas = []

    i = 0
    while i < len(datas):
        if np.absolute(datas[i]) >= ACC_THRESHOLD:
            start_index = i

            ''' constructs window before indicator time. '''
            if start_index-100 >= 0:
                window_times += [times[start_index-100 : start_index+1]]
                window_datas += [datas[start_index-100 : start_index+1]]
            else:
                window_times += [times[0 : start_index+1]]
                window_datas += [datas[0 : start_index+1]]

            ''' constructs window after indicator time. '''
            if start_index+100 < len(datas):
                window_times += [times[start_index : start_index+101]]
                window_datas += [datas[start_index : start_index+101]]
            else:
                window_times += [times[start_index : len(times)]]
                window_datas += [datas[start_index : len(datas)]]
            i += 100
        else:
            i += 1

    return window_times, window_datas

def featurizer_by_feature(feature, split):
    '''
    Parameters
    ----------
    feature: one of "min", max", "mean", "std", "rms", "arc-length", "arc-length*std", "mean-absolute"
    split: a list of lists (windows)

    Featurize data in each window, i.e. turn each window into a data point.
    Example function call: featurizer_by_feature("max", [[...], [...], [...], ...]).

    Returns
    -------
    a list of featurized data points.
    '''
    if feature == "min":
        return [np.min(lst) for lst in split]
    if feature == "max":
        return [np.max(lst) for lst in split]
    if feature == "mean":
        return [np.mean(lst) for lst in split]
    if feature == "std":
        return [np.std(lst) for lst in split]
    if feature =="rms":
        return [np.sqrt(np.mean(np.square(lst))) for lst in split]
    if feature == "arc-length":
        x = []
        for lst in split:
            sum_absolutes = 0
            for i in range(0, len(lst)-1):
                sum_absolutes += np.absolute(lst[i+1] - lst[i])
            x.append(sum_absolutes / (len(lst)-1))
        return x
    if feature == "arc-length*std":
        x = []
        for lst in split:
            std = np.std(lst)
            sum_absolutes = 0
            for i in range(0, len(lst)-1):
                sum_absolutes += np.absolute(lst[i+1] - lst[i])
            x.append(sum_absolutes / (len(lst)-1) * std)
        return x
    if feature == "mean-absolute":
        return [np.mean(np.absolute(lst)) for lst in split]
    # if feature == "steps":
    #     return split
    # if feature == "zero-crossings":
    #     return [len(np.where(lst == 0)[0]) for lst in split]
    # if feature == "slope-sign-change":
    #     return [np.std(lst) for lst in split]
    # if feature =="min-absolute":
    #     return [np.min(np.absolute(lst)) for lst in split]
    # if feature =="max-absolute":
    #     return [np.max(np.absolute(lst)) for lst in split]
    

def featurize_windows(window_times, window_datas):
    '''
    Parameters
    ----------
    window_times: output by data_to_windows.
    window_datas: output by data_to_windows.

    Constructs a design matrix.

    Returns
    -------
    X: design_matrix. rows are data points; columns are features.
    times_for_data_pts: start and end times of corresponding data points.
    '''
    features = ["max", "mean", "std", "rms", "arc-length", "arc-length*std", "mean-absolute"] # ambient light in windows before & after grab or threshold

    X = []
    times_for_data_pts = []
    feature_vector = {}
    for feature in features:
        feature_vector[feature] = featurizer_by_feature(feature, window_datas)
    for i in range(len(window_datas)):
        row = []
        # row.append(i)
        times_for_data_pts.append([window_times[i][0], window_times[i][len(window_times[i])-1]])
        row.append(feature_vector["max"][i])
        row.append(feature_vector["mean"][i])
        row.append(feature_vector["std"][i])
        row.append(feature_vector["rms"][i])
        row.append(feature_vector["arc-length"][i])
        row.append(feature_vector["arc-length*std"][i])
        row.append(feature_vector["mean-absolute"][i])
        X.append(row)

    return times_for_data_pts, X

def acc_featurizer(datas):
    '''
    Parameters
    ----------
    datas: a list of rows in csv files.

    Master function. Constructs a design matrix.

    Returns
    -------
    X: design_matrix, rows are data points; columns are features.
    '''

    times = [row[0] for row in datas if len(row) >= 4]
    norms = [LA.norm([row[1],row[2],row[3]]) for row in datas if len(row) >= 4]

    window_times, window_datas = data_to_windows(times, norms)

    times_for_data_pts, X = featurize_windows(window_times, window_datas)

    return times_for_data_pts, X

def test():
    datas = []
    path = 'data/test'

    ''' Constructs datas '''
    csv_files = os.listdir(path)
    for i in range(len(csv_files)):
        csv_file = open(os.getcwd() + "/" + path + "/" + csv_files[i])
        reader = csv.reader(csv_file, delimiter=",")

        for row in reader:
            if len(row) == 4:
                datas.append(row)

    ''' Featurize datas '''
    print(acc_featurizer(datas))

if __name__ == '__main__':
    test()
