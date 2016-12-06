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
    datas: a flat list of data. For acc, data is norm.

    Partitions data into windows before and after indicator times. For acc, window size is 1s.
    from [t0, t1, t2, t3, t4, t5, ...], [n0, n1, n2, n3, n4, n5, ...]
    to [[t0,t1, t2, t3], [t4, t5, ...], ...], [[n0, n1, n2, n3], [n4, n5, ...], ...]

    Returns
    -------
    window_times: a list of start times of each window.
    window_datas: a list of data of each window. For acc, data is acc norm.
    '''
    window_times = []
    window_datas = []
    
    i = 0
    while i < len(acc_data):
        if np.absolute(acc_data[i]) >= ACC_THRESHOLD:
            start_index = i

            ''' constructs window before indicator time. '''
            if start_index-100 >= 0:
                window_times += [times[start_index-100 : start_index]]
                window_datas += [acc_data[start_index-100 : start_index]]
            else:
                window_times += [times[0 : start_index]]
                window_datas += [acc_data[0 : start_index]]

            ''' constructs window after indicator time. '''
            if start_index+100 <= len(acc_data)-1:
                window_times += [times[start_index : start_index+100]]
                window_datas += [acc_data[start_index : start_index+100]]
            else:
                window_times += [times[start_index : len(times)]]
                window_datas += [acc_data[start_index : len(acc_data)]]
            i += 100
        else:
            i += 1

    return window_times, window_datas

def hist(feature, split):
    '''
    Parameters
    ----------
    feature: one of "min", max", "mean", "std", "rms", "arc-length", "arc-length*std", "mean-absolute"
    split: a list of lists (windows)

    Featurize data in each window, i.e. turn each window into a data point.
    Example function call: hist("max", [[...], [...], [...], ...]).

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
    design_matrix: rows are data points; columns are features.
    '''
    features = ["max", "mean", "std", "rms", "arc-length", "arc-length*std", "mean-absolute"] # ambient light in windows before & after grab or threshold

    design_matrix = []
    feature_vector = {}
    for feature in features:
        feature_vector[feature] = hist(feature, window_datas)
    for i in range(len(window_datas)):
        row = []
        row.append(str(i))
        row.append(str(window_times[i][0]))
        row.append(str(window_times[len(window_times[i])-1]))
        row.append(str(feature_vector["max"][i]))
        row.append(str(feature_vector["mean"][i]))
        row.append(str(feature_vector["std"][i]))
        row.append(str(feature_vector["rms"][i]))
        row.append(str(feature_vector["arc-length"][i]))
        row.append(str(feature_vector["arc-length*std"][i]))
        row.append(str(feature_vector["mean-absolute"][i]))
        design_matrix.append(row)

    return design_matrix

def acc_featurizer(acc_data):
    '''
    Parameters
    ----------
    acc_data: a list of rows in csv files.

    Master function. Constructs a design matrix.

    Returns
    -------
    design_matrix: rows are data points; columns are features.
    '''
    times = [row[0] for row in acc_data if len(row) == 4]
    norms = [LA.norm([row[1],row[2],row[3]]) for row in acc_data if len(row) == 4]
    window_times, window_datas = data_to_windows(times, norms)
    design_matrix = featurize_windows(window_times, window_datas)

def test():
    acc_data = []
    path = 'data/test'

    ''' Constructs acc_data '''
    csv_files = os.listdir(path)
    for i in range(len(csv_files)):
        csv_file = open(os.getcwd() + "/" + path + "/" + csv_files[i])
        reader = csv.reader(csv_file, delimiter=",")

        for row in reader:
            if len(row) == 4:
                acc_data.append(row)

    ''' Featurize acc_data '''
    print(acc_featurizer(acc_data))

if __name__ == '__main__':
    test()



