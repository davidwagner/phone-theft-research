
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from collections import Counter
from sklearn.metrics import zero_one_loss

import numpy as np
import csv
import os
import argparse
import pickle
import pandas as pd

import process_diary_study_all_data

WINDOW_SIZE = 50
MAX_TRIM = 2500

NUM_CLASSES = 5
CLASSES =  [
        "Backpack Classifier",
        "Pocket Classifier",
        "Bag Classifier",
        "Hand Classifier",
        "Table Classifier"
    ]

NPZ_NAME = "TRAIN_DIARY_STUDY_NoDerivative_NoSteady_multi.npz"
RAW_NPZ_NAME = "TRAIN_DIARY_STUDY_COMPILED_DATA.npz"

NPZ_NO_BOUNDARY_NAME = "TRAIN_DIARY_STUDY_NoDerivative_NoSteady_multi_NO_BOUNDARY.npz"
RAW_NPZ_NO_BOUNDARY_NAME = "TRAIN_DIARY_STUDY_COMPILED_DATA_NO_BOUNDARY.npz"

RAW_TEST_NPZ_NAME = "TEST_DIARY_STUDY_COMPILED_DATA_1.npz"
TEST_NPZ_NAME = "TEST_DIARY_STUDY_NoDerivative_NoSteady_multi_2.npz"

CHECK_MISCLF_LOG = "multiclass_results_log.pkl"

TRAINING_DATA = [
#   (dir_with_diary_data, diary_log_file_corresponding_to_data, user_id_of_data)
    ('../data/diary-study-3-19/diary_data/', '../data/diary-study-3-19/diary_state_3_19.txt', 'd792b61e'),
    ('../data/diary-study-11-14-15/diary_data/', '../data/diary-study-11-14-15/diary_state_no_steady.txt', 'd792b61e'),
    ('../data/diary-study-2-20/diary_data/', '../data/diary-study-2-20/diary_state_2_20.txt', 'd792b61e'),
    ('../data/diary-study-3-13/diary_data/', '../data/diary-study-3-13/diary_state_3_13.txt', 'd792b61e')
]

TEST_DATA = [
    # ('../data/4-25/diary_data/', '../data/4-25/diary_state.txt', 'd792b61e')
    ('../data/won_diary_1/diary_data/', '../data/won_diary_1/diary_state_3_13_won.txt', '34341b69')
    # ('../data/won_diary_2/diary_data/', '../data/won_diary_2/diary_state.txt', '34341b69')
]

DATA_BY_CLASS = [
    'BackpackClassifier_ALL.npz',
    'HandClassifier_ALL.npz',
    'PocketClassifier_ALL.npz',
    'TableClassifier_ALL.npz'
]


def train_model_kfold_multiclass(k=20, epochs=10, check_misclf=False, save_path='weights.h5', model_path=None, evaluate_test=False, results_dir='./', even_class_folds=False):
    if not even_class_folds:
        #Read from .npz
        data = np.load(NPZ_NO_BOUNDARY_NAME)
        conv_data = data['conv_data']
        dense_data = data['dense_data']
        labels = data['labels']


        #Converts labels to one-hot vector
        labels = keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

        conv_data, dense_data = add_more_features(conv_data, dense_data)
    else:
        data_by_class = {}
        for label in CLASSES:
            if label == "Bag Classifier":
                continue
            npz_name = "./class_npzs/" + label.replace(" ", "") + "_ALL.npz"
            data = np.load(npz_name)

            conv_data = data['conv_data']
            dense_data = data['dense_data']
            labels = data['labels']

            if label == "Table Classifier":
                conv_data = conv_data[:63050,:,:]
                dense_data = dense_data[:63050,:]
                labels = labels[:63050]
            
            #Converts labels to one-hot vector
            labels = keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

            conv_data, dense_data = add_more_features(conv_data, dense_data)

            print(label)
            for thing in (conv_data, dense_data, labels):
                print(thing.shape)

            data_by_class[label] = (conv_data, dense_data, labels)



    if evaluate_test:
        test_data = np.load(TEST_NPZ_NAME)
        test_conv = test_data['conv_data']
        test_dense = test_data['dense_data']
        test_labels = test_data['labels']

        test_labels = keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)

        test_conv, test_dense = add_more_features(test_conv, test_dense)

    
    if k == 0:
        train_conv = conv_data
        train_dense = dense_data
        train_labels = labels
        if model_path is None: 
            model = create_model(train_conv.shape[2], train_dense.shape[1])
            hist = model.fit([train_conv, train_dense], [labels], epochs=epochs)

            model.save(save_path)
            # print(hist.history['val_acc'][-1])
        else:
            model = load_model(model_path)

        if evaluate_test:
            predictions = model.predict([test_conv, test_dense])
            # mask = predictions != test_labels
            # mask.sum() / mask.shape[0] = misaccuracy

            train_dist = Counter(np.argmax(train_labels, axis=1).flatten())
            test_dist = Counter(np.argmax(test_labels, axis=1).flatten())
            print("Training dist.", train_dist)
            print("Test dist.", test_dist)

            # data_distributions.append((train_dist, test_dist))

            miss_mat = predict_and_check(model, [test_conv, test_dense], test_labels)
            # miss_mat = predict_and_check(model, [val_dense], val_labels)

            num_misses = np.sum(miss_mat)

            # miss_matrix = miss_mat / num_misses if num_misses > 0 else miss_mat 
            # miss_matrices.append(miss_mat)

            # for label, count in val_dist.items():
            #     if count > 500:
            #         result = miss_results[label] # ([], [])
            #         result[0].append(miss_mat) # misses by expected label
            #         result[1].append(k)


            worst_cases = np.argwhere(miss_mat > 5)
            filters = [tuple(indices) for indices in worst_cases]
            if len(filters) > 0:
                compare_cases = set([])

                for filt in filters:
                    compare_cases.add((filt[0], filt[0]))
                    compare_cases.add((filt[1], filt[1]))

                filters += list(compare_cases)

                print("FILTERS:", filters)

                data = {'conv' : test_conv, 'dense' : test_dense}

                # val_predictions = model.predict([val_conv, val_dense])
                # val_predictions = model.predict([val_dense])
                results_f = 'misclf_TEST' '.npz'
                find_misclassified_data(data, predictions, test_labels, results_f, filters=filters, 
                    training_data_map={'conv' : train_conv, 'dense' : train_dense}, training_labels=train_labels)

            conf_mat = confusion_matrix(model, [test_conv, test_dense], test_labels)

            print("CONFUSION MAT")
            print((conf_mat / np.sum(conf_mat)).round(3))

            M = miss_mat
            print("Total Misses")
            print(M.round(3))
            print("Proportion of Misses")
            print((M / np.sum(M)).round(3))

            # for label, results in enumerate(miss_results):
            #     print("Misses breakdown for:", CLASSES[label])
            #     miss_mat_s, k_s = results

            #     if len(miss_mat_s) <= 0:
            #         continue

            #     print("Misses by expected label")
            #     by_expected_label = np.array([np.nan_to_num(m / m.sum(axis=0)) for m in miss_mat_s]).mean(axis=0)
            #     print(by_expected_label.round(3))

            #     print("Misses by total misses")
            #     by_total_misses = np.array([np.nan_to_num(m / m.sum()) for m in miss_mat_s]).mean(axis=0)
            #     print(by_total_misses.round(3))

        
    else: 
        accuracies = []
        data_distributions = [] # (training, validation)
        indices = []
        
        # perm = np.random.permutation(n)
        # accel_data = accel_data[perm]
        # phone_active_data = phone_active_data[perm]
        # labels = labels[perm]
        
        miss_matrices = []
        confusion_matrices = []

        miss_results = [([], []) for _ in range(len(CLASSES))] # 0/1/2/3/4 -> ([], [])
        num_folds = k
        for k in range(num_folds):
            # k = fold // fold_size
            print("FOLD:", k)


            if not even_class_folds:
                # skip folds we're fine on
                if k <= 5:
                    continue

                fold_size = conv_data.shape[0] // num_folds
                fold = k * fold_size
                n = conv_data.shape[0]

                start, end = fold, fold + fold_size if k < num_folds - 1 else n
                print("GET DATA <", start, ">=", end)
                mask = np.arange(n)

                val_mask = (mask >= start) & (mask < end)
                train_mask = (mask < start) | (mask >= end)

                val_conv = conv_data[val_mask]
                val_dense = dense_data[val_mask]
                val_labels = labels[val_mask]

                train_conv = conv_data[train_mask]
                train_dense = dense_data[train_mask]
                train_labels = labels[train_mask]

            else:
                if k != 9:
                    continue
                val_conv = []
                val_dense = []
                val_labels = []

                train_conv = []
                train_dense = []
                train_labels = []
                
                for label, data in data_by_class.items():
                    conv_data, dense_data, labels = data

                    n = conv_data.shape[0]
                    fold_size = n // num_folds
                    fold = k * fold_size

                    start, end = fold, fold + fold_size if k < num_folds - 1 else n
                    print(label, start, end)

                    mask = np.arange(n)
                    val_mask = (mask >= start) & (mask < end)
                    train_mask = (mask < start) | (mask >= end)

                    val_conv.append(conv_data[val_mask])
                    val_dense.append(dense_data[val_mask])
                    val_labels.append(labels[val_mask])

                    train_conv.append(conv_data[train_mask])
                    train_dense.append(dense_data[train_mask])
                    train_labels.append(labels[train_mask])

                val_conv = np.concatenate(val_conv)
                print("VAL:", val_conv.shape)

                val_dense = np.concatenate(val_dense)
                val_labels = np.concatenate(val_labels)

                train_conv = np.concatenate(train_conv)
                print("TRAIN:", train_conv.shape)
                train_dense = np.concatenate(train_dense)
                train_labels = np.concatenate(train_labels)


            # add more features
            # is_flat = lambda row : 1.0 if row[9] < 0.3 and row[10] < 0.3 and abs(row[11] - 9.8) < 1.5 else 0.0
            # real_flatness = np.apply_along_axis(is_flat, 1, val_dense)
            # real_flatness = real_flatness.reshape((real_flatness.shape[0], 1))
            # val_dense = np.concatenate([val_dense, real_flatness], axis=1)

            # real_flatness = np.apply_along_axis(is_flat, 1, train_dense)
            # real_flatness = real_flatness.reshape((real_flatness.shape[0], 1))
            # train_dense = np.concatenate([train_dense, real_flatness], axis=1)

            # train_conv = train_conv[:,:,:2]
            # val_conv = val_conv[:,:,:2]

            # Use only some features
            # train_dense = train_dense[:,-1:]
            # val_dense = val_dense[:,-1]
            # train_conv = train_conv[:,:,:3]
            # val_conv = val_conv[:,:,:3]


            training_data = (
                            {
                            'conv_input' : train_conv, 
                            'dense_input' : train_dense
                            },
                            {'output' : train_labels}
            )

            validation_data = (
                {
                'conv_input': val_conv, 
                 'dense_input' : val_dense
                 },
                {'output' : val_labels}

            )

            model = create_model(train_conv.shape[2], train_dense.shape[1])

            hist = model.fit(training_data[0], training_data[1], epochs=epochs, validation_data=validation_data)
            accuracies.append(hist.history['val_acc'][-1])
            print(hist.history['val_acc'][-1])

            conf_mat = confusion_matrix(model, [val_conv, val_dense], val_labels)
            confusion_matrices.append(conf_mat)

            print("CONFUSION MAT")
            print(conf_mat)

            
            if check_misclf:
                train_dist = Counter(np.argmax(train_labels, axis=1).flatten())
                val_dist = Counter(np.argmax(val_labels, axis=1).flatten())
                print("Training dist.", train_dist)
                print("Validation dist.", val_dist)

                data_distributions.append((train_dist, val_dist))

                indices.append((start, end))

                miss_mat = predict_and_check(model, [val_conv, val_dense], val_labels)
                print("MISS MAT")
                print(miss_mat)
                # miss_mat = predict_and_check(model, [val_dense], val_labels)

                num_misses = np.sum(miss_mat)

                # miss_matrix = miss_mat / num_misses if num_misses > 0 else miss_mat 
                miss_matrices.append(miss_mat)

                for label, count in val_dist.items():
                    if count > 500:
                        result = miss_results[label] # ([], [])
                        result[0].append(miss_mat) # misses by expected label
                        result[1].append(k)

                # Save misclassified data to analyze
                # pred Table but really Backpack, pred Backpack but really Pocket
                # folds_of_note = {
                #     9: [(0, 1), (1, 1), (0, 0), (0, 4), (4, 4)],
                #     8: [(4, 0), (0, 0), (4, 4)],
                #     6: [(4, 0), (0, 0), (4, 4), (4, 3), (3, 3)],
                #     # 7: [(0, 4), (0, 0), (4, 4)],
                #     0: [(0, 1), (1, 1), (0, 0), (0, 4), (4, 4)],

                # }
                # if k in folds_of_note:
                #     print("LOOKING AT MISCLASSIFIED DATA")
                #     filters = folds_of_note[k]
                #     data = {'conv' : val_conv, 'dense' : val_dense}
                #     val_predictions = model.predict([val_conv, val_dense])
                #     results_f = 'misclf_data_fold_' + str(k) + '_' + str(num_folds) + 'folds' '.npz'
                #     find_misclassified_data(data, val_predictions, val_labels, results_f, filters=filters)

                worst_cases = np.argwhere(miss_mat > 1000)
                filters = [tuple(indices) for indices in worst_cases]
                if len(filters) > 0:
                    compare_cases = set([])

                    for filt in filters:
                        compare_cases.add((filt[0], filt[0]))
                        compare_cases.add((filt[1], filt[1]))

                    filters += list(compare_cases)

                    print("FILTERS:", filters)

                    data = {'conv' : val_conv, 'dense' : val_dense}

                    val_predictions = model.predict([val_conv, val_dense])
                    # val_predictions = model.predict([val_dense])
                    results_f = 'misclf_data_fold_' + str(k) + '_' + str(num_folds) + 'folds_1000_misses' '.npz'
                    find_misclassified_data(data, val_predictions, val_labels, results_f, 
                        filters=filters, training_data_map={'conv' : train_conv, 'dense' : train_dense}, training_labels=train_labels)

        
        M = np.mean(np.array(miss_matrices), axis=0)
        # M /= M.sum()
        print(M.round(3))

        C = np.mean(np.array(confusion_matrices), axis=0)
        print("CONFUSION MATRIX ALL")
        print(C.round(3))
        for i in [0, 1, 3, 4]:
            C[:,i] /= C[:,i].sum()

        print(C.round(3))

        print("Accuracies", accuracies)

        print("CROSS VAL. ACCURACY:", sum(accuracies) / len(accuracies))

        if check_misclf:
            # Check where most misses are coming from per label, and how many of total misses they account for
            for label, results in enumerate(miss_results):
                print("Misses breakdown for:", CLASSES[label])
                miss_mat_s, k_s = results

                if len(miss_mat_s) <= 0:
                    continue

                print("Misses by expected label")
                by_expected_label = np.array([np.nan_to_num(m / m.sum(axis=0)) for m in miss_mat_s]).mean(axis=0)
                print(by_expected_label.round(3))

                print("Misses by total misses")
                by_total_misses = np.array([np.nan_to_num(m / m.sum()) for m in miss_mat_s]).mean(axis=0)
                print(by_total_misses.round(3))

            # for label, results in enumerate(miss_results):
            #     print("Misses breakdown for:", CLASSES[label])
            #     miss_mat_s, k_s = results

            #     if len(miss_mat_s) <= 0:
            #         continue

            #     for miss_mat, k in zip(miss_mat_s, k_s):
            #         train_dist, val_dist = data_distributions[k]
            #         print("Train:", train_dist)
            #         print("Val:", val_dist)
            #         print("Num misses:", miss_mat.sum())
            #         print(np.nan_to_num(miss_mat / miss_mat.sum(axis=0).round(3)))
            #         print(np.nan_to_num(miss_mat / miss_mat.sum()).round(3))



            f = open(CHECK_MISCLF_LOG, 'wb+')
            results = list(zip(accuracies, indices, data_distributions, miss_matrices))
            results = sorted(results, key=lambda x: x[0])

            pickle.dump(results, f)

def create_model(num_conv_features, num_dense_features):
    # create model
    conv_input = keras.layers.Input(shape=(WINDOW_SIZE, num_conv_features), name='conv_input')
    first = Conv1D(64, 3,  activation='relu', use_bias=True, name="first_1D")(conv_input)
    first = Conv1D(64, 3,  activation='relu', use_bias=True, name="second_1D")(first)
    pool_first = MaxPooling1D(3,name="first_max_pool")(first)
    dropout_pool = Dropout(0.3, name="dropout_max_pool")(pool_first)
    third = Conv1D(128, 3, activation='relu', name="third_1D")(dropout_pool)
    fourth = Conv1D(128, 3, activation='relu', name="fourth_1D")(third)
    global_pool = GlobalAveragePooling1D(name="global_pool")(fourth)
    dropout_global = Dropout(0.5, name="dropout_global")(global_pool)
    print("D_FIRST:", dropout_global.shape)

    dense_input = keras.layers.Input(shape=(num_dense_features,), name='dense_input')
    print("DENSE:", dense_input.shape)

    dense_conv = keras.layers.concatenate([dropout_global, dense_input], name='concatenate', axis=-1)
    fully_connected_1 = Dense(64, activation='relu', name="fully_connected_1")(dense_conv)
    fully_connected_2 = Dense(64, activation='relu', name="fully_connected_2")(fully_connected_1)
    fully_connected_3 = Dense(64, activation='relu', name="fully_connected_3")(fully_connected_2)
    # dropout_5 = Dropout(0.3, name="dropout_dense_5")(fully_connected_5)
    fully_connected_4 = Dense(64, activation='relu', name="fully_connected_4")(fully_connected_3)
    fully_connected_5 = Dense(64, activation='relu', name="fully_connected_5")(fully_connected_4)
    dropout_5 = Dropout(0.3, name="dropout_dense_5")(fully_connected_5)
    fully_output = Dense(64, activation='relu', name="fully_connected_output")(dropout_5)
    output = Dense(NUM_CLASSES, activation='softmax', name='output')(fully_output)
    # output = Flatten(name='flatten')(output_without_flatten)
    model = Model(inputs=[conv_input, dense_input], outputs=[output])

    #compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def long_seq_of_true(S):
    start, end, max_length = 0, 0, 0
    i, j = 0, 0 
    seqs = []

    while j < S.shape[0]:
        if S[j]:
            j += 1
        else:
            length = j - i
            if length > max_length:
                max_length = length
                start = i
                end = j

            if length > 0:
                seqs.append((i, j))

            i = j + 1
            j += 1
    length = j - i
    if length > max_length:
        max_length = length
        start = i
        end = j

    return start, end, seqs


def is_flat(row):
    if (row[9] < 2.0 and 
        row[10] < 2.0 and 
        abs(row[11] - 9.8) < 1.5 and 
        row[6] < 0.1 and 
        row[7] < 0.1 and 
        row[8] < 0.1):
        return 1.0
    else:
        return 0.0

def add_more_features(conv_data, dense_data):
    print(dense_data.shape)
    # Add more features
    real_flatness = np.apply_along_axis(is_flat, 1, dense_data)
    real_flatness = real_flatness.reshape((real_flatness.shape[0], 1))
    dense_data = np.concatenate([dense_data, real_flatness], axis=1)

    return conv_data, dense_data

def confusion_matrix(model, data, labels):
    predictions = model.predict(data)
    predictions = predictions.argmax(axis=1)

    labels = np.argmax(labels, axis=1).flatten()

    confusion_matrix = []
    for label in range(len(CLASSES)):
        mask = labels == label
        predictions_with_label = predictions[mask]
        pred_dist_for_label = np.bincount(predictions_with_label)
        pred_dist_for_label = np.append(pred_dist_for_label, np.zeros(5 - pred_dist_for_label.shape[0])).reshape((5, 1))
        confusion_matrix.append(pred_dist_for_label)
    return np.concatenate(confusion_matrix, axis=1)

def predict_and_check(model, data, labels):
    predictions = model.predict(data)

    labels = np.argmax(labels, axis=1).flatten()

    num_per_class = Counter(labels)
    # print("NUMPERCLASS:", num_per_class)

    miss_by_class = Counter()
    acc_by_pred = Counter()
    miss_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    # Vectorized
    predictions = predictions.argmax(axis=1)
    mask = predictions != labels


    #### Find bad items ###
    bad_mask = labels == 4
    start, end, seqs = long_seq_of_true(bad_mask)
    print("Bad mask", start, end, seqs)
    worse_mask = (labels == 4) & (predictions == 0) 
    start, end, seqs = long_seq_of_true(worse_mask)
    print("Worse mask", start, end, seqs)

    misclf_preds, misclf_labels = predictions[mask], labels[mask]


    for pred, label in zip(misclf_preds, misclf_labels):
        miss_matrix[pred][label] += 1
        miss_by_class[label] += 1
        acc_by_pred[pred] += 1

    miss_mat = miss_matrix.copy()

    for i, lbl in enumerate(CLASSES):
        print(i, lbl)

    print("Misses by expected label:")
    miss_matrix_by_exp_label = miss_matrix.copy()
    for label, count in miss_by_class.items():
        miss_matrix_by_exp_label[:,label] /= float(count)
    print(miss_matrix_by_exp_label.round(3))
    
    num_misses = np.sum(miss_matrix)

    if num_misses > 0:
        miss_matrix /= np.sum(miss_matrix)

    print("Total misses:", num_misses)
    print("Misses by total misses")

    print(miss_matrix.round(3))
    # for label, row in zip(CLASSES, miss_matrix.round(3)):
    #     print(label, "\t" * 2, row)
    
    return miss_mat

def find_misclassified_data(data_map, predictions, labels, results_f, training_data_map=None, training_labels=None, filters=None):
    labels = np.argmax(labels, axis=1).flatten()
    predictions = predictions.argmax(axis=1)
    mask = predictions != labels

    if training_labels is not None:
        training_labels = np.argmax(training_labels, axis=1).flatten()
    
    if filters is not None:
        results = {}
        for actual_label, expected_label in filters:
            filtered_mask = (predictions == actual_label) & (labels == expected_label)

            for data_name, data in data_map.items():
                misclassified = data[filtered_mask]

                key = 'pred_' + str(actual_label) + '_exp_' + str(expected_label) + '_' + str(data_name)
                results[key] = misclassified

            if training_data_map is not None and training_labels is not None:
                mask = training_labels == expected_label
                for data_name, data in training_data_map.items():
                    key = 'training_label_' + str(expected_label) + '_' + str(data_name)
                    if key not in results:
                        results[key] = data[mask]

    else:
        misclassified = data[mask]
        results['all_misclf'] = misclassified

    np.savez_compressed(results_f, **results)

    return results



def blockshaped(arr, nrows, ncols):
    """
    https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays/16858283#16858283
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def get_data(dir_diary_user_trips, calibrate_accel=False, no_boundary=False, test_npz=None):
    all_data = []
    data_cols = []

    print("Calibrate accel in get data?", calibrate_accel)
    
    for data_dir, diary_file, user_id in dir_diary_user_trips:
        print("Getting data from:", data_dir)
        d, cols = process_diary_study_all_data.getAllUserData(data_dir, diary_file, user_id, as_matrix=False, calibrate_accel=calibrate_accel, no_boundary=no_boundary)

        print("DATA", d.shape, cols)
        print(d["Class"])
        d = d[d["Class"] != '']

        labels = d["Class"].values

        print(data_dir)
        print([(key, value // WINDOW_SIZE) for key, value in Counter(labels.flatten()).items()])

        all_data.append(d)
        data_cols = cols
        
    all_data = pd.concat(all_data)
    if calibrate_accel:
        all_data = process_diary_study_all_data.calibrate_accel(all_data)

    labels = all_data["Class"].values
    print("ALL DATA")
    print([(key, value // WINDOW_SIZE) for key, value in Counter(labels.flatten()).items()])
    
    if test_npz is not None:
        np.savez(test_npz, all_data=all_data, data_cols=data_cols)
    elif no_boundary:
        np.savez(RAW_NPZ_NO_BOUNDARY_NAME, all_data=all_data, data_cols=data_cols)
    else:
        np.savez(RAW_NPZ_NAME, all_data=all_data, data_cols=data_cols)

    return all_data, data_cols


def get_features_and_labels(d, cols, npz_name=NPZ_NAME):
    
    # Replace labels with ints
    d['Label'] = 0

    
    for i, label in enumerate(CLASSES):
        d.loc[d["Class"].str.strip() == label, "Label"] = i

    print("Label count:", Counter(d["Label"]))
    print("Class count:", Counter(d["Class"].str.strip()))

    # Use only the relevant columns
    data = d[['X accel.', 'Y accel.', 'Z accel.',
    #  'X direction changed', 'Y direction changed', 'Z direction changed',
     'Num. Touches', 'Screen State', 'isUnlocked',
     'Label']]

    # Remove any NaN values
    data = data.dropna(axis=0)
    
    
    # Convert to matrix, Break into windows
    data = data.values
    
    conv_data, dense_data, labels = [], [], []
    for i in range(0, data.shape[0], WINDOW_SIZE):
        window = data[i:i+WINDOW_SIZE,]
        
        total = np.any(window == window[0,-1], axis = 1)
        total = np.sum(total)
        
        # Discard windows that are not only of one class
        if total != WINDOW_SIZE:
            continue
        
        accel, phone_active, label = window[:,:3], window[:,3:-1], window[:,-1]

        orientation = get_orientation(accel)
        accel = np.concatenate([accel, orientation], axis=1)
        
        conv_data.append(accel)
        
        raw_accel = accel[:,:3].astype(np.float64)

        raw_accel_x = raw_accel[:,0]
        raw_accel_y = raw_accel[:,1]
        raw_accel_z = raw_accel[:,2]
        raw_accel_abs = np.absolute(raw_accel)

        # is_flat_val = np.absolute(raw_accel_x - raw_accel_y).mean()
        # is_flat = 1.0 if is_flat_val < 0.3 and np.absolute(9.8 - np.absolute(raw_accel_z.mean())) < 0.5 else 0.0
        # is_flat = lambda row : 1.0 if row[9] < 0.3 and row[10] < 0.3 and abs(row[11] - 9.8) < 1.5 else 0.0
        dense_features = [
            phone_active.sum(axis=0), # phone active data
            raw_accel.mean(axis=0), # mean X, Y, Z acceleration
            raw_accel.std(axis=0), # std X, Y, Z acceleration
            raw_accel_abs.mean(axis=0), # mean X, Y, Z magnitude
            raw_accel_abs.std(axis=0), # std of X, Y, Z magnitude
            # np.array([is_flat_val, is_flat])

        ]
        dense_data.append(np.concatenate(dense_features, axis=0))
        
        labels.append(label[0])
        
    conv_data = np.array(conv_data)
    dense_data = np.array(dense_data)
    labels = np.array(labels)
    
    np.savez(npz_name, conv_data=conv_data, dense_data=dense_data, labels=labels)
    
    return conv_data, dense_data, labels

"""
Source: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
"""
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# Gets angle between v and the x, y, z axes
def get_orientation(accel_window):
    x_axis, y_axis, z_axis = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    angles = [np.apply_along_axis(lambda row: angle_between(row[:3], ax), 1, accel_window) for ax in [x_axis, y_axis, z_axis]]
    # print("Angles:", [a.shape for a in angles])
    return np.stack(angles, axis=1)


def generate_features(args):
    # Compile diary study data
    if args.get_raw_data:
        d, cols = get_data(TRAINING_DATA, calibrate_accel=args.calibrate_accel, no_boundary=args.no_boundary)
    else:
        print("Loading from saved data")
        data = np.load(RAW_NPZ_NAME)
        d, cols = data['all_data'], list(data['data_cols'])
        d = pd.DataFrame(data=d, columns=cols)

    # Convert data to features and save to disk
    npz = NPZ_NO_BOUNDARY_NAME if args.no_boundary else NPZ_NAME
    accel_data, phone_active_data, labels = get_features_and_labels(d, cols, npz_name=npz)

def train_net(args):
    train_model_kfold_multiclass(k=args.k, epochs=args.epochs, 
        check_misclf=args.check_misclf, save_path=args.save_path, model_path=args.model_path,
        evaluate_test=args.test, results_dir=args.results_dir, even_class_folds=args.even_class_folds)


def parse_args():
  """Parse arguments."""
  parser = argparse.ArgumentParser(add_help=True, description=__doc__)

  parser.add_argument(
    '-e', metavar='epochs', dest='epochs', type=int,
    default=10,  
    help="Number of training epochs.",
  )

  parser.add_argument(
    '-k', metavar='k', dest='k', type=int,
    default=10,  
    help="Number of folds for k-folds cross validation",
  )

  parser.add_argument(
    '-w', metavar='window', dest='window', type=int,
    default=50,  
    help="Number of timesteps (10 ms) per window",
  )

  parser.add_argument(
    '--get_raw_data', action='store_true',
    help='whether to scrape raw data')

  parser.add_argument(
    '--gen_features', action='store_true',
    help='whether to generate features')

  parser.add_argument(
    '--check_misclf', action='store_true',
    help='whether to print confusion matrix of misclassified')

  parser.add_argument(
    '--calibrate_accel', action='store_true',
    help='whether to calibrate acceleration data')

  parser.add_argument(
    '--test', action='store_true',
    help='whether to test on TEST_DATA after training')

  parser.add_argument(
    '--even_class_folds', action='store_true',
    help='whether to have equal amounts of each class in each fold'
    )

  parser.add_argument(
    '--no_boundary', action='store_true',
    help='whether to remove data on diary boundaries')

  parser.add_argument(
    '-s', metavar='save_path', dest='save_path', type=str,
    help='path to save model in'
    )

  parser.add_argument(
    '-m', metavar='model_path', dest='model_path', type=str, default=None,
    help='path to load model from'
    )


  parser.add_argument(
    '-d', metavar='results_dir', dest='results_dir', type=str,
    default='../multiclass_net_results',  
    help="path to directory to save results in",
  )


  args = parser.parse_args()

  global WINDOW_SIZE
  WINDOW_SIZE = args.window

  if args.gen_features:
    generate_features(args)
  
  # if args.test:
  #   d, cols = get_data(TEST_DATA, calibrate_accel=args.calibrate_accel, no_boundary=True, test_npz=RAW_TEST_NPZ_NAME)
  #   get_features_and_labels(d, cols, npz_name=TEST_NPZ_NAME)
    # generate_features(args)


  train_net(args)

  
if __name__ == '__main__':
  parse_args()
