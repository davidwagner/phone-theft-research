
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from collections import Counter

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

NPZ_NAME = "DIARY_STUDY_NoDerivative_NoSteady_multi.npz"
CHECK_MISCLF_LOG = "multiclass_results_log.pkl"

DIARY_STUDIES = [
#   (dir_with_diary_data, diary_log_file_corresponding_to_data, user_id_of_data)
    ('../data/diary-study-11-14-15/diary_data/', '../data/diary-study-11-14-15/diary_state_no_steady.txt', 'd792b61e'),
    ('../data/diary-study-2-20/diary_data/', '../data/diary-study-2-20/diary_state_2_20.txt', 'd792b61e'),
    ('../data/diary-study-3-13/diary_data/', '../data/diary-study-3-13/diary_state_3_13.txt', 'd792b61e')
]


def train_model_kfold_multiclass(k=20, epochs=10, check_misclf=False):

    #Read from .npz
    data = np.load(NPZ_NAME)
    accel_data = data['accel_data']
    phone_active_data = data['phone_active_data']
    labels = data['labels']


    #Converts labels to one-hot vector
    labels = keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)


    # create model
    accel_input = keras.layers.Input(shape=(WINDOW_SIZE, 6), name='accel_input')
    first = Conv1D(64, 3,  activation='relu', use_bias=True, name="first_1D")(accel_input)
    first = Conv1D(64, 3,  activation='relu', use_bias=True, name="second_1D")(first)
    pool_first = MaxPooling1D(3,name="first_max_pool")(first)
    third = Conv1D(128, 3, activation='relu', name="third_1D")(pool_first)
    fourth = Conv1D(128, 3, activation='relu', name="fourth_1D")(third)
    global_pool = GlobalAveragePooling1D(name="global_pool")(fourth)
    d_first = Dropout(0.5, name="dropout")(global_pool)

    other_input = keras.layers.Input(shape=(3,), name='phone_active_input')

    x = keras.layers.concatenate([d_first, other_input], name='concatenate', axis=-1)
    fully_output = Dense(64, activation='relu', name="fully_connected")(x)
    output = Dense(NUM_CLASSES, activation='softmax', name='output')(fully_output)
    # output = Flatten(name='flatten')(output_without_flatten)
    model = Model(inputs=[accel_input, other_input], outputs=[output])

    #compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    if k == 0:
        hist = model.fit([accel_data, phone_active_data], [labels], epochs=epochs, validation_split=0.9)
        print(hist.history['val_acc'][-1])
        
    else: 
        accuracies = []
        data_distributions = [] # (training, validation)
        indices = []
        
        fold_size = accel_data.shape[0] // k
        n = accel_data.shape[0]
        
        # perm = np.random.permutation(n)
        # accel_data = accel_data[perm]
        # phone_active_data = phone_active_data[perm]
        # labels = labels[perm]
        
        miss_matrices = []
        for fold in range(0, n // fold_size * fold_size, fold_size):
            print("FOLD:", fold // fold_size)
            start, end = fold, fold + fold_size if fold // fold_size < n - 1 else n
            mask = np.arange(n)

            val_mask = (mask >= start) & (mask < end)
            train_mask = (mask < start) | (mask >= end)

            training_data = (
                            {'accel_input' : accel_data[train_mask], 
                            'phone_active_input' : phone_active_data[train_mask]},
                            {'output' : labels[train_mask]}
            )

            validation_data = (
                {'accel_input': accel_data[val_mask], 
                 'phone_active_input' : phone_active_data[val_mask]},
                {'output' : labels[val_mask]}

            )
            hist = model.fit(training_data[0], training_data[1], epochs=epochs, validation_data=validation_data)
            accuracies.append(hist.history['val_acc'][-1])
            print(hist.history['val_acc'][-1])
            
            if check_misclf:
                train_dist = Counter(np.argmax(labels[train_mask], axis=1).flatten())
                val_dist = Counter(np.argmax(labels[val_mask], axis=1).flatten())
                print("Training dist.", train_dist)
                print("Validation dist.", val_dist)

                data_distributions.append((train_dist, val_dist))
                indices.append((start, end))

                miss_matrix = predict_and_check(model, [accel_data[val_mask], phone_active_data[val_mask]], labels[val_mask])
                miss_matrices.append(miss_matrix)

        
        print(np.mean(np.array(miss_matrices), axis=0).round(3))

        print("Accuracies", accuracies)

        print("CROSS VAL. ACCURACY:", sum(accuracies) / len(accuracies))

        if check_misclf:
            f = open(CHECK_MISCLF_LOG, 'wb+')
            results = list(zip(accuracies, indices, data_distributions, miss_matrices))
            results = sorted(results, key=lambda x: x[0])

            pickle.dump(results, f)




def predict_and_check(model, data, labels):
    predictions = model.predict(data)
    
    acc_by_class = Counter()
    miss_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    idx = np.arange(NUM_CLASSES)
    for pred, label in zip(predictions, labels):
        pred = int(np.argmax(pred))
        label = int(np.argmax(label))
        if pred != label:
            miss_matrix[pred][label] += 1
            
            acc_by_class[label] += 1
    
    num_misses = np.sum(miss_matrix)
    if num_misses > 0:
        miss_matrix /= np.sum(miss_matrix)
    for i, lbl in enumerate(CLASSES):
        print(i, lbl)

    print("Total misses:", num_misses)

    print(miss_matrix.round(3))
    for label, row in zip(CLASSES, miss_matrix.round(3)):
        print(label, "\t" * 2, row)
    
    return miss_matrix


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


def get_data(dir_diary_user_trips, calibrate_accel=False):
    all_data = []
    data_cols = []
    
    for data_dir, diary_file, user_id in dir_diary_user_trips:
        d, cols = process_diary_study_all_data.getAllUserData(data_dir, diary_file, user_id, as_matrix=False, calibrate_accel=calibrate_accel)
        
        all_data.append(d)
        data_cols = cols
        
    all_data = pd.concat(all_data)
    
    return all_data, data_cols


def get_features_and_labels(d, cols):
    
    # Replace labels with ints
    d['Label'] = 0

    
    for i, label in enumerate(CLASSES):
        d.loc[d["Class"].str.strip() == label, "Label"] = i

    print("Label count:", Counter(d["Label"]))
    print("Class count:", Counter(d["Class"].str.strip()))

    # Use only the relevant columns
    data = d[['X accel.', 'Y accel.', 'Z accel.',
     'X direction changed', 'Y direction changed', 'Z direction changed',
     'Num. Touches', 'Screen State', 'isUnlocked',
     'Label']]

    # Remove any NaN values
    data = data.dropna(axis=0)
    
    
    # Convert to matrix, Break into windows
    data = data.values
    
    accel_data, phone_active_data, labels = [], [], []
    for i in range(0, data.shape[0], WINDOW_SIZE):
        window = data[i:i+WINDOW_SIZE,]
        
        total = np.any(window == window[0,-1], axis = 1)
        total = np.sum(total)
        
        # Discard windows that are not only of one class
        if total != WINDOW_SIZE:
            continue
        
        accel, phone_active, label = window[:,:6], window[:,6:-1], window[:,-1]
        
        accel_data.append(accel)
        
        phone_active_data.append(phone_active.sum(axis=0))
        
        labels.append(label[0])
        
    accel_data = np.array(accel_data)
    phone_active_data = np.array(phone_active_data)
    labels = np.array(labels)
    
    np.savez(NPZ_NAME, accel_data=accel_data, phone_active_data=phone_active_data, labels=labels)
    
    return accel_data, phone_active_data, labels

def generate_features(args):
    # Compile diary study data
    d, cols = get_data(DIARY_STUDIES, calibrate_accel=args.calibrate_accel)

    # Convert data to features and save to disk
    accel_data, phone_active_data, labels = get_features_and_labels(d, cols)

def train_net(args):
    train_model_kfold_multiclass(k=args.k, epochs=args.epochs, check_misclf=args.check_misclf)


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
    '--gen_features', action='store_true',
    help='whether to generate features')

  parser.add_argument(
    '--check_misclf', action='store_true',
    help='whether to print confusion matrix of misclassified')

  parser.add_argument(
    '--calibrate_accel', action='store_true',
    help='whether to calibrate acceleration data')


  args = parser.parse_args()

  global WINDOW_SIZE
  WINDOW_SIZE = args.window

  if args.gen_features:
    generate_features(args)

  train_net(args)

  
if __name__ == '__main__':
  parse_args()
