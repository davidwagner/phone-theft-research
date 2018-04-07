How to use multiclass_net.py:

To load the diary study data that you want to use, fill in the TRAINING_DATA list
near the top of multiclass_net.py. For each diary study, you should specify a tuple of:

(path_to_directory_with_diary_study_data, path_to_corresponding_log_file, user_id)
For example: ('../data/diary-study-3-19/diary_data/', '../data/diary-study-3-19/diary_state_3_19.txt', 'd792b61e')

You can then run the net (i.e. train and validate on the diary study data) with:

`python multiclass_net.py -k 20 -e 2 --check_misclf --gen_features --calibrate_accel`

in order to generate the features from the diary study data and then save them to a npz file.

OR if you have already generated the features, then you can just run

`python multiclass_net.py -k 20 -e 2 --check_misclf`

where k = the number of folds and e = the number of epochs.

For each fold, the following will be printed:

<The class distribution of the training and validation sets.>
Training dist. Counter({4: 63155, 0: 30701, 3: 4145, 1: 2011})
Validation dist. Counter({4: 3729, 1: 1534})

<What class each label number corresponds to.>
0 Backpack Classifier
1 Pocket Classifier
2 Bag Classifier
3 Hand Classifier
4 Table Classifier

<Two matrices are printed out, where the row indicates
the label that our NN predicted and the column indicates
the actual label of the data.>

<The proportion of misses by expected label.>
<For example, in the 1-th column, out of all misses
on samples that were actually POCKET, 0.448 were 
incorrectly predicted as BACKPACK, 0.51 incorrectly
predicted as HAND, and 0.042 as TABLE.>
Misses by expected label:
[[ 0.     0.448  0.     0.     0.   ]
 [ 0.     0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.     0.   ]
 [ 0.     0.51   0.     0.     0.   ]
 [ 0.     0.042  0.     0.     0.   ]]

<The total number of misses on this fold.>
Total misses: 1439.0

<The proportion of each type of miss out of all misses.>
<For example, in the 4-th row, 0-th column, 0.171 of the 1439
total misses were incorrectly predicting TABLE when it was actually BACKPACK.>
Misses by total misses
[[ 0.     0.077  0.     0.08   0.214]
 [ 0.179  0.     0.     0.001  0.001]
 [ 0.     0.     0.     0.     0.   ]
 [ 0.201  0.     0.     0.     0.009]
 [ 0.171  0.     0.     0.068  0.   ]]

 At the end, these two matrices are printed out for each class,
 averaging the matrices from each fold where the validation set
 contained at least 500 samples of the class.


 Lastly, if you want to do error analysis and take a look at the misclassified data,
 go to line 160 and modify the dictionary folds_of_note, which maps a FOLD_NUMBER
 to a list of FILTERS.

 For example:
 	9: [(0, 1), (1, 1), (0, 0)]

 means that for the validation data in fold 9, get all the data that we predicted as 0
 but was actually 1 (i.e. the tuple (0, 1)), all the data we predicted as 1 and was actually
 1 (i.e. the tuple (1, 1)), and all the data we predicted as 0 and was actually 0 (i.e. the
 tuple (0, 0)).