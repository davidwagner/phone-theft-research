import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D

import numpy as np
import csv
import os
import argparse

POSITIVE_DATA = "../../Backpack_PostData"
NEGATIVE_DATA = "../../Backpack_NegData"

VALIDATION_FOLDER = "../../validation_backpack"

FEATURES_FILE = "HandFeatures_50.npz"
VALIDATION_FILE = "hand_validation.npz"

CLASSIFIER_NAME = "HandClassifierNN.h5"

WINDOW_SIZE = 50
MAX_TRIM = 2500


def create_training_data_diary_multiclass():
	
	#Import Steven's script
	import process_diary_study


	full_csv = process_diary_study.obtain_data('../../DiaryStudies/2017_11_15/',
		DIARY_FILE='../../DiaryStudies/2017_11_15/diary_state_no_steady.txt')

	full_feature_vector = []
	all_labels = []

	for i in range(0, len(full_csv), WINDOW_SIZE):
		working_window = full_csv[i:i+WINDOW_SIZE]
		
		#Makes sure that all the labels in a single window is the same
		total = np.any(working_window == working_window[0,-1], axis = 1)
		total = np.sum(total)

		#If this window contains both positive and negative data, then discard this data
		if total != WINDOW_SIZE:
			continue


		#Pick one as needed
		feature_window, code = obtain_derivative_features(working_window, WINDOW_SIZE)
		# feature_window, code = obtain_regular_features(working_window, WINDOW_SIZE)

		#If there is some sort of error, discard this window
		if code == -1:
			continue


		full_feature_vector.append(feature_window)

		if working_window[0,-1] == "Backpack Classifier":
			all_labels.append(0)
		elif working_window[0,-1] == "Pocket Classifier":
			all_labels.append(1)
		elif working_window[0,-1] == "Bag Classifier":
			all_labels.append(2)
		elif working_window[0,-1] == "Hand Classifier":
			all_labels.append(3)
		elif working_window[0,-1] == "Steady State Classifier":
			all_labels.append(4)
		elif working_window[0,-1] == "Table Classifier":
			all_labels.append(5)
		else:
			raise Exception("No class found")	

	full_feature_vector = np.array(full_feature_vector)
	all_labels = np.array(all_labels)


	#Save to .npz
	np.savez("DIARY_STUDY_Derivative_NoSteady_multi.npz", full_feature_vector = full_feature_vector, all_labels = all_labels)

def obtain_derivative_features(window, window_size):
	feature_array = []

	for i in range(1, window_size):
		temp = []
		x_feature = float(window[i][1] - window[i-1][1])
		y_feature = float(window[i][2] - window[i-1][2])
		z_feature = float(window[i][3] - window[i-1][3])

		if np.isnan(x_feature) or np.isnan(y_feature) or np.isnan(z_feature):
			return np.zeros(2), -1

		temp.append(x_feature)
		temp.append(y_feature)
		temp.append(z_feature)

		feature_array.append(temp)

	feature_array = np.array(feature_array)
	return feature_array, 1


def obtain_regular_features(window, window_size):
	feature_array = []
	length = min(window_size, len(window))
	for i in range(0, length):
		temp = []

		if np.isnan(float(window[i][1])) or np.isnan(float(window[i][2])) or np.isnan(float(window[i][3])):
			return np.zeros(2), -1

		temp.append(float(window[i][1]))
		temp.append(float(window[i][2]))
		temp.append(float(window[i][3]))
		temp.append(float(window[i][9]))
		temp.append(float(window[i][10]))
		temp.append(float(window[i][11]))

		temp.append(float(window[i][6]))
		temp.append(float(window[i][7]))
		temp.append(float(window[i][8]))

		feature_array.append(temp)

	feature_array = np.array(feature_array)
	return feature_array, 1



def train_model_kfold_multiclass(args, k=10):

	#Read from .npz
	data = np.load("DIARY_STUDY_Derivative_NoSteady_multi.npz")
	features = data['full_feature_vector']
	labels = data['all_labels']


	#Converts labels to one-hot vector
	labels = keras.utils.to_categorical(labels, num_classes = 6)



	#Copied from Joanna (Despite the name, shuffled_array is not actually shuffled and same for others)
	shuffled_accel = np.array([np.array([f[0], f[1], f[2], f[3], f[4], f[5]]).transpose() for f in features])
	shuffled_other = np.array([np.array([f[6], f[7], f[8]]).transpose() for f in features])

	shuffled_other_long = np.array([ np.reshape(f, (1, WINDOW_SIZE * 3)) for f in shuffled_other])


	# create model
	accel_input = keras.layers.Input(shape=(WINDOW_SIZE, 6))
	first = Conv1D(64, 6,  activation='relu')(accel_input)
	first = Conv1D(64, 6,  activation='relu')(first)
	pool_first = MaxPooling1D(6)(first)
	first = Conv1D(128, 6, activation='relu')(pool_first)
	d_first = Dropout(0.5)(first)

	other_input = keras.layers.Input(shape=(1, WINDOW_SIZE* 3))

	x = keras.layers.concatenate([d_first, other_input])
	fully_output = Dense(64, activation='relu')(x)
	output_without_flatten = Dense(1, activation='sigmoid')(fully_output)
	output = Flatten()(output_without_flatten)
	model = Model(inputs=[accel_input, other_input], outputs=[output])

	#compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.save_weights('model.h5')

	section_length = int(len(labels) / k)


	#Chunk out for different values of k
	for i in range(k):
		training_data = np.delete(features, np.s_[i*section_length:(i+1) * section_length], axis = 0)
		training_labels = np.delete(labels, np.s_[i*section_length:(i+1) * section_length], axis = 0)

		validation_data = features[i*section_length:(i+1) * section_length]
		validation_labels = labels[i*section_length:(i+1) * section_length]
	
		model.load_weights('model.h5')

		print(np.shape(training_labels))

		#TODOI FIX THE INPUTS
		hist = model.fit(training_data, training_labels, epochs=args.epochs, batch_size=10, validation_data = (validation_data, validation_labels))
		average_scores.append(hist.history['val_acc'][-1])
	

	print(average_scores)



def parse_args():
  """Parse arguments."""
  parser = argparse.ArgumentParser(add_help=True, description=__doc__)

  parser.add_argument(
	'-epochs', metavar='epochs', dest='epochs', type=int,
	default=10,  
	help="Number of training epochs.",
  )

  parser.add_argument(
	'-repeat', metavar='repeat', dest='repeat', type=int,
	default=1,  
	help="Number of repeats.",
  )
  parser.add_argument(
	'-method', metavar='method', dest='method', type=float,
	required=True,  
  )

  parser.add_argument(
	'-model_num', metavar='model_num', dest='model_num', type=int,
	required=True
  )

  parser.add_argument(
	'-save_model', metavar='save_model', dest='save_model', type=bool,
	required=True
  )

  parser.add_argument(
	'-use_config', metavar='use_config', dest='use_config', type=bool,
	required=True
  )

  args = parser.parse_args()

  if args.method == 0:
  	create_training_data_diary_multiclass(args)
  else:
  	train_model_kfold_multiclass(args)

  

if __name__ == '__main__':
  parse_args()

