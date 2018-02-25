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

def create_training_data(args):
	# if args.use_config:
	# 	import BackpackConfig

	# 	POSITIVE_DATA = BackpackConfig.POSITIVE_DATA
	# 	NEGATIVE_DATA = BackpackConfig.NEGATIVE_DATA

	# 	FEATURES_FILE = BackpackConfig.FEATURES_FILE
	# 	WINDOW_SIZE = BackpackConfig.WINDOW_SIZE
	# 	MAX_TRIM = BackpackConfig.MAX_TRIM


	full_feature_vector = []
	all_labels = []

	for folder in [POSITIVE_DATA, NEGATIVE_DATA]:

		for filename in os.listdir(folder):

			if not filename.endswith(".csv"):
				continue
			with open(folder + "/" + filename, 'r') as datafile:
				print(filename)

				all_data = list(csv.reader(datafile))

				if os.path.isfile(folder + "/" + filename[:-4] + "_trim.txt"):
					fp = open(folder + "/" + filename[:-4] + "_trim.txt")
					trim = int(fp.readline())

					all_data = all_data[:trim]
			
			for i in range(min(MAX_TRIM, len(all_data)) - WINDOW_SIZE + 1):

				feature_vector = []
				for j in range(WINDOW_SIZE):
					if len(all_data[i+j]) < 4:
						break

					depth_vector = []
					depth_vector.append(float(all_data[i + j][1]))
					depth_vector.append(float(all_data[i + j][2]))
					depth_vector.append(float(all_data[i + j][3]))

					feature_vector.append(depth_vector)

				full_feature_vector.append(feature_vector)

				if folder == POSITIVE_DATA:
					all_labels.append(1)
				else:
					all_labels.append(0)
	#Save to .npz
	np.savez(FEATURES_FILE, full_feature_vector = full_feature_vector, all_labels = all_labels)

def create_training_data_diary():
	
	import process_diary_study_all_data

	
	data_matrix, data_col = process_diary_study_all_data.getAllUserData()
	pos_data_matrix, pos_data_col = process_diary_study_all_data.getAllPosData()
	neg_data_matrix, neg_data_col = process_diary_study_all_data.getAllNegData()


	full_feature_vector = []
	all_labels = []

	for i in range(0, len(data_matrix), WINDOW_SIZE):
		working_window = data_matrix[i:i+WINDOW_SIZE]
		
		total = np.any(working_window == 'Hand Classifier', axis = 1)
		total=np.sum(total)

		#If this window contains both positive and negative data, then discard this data
		if total != 0 and total != WINDOW_SIZE:
			continue

		# feature_window, code = obtain_derivative_features(working_window, WINDOW_SIZE)
		feature_window, code = obtain_regular_features(working_window, WINDOW_SIZE)
		# print(feature_window.shape)
		if code == -1:
			continue

		full_feature_vector.append(feature_window)

		if total == 0:
			all_labels.append(0)
		else:
			all_labels.append(1)


	for i in range(0, len(pos_data_matrix), WINDOW_SIZE):
		working_window = pos_data_matrix[i:i+WINDOW_SIZE]
		
		total = np.any(working_window == 'Hand Classifier', axis = 1)
		total=np.sum(total)

		#If this window contains both positive and negative data, then discard this data
		if total != 0 and total != WINDOW_SIZE:
			continue

		# feature_window, code = obtain_derivative_features(working_window, WINDOW_SIZE)
		feature_window, code = obtain_regular_features(working_window, WINDOW_SIZE)
		# print(feature_window.shape)
		if code == -1:
			continue

		full_feature_vector.append(feature_window)


		if total == 0:
			all_labels.append(0)
		else:
			all_labels.append(1)

	# for i in range(0, len(neg_data_matrix), WINDOW_SIZE):
	# 	working_window = neg_data_matrix[i:i+WINDOW_SIZE]
		
	# 	total = np.any(working_window == 'Hand Classifier', axis = 1)
	# 	total=np.sum(total)

	# 	#If this window contains both positive and negative data, then discard this data
	# 	if total != 0 and total != WINDOW_SIZE:
	# 		continue

	# 	# print(working_window)

	# 	# feature_window, code = obtain_derivative_features(working_window, WINDOW_SIZE)
	# 	feature_window, code = obtain_regular_features(working_window, WINDOW_SIZE)
	# 	# print(feature_window.shape)
	# 	if code == -1:
	# 		continue

	# 	full_feature_vector.append(feature_window)

	# 	if total == 0:
	# 		all_labels.append(0)
	# 	else:
	# 		all_labels.append(1)

	

	full_feature_vector = np.array(full_feature_vector)
	all_labels = np.array(all_labels)


	#Save to .npz
	np.savez("DIARY_STUDY_NonDerivative_Synthetic.npz", full_feature_vector = full_feature_vector, all_labels = all_labels)

def create_training_data_diary_multiclass():
	
	import process_diary_study

	

	full_csv = process_diary_study.obtain_data('../../DiaryStudies/2017_11_15/',
		DIARY_FILE='../../DiaryStudies/2017_11_15/diary_state_no_steady.txt')

	full_feature_vector = []
	all_labels = []

	for i in range(0, len(full_csv), WINDOW_SIZE):
		working_window = full_csv[i:i+WINDOW_SIZE]
		
		total = np.any(working_window == working_window[0,-1], axis = 1)
		total= np.sum(total)

		#If this window contains both positive and negative data, then discard this data
		if total != WINDOW_SIZE:
			continue

		feature_window, code = obtain_derivative_features(working_window, WINDOW_SIZE)
		# feature_window, code = obtain_regular_features(working_window, WINDOW_SIZE)
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

		#Sanity check

		feature_array.append(temp)

	feature_array = np.array(feature_array)
	return feature_array, 1


def create_validation_data(isPositive=True):
	data = []

	full_feature_vector = []

	for filename in os.listdir(VALIDATION_FOLDER):
		print(filename)
		if not filename.endswith(".csv"):
			continue
		with open(VALIDATION_FOLDER + "/" + filename, 'r') as datafile:
			all_data = list(csv.reader(datafile))	
		
		for i in range(len(all_data) - WINDOW_SIZE + 1):
			feature_vector = []
			for j in range(WINDOW_SIZE):
				if len(all_data[i+j]) < 4:
					break
				feature_vector.append(float(all_data[i + j][1]))
				feature_vector.append(float(all_data[i + j][2]))
				feature_vector.append(float(all_data[i + j][3]))

			if len(feature_vector) == WINDOW_SIZE * 3:
				full_feature_vector.append(feature_vector)

	data = full_feature_vector

	if isPositive:
		labels = [1] * len(data)
	else:
		labels = [0] * len(data)

	#Save to .npz
	# vals_to_save = {temp_features[0]:"positive", temp_features[1]:"negative"}
	np.savez(VALIDATION_FILE, labels=labels, data=data)


def train_model():

	# if args.use_config:
	# 	import BackpackConfig

	# 	FEATURES_FILE = BackpackConfig.FEATURES_FILE
	# 	CLASSIFIER_NAME = BackpackConfig.CLASSIFIER_NAME

	# 	WINDOW_SIZE = BackpackConfig.WINDOW_SIZE


	#Read from .npz
	# data = np.load(FEATURES_FILE)
	data = np.load("DIARY_STUDY_NonDerivative_nosteady.npz")
	features = data['full_feature_vector']
	labels = data['all_labels']

	#Shuffle_Data
	# Generate the permutation index array.
	permutation = np.random.permutation(features.shape[0])
	# Shuffle the arrays by giving the permutation in the square brackets.
	shuffled_features = features[permutation]
	shuffled_labels = labels[permutation]

	# shuffled_features = features
	# shuffled_labels = labels
	# print(features.shape)

	reshaped_features = np.array([f.transpose() for f in shuffled_features])


	shuffled_accel = np.array([np.array([f[0], f[1], f[2], f[3], f[4], f[5]]).transpose() for f in reshaped_features])
	shuffled_other = np.array([np.array([f[6], f[7], f[8]]).transpose() for f in reshaped_features])

	shuffled_other_long = np.array([ np.reshape(f, (1, WINDOW_SIZE * 3)) for f in shuffled_other])
	print(shuffled_labels.shape)

	repeat_scores = []

	for _ in range(3):

		accel_input = keras.layers.Input(shape=(WINDOW_SIZE, 6))
		first = Conv1D(64, 6,  activation='relu')(accel_input)
		first = Conv1D(64, 6,  activation='relu')(first)
		print(first.shape)
		pool_first = MaxPooling1D(6)(first)
		print(pool_first.shape)
		first = Conv1D(128, 6, activation='relu')(pool_first)
		print(first.shape)
		# first = Conv1D(128, 6, activation='relu')(first)
		# g_avg_first = GlobalAveragePooling1D()(first)
		# print(g_avg_first.shape)
		d_first = Dropout(0.5)(first)

		# first_output = Dense(1, activation='sigmoid')(d_first)
		print(d_first.shape)
		other_input = keras.layers.Input(shape=(1, WINDOW_SIZE* 3))

		x = keras.layers.concatenate([d_first, other_input])
		print(x.shape)
		fully_output = Dense(64, activation='relu')(x)
		output_without_flatten = Dense(1, activation='sigmoid')(fully_output)
		output = Flatten()(output_without_flatten)
		print(output.shape)

		model = Model(inputs=[accel_input, other_input], outputs=[output])

		#compile
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		model.fit([shuffled_accel, shuffled_other_long], [shuffled_labels], epochs=10, batch_size=10, validation_split = 0.2)
		#Save model


	# 	#Evaluate model
	# 	scores = model.evaluate(data_test, labels_test)
	# 	print(scores)
	# 	repeat_scores.append(scores[1])
	# 	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	# print(np.mean(repeat_scores), np.std(repeat_scores))
	if args.save_model:
		model.save(CLASSIFIER_NAME)


def evaluate_classifier():
	model = load_model(CLASSIFIER_NAME)

	#Load Data
	#Read from .npz
	features = np.load(VALIDATION_FILE)

	print(features.keys())

	data = features['data']
	labels= features['labels']

	scores = model.evaluate(data, labels)
	# repeat_scores.append(scores[1])
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



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
  	create_training_data(args)
  else:
  	train_model_kfold_multiclass(args)

  

if __name__ == '__main__':
  # parse_args()
  # create_training_data_diary()
  train_model()

