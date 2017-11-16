import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

from sklearn.model_selection import train_test_split

import numpy as np
import csv
import os
import argparse

POSITIVE_DATA = "PostData"
NEGATIVE_DATA = "NegData"

VALIDATION_FOLDER = "validation_pocket"

FEATURES_FILE = "features.npz"
VALIDATION_FILE = "pocket_validation.npz"

CLASSIFIER_NAME = "PocketNN.h5"

WINDOW_SIZE = 50

def create_training_data():

	positive_features = []
	negative_features = []

	for folder in [POSITIVE_DATA, NEGATIVE_DATA]:

		full_feature_vector = []

		for filename in os.listdir(folder):
			print(filename)
			if not filename.endswith(".csv"):
				continue
			with open(folder + "/" + filename, 'r') as datafile:
				all_data = list(csv.reader(datafile))

			
			
			for i in range(len(all_data) - WINDOW_SIZE + 1):
				feature_vector = []
				for j in range(WINDOW_SIZE):
					feature_vector.append(float(all_data[i + j][1]))
					feature_vector.append(float(all_data[i + j][2]))
					feature_vector.append(float(all_data[i + j][3]))

				assert len(feature_vector) == WINDOW_SIZE * 3

				full_feature_vector.append(feature_vector)

		if folder == POSITIVE_DATA:
			positive_features = full_feature_vector
		else:
			negative_features = full_feature_vector



	#Save to .npz
	# vals_to_save = {temp_features[0]:"positive", temp_features[1]:"negative"}
	np.savez(FEATURES_FILE, positive_features = positive_features, negative_features = negative_features)

def create_validation_data(isPositive=False):
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
				feature_vector.append(float(all_data[i + j][1]))
				feature_vector.append(float(all_data[i + j][2]))
				feature_vector.append(float(all_data[i + j][3]))

			assert len(feature_vector) == WINDOW_SIZE * 3

			full_feature_vector.append(feature_vector)

	data = full_feature_vector

	if isPositive:
		labels = [1] * len(data)
	else:
		labels = [0] * len(data)

	#Save to .npz
	# vals_to_save = {temp_features[0]:"positive", temp_features[1]:"negative"}
	np.savez(VALIDATION_FILE, labels=labels, data=data)

def train_model(args):

	#Read from .npz
	feature_vectors = np.load(FEATURES_FILE)

	positive_features = feature_vectors['positive_features']
	negative_features = feature_vectors['negative_features']

	full_data = np.concatenate((positive_features, negative_features))

	labels = [1] * len(positive_features)
	labels.extend([0] * len(negative_features))

	labels = np.reshape(labels, (len(labels),1))

	full_data, _, labels, __ =  train_test_split(full_data, labels, test_size=0.80)

	#Split data randomly for validation/training
	data_train, data_test, labels_train, labels_test = train_test_split(full_data, labels, \
														test_size=0.20)

	data_train = np.expand_dims(data_train, axis=2)
	data_test = np.expand_dims(data_test,axis=2)

	#Recurrent, 1D Convluitional 

	repeat_scores = []

	for _ in range(args.repeat):

		# create model
		model = Sequential()
		model.add(Conv1D(64, 3, activation='relu', input_shape=(WINDOW_SIZE * 3, 1)))
		model.add(Conv1D(64, 3, activation='relu'))
		model.add(MaxPooling1D(3))
		model.add(Conv1D(128, 3, activation='relu'))
		model.add(Conv1D(128, 3, activation='relu'))
		model.add(GlobalAveragePooling1D())
		model.add(Dropout(0.5))
		model.add(Dense(1, activation='sigmoid'))

		#compile
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		model.fit(data_train, labels_train, epochs=args.epochs, batch_size=10)
		#Save model


		#Evaluate model
		scores = model.evaluate(data_test, labels_test)
		repeat_scores.append(scores[1])
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


	model.save(CLASSIFIER_NAME)
	print(np.mean(repeat_scores), np.std(repeat_scores))

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
	default=3,  
	help="Number of repeats.",
  )
  parser.add_argument(
	'-method', metavar='method', dest='method', type=float,
	required=True,  
	help="Number of training steps.",
  )
  args = parser.parse_args()

  if args.method == 0:
  	create_training_data()
  elif args.method == 1:
  	train_model(args)
  else:
  	# create_validation_data()
  	1+1

  

if __name__ == '__main__':
  	parse_args()
	# create_validation_data()
	# evaluate_classifier()

