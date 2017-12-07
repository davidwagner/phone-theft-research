import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

import numpy as np
import csv
import os
import argparse
import random

POSITIVE_DATA = "../../Pocket_PosData"
NEGATIVE_DATA = "../../Pocket_NegData"

VALIDATION_FOLDER = "../../validation_pocket"

FEATURES_FILE = "pocket_features.npz"
VALIDATION_FILE = "pocket_validation.npz"

CLASSIFIER_NAME = "PocketNN.h5"

WINDOW_SIZE = 150

def create_training_data():

	full_feature_vector = []
	all_labels = []

	for folder in [POSITIVE_DATA, NEGATIVE_DATA]:

		for filename in os.listdir(folder):

			if not filename.endswith(".csv"):
				continue
			with open(folder + "/" + filename, 'r') as datafile:
				print(filename)
				all_data = list(csv.reader(datafile))	
			
			for i in range(len(all_data) - WINDOW_SIZE + 1):

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
	data = np.load(FEATURES_FILE)
	print("Loaded")

	features = data['full_feature_vector']
	labels = data['all_labels']

	repeat_scores = []

	#Shuffle_Data
	# Generate the permutation index array.
	permutation = np.random.permutation(a.shape[0])
	# Shuffle the arrays by giving the permutation in the square brackets.
	shuffled_features = features[permutation]
	shuffled_labels = labels[permutation]

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

		model.fit(shuffled_features, shuffled_labels, epochs=args.epochs, batch_size=10, validation_split=0.2)
		#Save model


		# #Evaluate model
		# scores = model.evaluate(data_test, labels_test)
		# repeat_scores.append(scores[1])
		# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


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
	default=1,  
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

