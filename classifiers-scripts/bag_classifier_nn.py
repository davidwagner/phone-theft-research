import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split

import numpy as np
import csv
import os
import argparse

POSITIVE_DATA = "Bag_PostData"
NEGATIVE_DATA = "Bag_NegData"

FEATURES_FILE = "BagFeatures.npz"

CLASSIFIER_NAME = "BagClassifierNN.h5"

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




def train_model(args):

	#Read from .npz
	feature_vectors = np.load(FEATURES_FILE)

	positive_features = feature_vectors['positive_features']
	negative_features = feature_vectors['negative_features']

	full_data = np.concatenate((positive_features, negative_features))

	labels = [1] * len(positive_features)
	labels.extend([0] * len(negative_features))

	labels = np.reshape(labels, (len(labels),1))

	#Split data randomly for validation/training
	data_train, data_test, labels_train, labels_test = train_test_split(full_data, labels, \
														test_size=0.20)

	#Recurrent, 1D Convluitional 

	repeat_scores = []

	for _ in range(args.repeat):

		# create model
		model = Sequential()
		model.add(Dense(12, input_dim=WINDOW_SIZE * 3, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))

		#compile
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		model.fit(data_train, labels_train, epochs=args.epochs, batch_size=10)
		#Save model


		#Evaluate model
		scores = model.evaluate(data_test, labels_test)
		print(scores)
		repeat_scores.append(scores[1])
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	print(np.mean(repeat_scores), np.std(repeat_scores))
	model.save(CLASSIFIER_NAME)

def evaluate_model(args):
	data = np.load(args.data)

	model = keras.models.load_model(CLASSIFIER_NAME)



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
  )

  parser.add_argument(
	'-data', metavar='data', dest='data', type=str,

  )

  args = parser.parse_args()

  if args.method == 0:
  	create_training_data()
  else:
  	train_model(args)

  

if __name__ == '__main__':
  parse_args()
	

