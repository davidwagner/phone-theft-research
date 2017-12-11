import BaseClassifier
import backpack_classifier_nn as featurizer
import Sensors as s

from keras.models import load_model

import numpy as np


CLASSIFIER_PATH = './classifier_pickles/BackpackClassifier/BackpackClassifierNN_New.h5'
clf = load_model(CLASSIFIER_PATH)

class Classifier(BaseClassifier.BaseClassifier):

	def classify(self, windowOfData):
		accelData = windowOfData[s.ACCELEROMETER]
		
		features = []

		for i in range(len(accelData)):
			column_vector = []
			column_vector.append(float(accelData[i][1]))
			column_vector.append(float(accelData[i][2]))
			column_vector.append(float(accelData[i][3]))

			features.append(column_vector)

		feature_vector = np.asarray(features)
		feature_vector = np.expand_dims(feature_vector, axis=0)
		results = clf.predict(np.asarray(feature_vector))
		
		return round(results[0][0])

	# Need to change to time
	def getWindowTime(self):
		return featurizer.WINDOW_SIZE

	def getRelevantSensors(self):
		return [s.ACCELEROMETER]

	def getName(self):
		return "Backpack Classifier"




