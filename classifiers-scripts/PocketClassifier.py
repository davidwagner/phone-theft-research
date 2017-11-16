import BaseClassifier
import phone_pocket_nn as featurizer
import Sensors as s

from keras.models import load_model

import numpy as np


CLASSIFIER_PATH = './classifier_pickles/PocketClassifier/PocketNN.h5'
clf = load_model(CLASSIFIER_PATH)

class Classifier(BaseClassifier.BaseClassifier):

	def classify(self, windowOfData):
		accelData = windowOfData[s.ACCELEROMETER]
		
		features = []

		for i in range(len(accelData)):
			features.append(float(accelData[i][1]))
			features.append(float(accelData[i][2]))
			features.append(float(accelData[i][3]))

		features = np.expand_dims(features, axis=0)
		features = np.expand_dims(features, axis=2)
		print(np.shape(features))
		results = clf.predict(features)
		
		if results[0] > 0.5:
			return 1
		else:
			return 0

	# Need to change to time
	def getWindowTime(self):
		return featurizer.WINDOW_SIZE

	def getRelevantSensors(self):
		return [s.ACCELEROMETER]

	def getName(self):
		return "Pocket Classifier"




