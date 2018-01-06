import BaseClassifier

import Sensors as s

import numpy as np
import math

# CLASSIFIER_PATH = './classifier_pickles/TrainedClassifier_PhoneSteadyState/PocketSteadyStateClassifier_Beta.pkl'
# clf = joblib.load(CLASSIFIER_PATH)
THRESHOLD_MAG = 2

WINDOW_SIZE = 100

class Classifier(BaseClassifier.BaseClassifier):

	def classify(self, windows):

		window_vector = []

		for row in windows[s.ACCELEROMETER]:
			xVal = float(row[1])
			yVal = float(row[2])
			zVal = float(row[3])

			mag = math.sqrt(xVal * xVal + yVal * yVal + zVal * zVal)
			window_vector.append(mag)

		if np.ptp(window_vector) < THRESHOLD_MAG:
			return 1

		return 0

	# Need to change to time
	def getWindowTime(self):
		return WINDOW_SIZE

	def getRelevantSensors(self):
		return [s.ACCELEROMETER]

	def getName(self):
		return "Steady State Classifier"




