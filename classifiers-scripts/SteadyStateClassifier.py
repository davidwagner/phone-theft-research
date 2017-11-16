import BaseClassifier

import Sensors as s

import numpy as np

# CLASSIFIER_PATH = './classifier_pickles/TrainedClassifier_PhoneSteadyState/PocketSteadyStateClassifier_Beta.pkl'
# clf = joblib.load(CLASSIFIER_PATH)
THRESHOLD_X = 1
THRESHOLD_Y = 1
THRESHOLD_Z = 1
class Classifier(BaseClassifier.BaseClassifier):

	def classify(self, windowOfData):
		data = windowOfData[s.ACCELEROMETER]

		x_data = [float(i[1]) for i in data]
		y_data = [float(i[1]) for i in data]
		z_data = [float(i[1]) for i in data]

		if np.ptp(x_data) < THRESHOLD_X and np.ptp(y_data) < THRESHOLD_Y and np.ptp(z_data) < THRESHOLD_Z:
			return 1

		return 0

	# Need to change to time
	def getWindowTime(self):
		return 50

	def getRelevantSensors(self):
		return [s.ACCELEROMETER]




