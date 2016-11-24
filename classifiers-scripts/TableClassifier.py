import BaseClassifier
import TableFeaturizer as featurizer
from sklearn.externals import joblib

import Sensors as s

class Classifier(BaseClassifier.BaseClassifier):

	def classify(self, windowOfData):
		features = featurizer.dataToFeatures(windowOfData)
		results = featurizer.checkFeatures(features)

		return int(results[0])
		# return results

	# Need to change to time
	def getWindowTime(self):
		return 3

	def getRelevantSensors(self):
		return [s.ACCELEROMETER]




