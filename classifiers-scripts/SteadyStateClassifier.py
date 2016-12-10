import BaseClassifier
import SteadyStatePhoneFeaturizer as featurizer
from sklearn.externals import joblib

import Sensors as s

CLASSIFIER_PATH = './classifier_pickles/TrainedClassifier_PhoneSteadyState/PocketSteadyStateClassifier_Beta.pkl'
clf = joblib.load(CLASSIFIER_PATH)
class Classifier(BaseClassifier.BaseClassifier):

	def classify(self, windowOfData):
		features = featurizer.dataToFeatures(windowOfData)
		results = clf.predict(features)

		return int(results[0])
		# return results

	# Need to change to time
	def getWindowTime(self):
		return 10

	def getRelevantSensors(self):
		return [s.ACCELEROMETER]




