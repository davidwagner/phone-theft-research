import BaseClassifier
import HandAndPossessionFeaturizer as featurizer
from sklearn.externals import joblib

import Sensors as s

CLASSIFIER_PATH = './classifier_pickles/Hand_Classifier/hand_clf_1_0.pkl'
clf = joblib.load(CLASSIFIER_PATH)
class Classifier(BaseClassifier.BaseClassifier):

	def classify(self, windowOfData):
		accelData = windowOfData[s.ACCELEROMETER]
		activeData = windowOfData[s.PHONE_ACTIVE_SENSORS]

		combinedData = []

		for i in range(len(accelData)):
			row = accelData[i][:4] + activeData[i][1:]
			combinedData.append(row)

		features = featurizer.getFeatures(combinedData)
		results = clf.predict(features)[0]
		# print("Type:", type(results))
		return results
		# return results

	# Need to change to time
	def getWindowTime(self):
		return 50

	def getRelevantSensors(self):
		return [s.ACCELEROMETER, s.PHONE_ACTIVE_SENSORS]

	def getName(self):
		return "Hand Classifier"




