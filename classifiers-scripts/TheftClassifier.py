import TheftFeaturizer as featurizer
import Classifiers
import Sensors
import BaseClassifier

from sklearn.externals import joblib

class Classifier(BaseClassifier.BasecClassifier):

	"""
	Input: windowOfData is a dictionary with the following format of key-values:
			{SENSOR : [dataRow1, dataRow2, ..., dataRowN]} where 
			SENSOR is a sensor needed for your classifier (found in Sensors.py)
			N, the number of dataRows = getWindowTime() / (ms / row)
			dataRow is a Python list of the corresponding data csv row (e.g. [timestamp, x, y, z])
			*all timestamps will be converted to Python DateTime objects corresponding to actual time*

	Output: a 0 or 1 (the classification)
	"""
	def classify(windowOfData):
		acc_data = windowOfData[Sensors.ACCELEROMETER]
		features = featurizer.acc_featurizer(acc_data)

		filename = 'classifier_weights/random_forest_weights.pkl'
		clf = joblib.load(filename)
		predictions = clf.predict(features)

		return predictions

	"""
	Returns data in a single file.
	"""
	def getWindowTime():
		return Classifiers.DAY_INTERVAL
