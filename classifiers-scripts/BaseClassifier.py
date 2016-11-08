"""
The easiest way to extend this, would probably be creating a new file, e.g. TableClassifier.py, that imports
the classifier/featurizer code that you've already written, and then having a class in that file 
`class Classifier(BaseClassifier)` that implements the methods below.
Classifiers.py (CLASSIFIERS) and logClassifier.py ( runClassifier() )have the intended usage.
"""

class BaseClassifier():

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
		return 0 

	"""
	Returns the # of milliseconds of data needed to classify a window of data
	e.g. Table classifier needs 30ms of data (3 rows of accelerometer data) for one window
	"""
	def getWindowTime():
		return 1000

	"""
	Returns a list of sensors, whose data is needed for the classifier (see LogConstants.py)
	e.g. Table classifier would return [ACCELEROMETER]
	"""
	def getRelevantSensors():
		return []

	"""
	Returns meaning of classification values (useful for interpreting results in post-processing)
	"""
	def getClassifications():
		return {0 : "Not in given state", 1 : "In given state"}