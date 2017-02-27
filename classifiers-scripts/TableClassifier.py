import BaseClassifier

import Sensors as s
import numpy

windowSize = 5

class Classifier(BaseClassifier.BaseClassifier):

	"""Input: dictionary"""
	"""Return value: 0 if window is deemed negative. 1 if window is positive"""
	def classify(self, windows):

		"""Edit as Necessary"""
		thresholdX = 0.8
		thresholdY = 0.2
		thresholdZ = 0.65

		if s.ACCELEROMETER not in windows:
			raise Exception("Accelerometer not found")


		if len(windows[s.ACCELEROMETER]) != windowSize:
			raise Exception("Window Size is incorrect")

		

		"""Take average of the entire window"""
		xValues = []
		yValues = []
		zValues = []

		for row in windows[s.ACCELEROMETER]:
			xValues.append(float(row[1]))
			yValues.append(float(row[2]))
			zValues.append(float(row[3]))

		xVal = numpy.mean(xValues)
		yVal = numpy.mean(yValues)
		zVal = numpy.mean(zValues)


		if (abs(xVal-0) < thresholdX and abs(yVal-0) < thresholdY and abs(zVal-9) < thresholdZ):
			return 1

		return 0

	# Need to change to time
	def getWindowTime(self):
		return windowSize

	def getRelevantSensors(self):
		return [s.ACCELEROMETER]

	def getName(self):
		return "Table Classifier"




