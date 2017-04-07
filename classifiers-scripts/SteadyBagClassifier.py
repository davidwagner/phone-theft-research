import BaseClassifier

import Sensors as s
import numpy

windowSize = 50

class Classifier(BaseClassifier.BaseClassifier):

	"""Input: dictionary"""
	"""Return value: 0 if window is deemed negative. 1 if window is positive"""
	def classify(self, windows):

		"""Edit as Necessary"""
		thresholdX = 0.3
		thresholdY = 0.3
		thresholdZ = 0.3

		if s.ACCELEROMETER not in windows:
			raise Exception("Accelerometer not found")


		if len(windows[s.ACCELEROMETER]) != windowSize:
			raise Exception("Window Size is incorrect")

		for i in windows[s.LIGHT_SENSOR]:
			if float(i[1]) > 25:
				return 0

		"""Take average of the entire window"""
		xValues = []
		yValues = []
		zValues = []

		for row in windows[s.ACCELEROMETER]:
			xValues.append(float(row[1]))
			yValues.append(float(row[2]))
			zValues.append(float(row[3]))


		if (max(xValues) - min(xValues) < thresholdX and max(yValues) - min(yValues) < thresholdY and max(zValues) - min(zValues) < thresholdZ):
			return 1

		return 0

	# Need to change to time
	def getWindowTime(self):
		return windowSize

	def getRelevantSensors(self):
		return [s.ACCELEROMETER, s.LIGHT_SENSOR]

	def getName(self):
		return "Steady State Bag Classifier"




