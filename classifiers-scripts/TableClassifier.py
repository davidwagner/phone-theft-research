import BaseClassifier

import Sensors as s
import numpy as np
import math

windowSize = 50

class Classifier(BaseClassifier.BaseClassifier):

	"""Input: dictionary"""
	"""Return value: 0 if window is deemed negative. 1 if window is positive"""
	def classify(self, windows):

		"""Edit as Necessary"""
		thresholdMag = 0.68

		if s.ACCELEROMETER not in windows:
			raise Exception("Accelerometer not found")


		if len(windows[s.ACCELEROMETER]) != windowSize:
			raise Exception("Window Size is incorrect")

		

		"""Take average of the entire window"""
		window_vector = []

		for row in windows[s.ACCELEROMETER]:
			xVal = float(row[1])
			yVal = float(row[2])
			zVal = float(row[3])

			mag = math.sqrt(xVal * xVal + yVal * yVal + zVal * zVal)
			window_vector.append(mag)

		avg_magnitude = np.mean(window_vector)

		if (abs(avg_magnitude - 9.5) < thresholdMag):
			return 1

		return 0


	# Need to change to time
	def getWindowTime(self):
		return windowSize

	def getRelevantSensors(self):
		return [s.ACCELEROMETER]

	def getName(self):
		return "Table Classifier"




