import BaseClassifier
from sklearn.externals import joblib

import Sensors as s

CLASSIFIER_PATH = './classifier_pickles/Trained_PBBvsAll/PBBvsAllClassifier.pkl'
clf = joblib.load(CLASSIFIER_PATH)


class Classifier(BaseClassifier.BaseClassifier):

	def classify(self, windowOfData):
		features = self.createFeatures(windowOfData)
		results = clf.predict(features)

		return int(results[0])
		

	# Need to change to time
	def getWindowTime(self):
		return 20

	def getRelevantSensors(self):
		return [s.ACCELEROMETER, s.LIGHT_SENSOR]

	def getName(self):
		return "Pocket/Bag Classifier"


	def createFeatures(self, windowOfData):

		import numpy
		import csv
		import math

		import datetime

		allFeatures = []

		accelRows = windowOfData[s.ACCELEROMETER]
		count = 0
	
		time = []
		windowX = []
		windowY = []
		windowZ = []

		for i in range(20):
			time.append(accelRows[i][0])
			windowX.append(float(accelRows[i][1]))
			windowY.append(float(accelRows[i][2]))
			windowZ.append(float(accelRows[i][3]))


		# Do isDay
		# Compare DateTime

		reference_time_night = datetime.datetime(2000, 1, 1, 18, 0)
		reference_time_day = datetime.datetime(2000, 1, 1, 7, 0)

		current_time = time[0]

		if (current_time.time() > reference_time_day.time() and current_time.time() < reference_time_night.time()):
			isDay = True
		else:
			isDay = False

		features = []

		avgX = abs(self.getAvg(windowX))
		avgY = abs(self.getAvg(windowY))
		avgZ = abs(self.getAvg(windowZ))
		stdX = self.getStd(windowX)
		stdY = self.getStd(windowY)
		stdZ = self.getStd(windowZ)

		features.append(avgX)
		features.append(avgY)
		features.append(avgZ)
		features.append(stdX)
		features.append(stdY)
		features.append(stdZ)

		mag = self.getMagnitude(windowX, windowY, windowZ)
		avgMag = mag[0]
		stdMag = mag[1]
		rangeMag = mag[2]
		features.append(avgMag)
		features.append(stdMag)

		features.append(rangeMag)


		#orientation
		x_angle_mean, x_angle_std = self.getAngle(windowX, windowY, windowZ)
		y_angle_mean, y_angle_std = self.getAngle(windowY, windowX, windowZ)
		z_angle_mean, z_angle_std = self.getAngle(windowZ, windowX, windowY)

		features.append(x_angle_mean)
		features.append(x_angle_std)
		features.append(y_angle_mean)
		features.append(y_angle_std)
		features.append(z_angle_mean)
		features.append(z_angle_std)

		meanFFT, stdFFT = self.fft(windowX, windowY, windowZ)
		features.append(meanFFT)
		features.append(stdFFT)

		xChange, yChange, zChange =  self.changeSign(windowX, windowY, windowZ)
		features.append(xChange)
		features.append(yChange)
		features.append(zChange)

		
		# print("wyee")
		lightReading = float(windowOfData[s.LIGHT_SENSOR][0][1])
	
		if not isDay:
			features.append(0)
			features.append(0)
			features.append(1)
		elif isDay and lightReading <= 20:
			features.append(1)
			features.append(0)
			features.append(0)
		else:
			features.append(0)
			features.append(1)
			features.append(0)

		featureArray = numpy.empty([1, 23])
		featureArray[0] = features

		return featureArray

	def fft(self, windowX, windowY, windowZ):
		import numpy
		import math

		mag = []
		for i in range(20):
			windowX[i] * windowX[i]
			windowY[i] * windowY[i]
			windowZ[i] * windowZ[i]
			mag.append(math.sqrt(windowX[i] * windowX[i] + windowY[i] * windowY[i] + windowZ[i] * windowZ[i]))
		fft = numpy.fft.fft(mag)
		fftAbs = numpy.abs(fft)

		return numpy.mean(fftAbs), numpy.std(fftAbs)

	def changeSign(self, windowX, windowY, windowZ):
		import itertools
		xChange = len(list(itertools.groupby(windowX, lambda windowX: windowX > 0))) - (windowX[0] > 0)
		yChange = len(list(itertools.groupby(windowY, lambda windowY: windowY > 0))) - (windowY[0] > 0)
		zChange = len(list(itertools.groupby(windowZ, lambda windowZ: windowZ > 0))) - (windowZ[0] > 0)
		
		return xChange/10.0, yChange/10.0, zChange/10.0


	def getAvg(self, window):

		import numpy

		avg =  abs( numpy.mean(window))
		return float(avg) / (35.0)


	def getStd(self, window):
		import numpy

		std_dev = abs(numpy.std(window))
		return std_dev / (10.0)


	def getMagnitude(self, windowX, windowY, windowZ):
		import numpy
		import math
		mag = []
		for i in range(20):
			mag.append(math.sqrt(windowX[i] * windowX[i] + windowY[i] * windowY[i] + windowZ[i] * windowZ[i]))

		return (numpy.mean(mag) / 40.0, numpy.std(mag) / 15.0, max(mag) - min(mag) / 40.0)

	def getAngle(self, a, b, c):
		import numpy
		import math
		angle = []

		for i in range(20):
			try:
				angle.append(math.atan(a[i] / (b[i] * b[i] + c[i] * c[i])))
			except(ZeroDivisionError):
				if a[i] > 0:
					angle.append(math.pi / 2)
				else:
					angle.append(-1 * math.pi / 2)
		norm_mean = numpy.mean(angle) - (math.pi / 2)  / math.pi
		norm_std = numpy.std(angle) - (math.pi/2) / math.pi

		return norm_mean, norm_std
	

#only for testing
# def main():
# 	classifier = Classifier()
# 	window = dict()

# 	import datetime
# 	sample_date = datetime.datetime(2000, 1, 1, 17,0) 

# 	accel = []
# 	light = []
# 	for _ in range(20):
# 		accel.append((sample_date, 9.81, 0, 0))
# 		light.append((sample_date, 5))

# 	window[s.ACCELEROMETER] = accel
# 	window[s.LIGHT_SENSOR] = light


# 	result = classifier.classify(window)
# 	print(result)

# if __name__ == '__main__':
# 	main()
