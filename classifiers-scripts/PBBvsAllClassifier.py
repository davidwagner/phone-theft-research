import BaseClassifier
from sklearn.externals import joblib

import Sensors as s

CLASSIFIER_PATH = './classifier_pickles/TrainedClassifier_PBBvsAll/PBBvsAllClassifier.pkl'
clf = joblib.load(CLASSIFIER_PATH)
class Classifier(BaseClassifier.BaseClassifier):

	def classify(self, windowOfData):
		features = featurizer.dataToFeatures(windowOfData)
		results = clf.predict(features)

		return int(results[0])
		# return results

	# Need to change to time
	def getWindowTime(self):
		return 20

	def getRelevantSensors(self):
		return [s.ACCELEROMETER, s.LIGHT_SENSOR]



def dataToFeatures(windowOfData):

	import numpy
	import math

	accelFeatures = windowOfData[s.ACCELEROMETER]
	time = []
	windowX = []
	windowY = []
	windowZ = []
	for value in accelFeatures:
		time.append(float(value[0]))
		windowX.append(float(value[1]))
		windowY.append(float(value[2]))
		windowZ.append(float(value[3]))



	features = []

	avgX = abs(getAvg(windowX))
	avgY = abs(getAvg(windowY))
	avgZ = abs(getAvg(windowZ))
	stdX = getStd(windowX)
	stdY = getStd(windowY)
	stdZ = getStd(windowZ)

	features.append(avgX)
	features.append(avgY)
	features.append(avgZ)
	features.append(stdX)
	features.append(stdY)
	features.append(stdZ)

	mag = getMagnitude(windowX, windowY, windowZ)
	avgMag = mag[0]
	stdMag = mag[1]
	rangeMag = mag[2]
	features.append(avgMag)
	features.append(stdMag)

	features.append(rangeMag)


	#orientation
	x_angle_mean, x_angle_std = getAngle(windowX, windowY, windowZ)
	y_angle_mean, y_angle_std = getAngle(windowY, windowX, windowZ)
	z_angle_mean, z_angle_std = getAngle(windowZ, windowX, windowY)

	features.append(x_angle_mean)
	features.append(x_angle_std)
	features.append(y_angle_mean)
	features.append(y_angle_std)
	features.append(z_angle_mean)
	features.append(z_angle_std)

	meanFFT, stdFFT = fft(windowX, windowY, windowZ)
	features.append(meanFFT)
	features.append(stdFFT)

	xChange, yChange, zChange =  changeSign(windowX, windowY, windowZ)
	features.append(xChange)
	features.append(yChange)
	features.append(zChange)

	
	# print("wyee")
	lightReading = getLightSensor(time[0],time[9], windowOfData[s.LIGHT_SENSOR])

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


	return features
# 	allFeatures.append(features)
# 	count += 1

# featureArray = numpy.empty([len(allFeatures), 23])
# for i in range(len(allFeatures)):
# 	featureArray[i] = allFeatures[i]

# if outputfile != None:
# 	numpy.savetxt(outputfile[0] + "/PBvsAllFeatures.txt", featureArray)
# return featureArray;

def fft(windowX, windowY, windowZ):
	mag = []
	for i in range(20):
		windowX[i] * windowX[i]
		windowY[i] * windowY[i]
		windowZ[i] * windowZ[i]
		mag.append(math.sqrt(windowX[i] * windowX[i] + windowY[i] * windowY[i] + windowZ[i] * windowZ[i]))
	fft = numpy.fft.fft(mag)
	fftAbs = numpy.abs(fft)

	return numpy.mean(fftAbs), numpy.std(fftAbs)

def changeSign(windowX, windowY, windowZ):
	import itertools
	xChange = len(list(itertools.groupby(windowX, lambda windowX: windowX > 0))) - (windowX[0] > 0)
	yChange = len(list(itertools.groupby(windowY, lambda windowY: windowY > 0))) - (windowY[0] > 0)
	zChange = len(list(itertools.groupby(windowZ, lambda windowZ: windowZ > 0))) - (windowZ[0] > 0)
	
	return xChange/10.0, yChange/10.0, zChange/10.0


def getAvg(window):
	avg =  abs( numpy.mean(window))
	return float(avg) / (35.0)


def getStd(window):
	std_dev = abs(numpy.std(window))
	return std_dev / (10.0)


def getMagnitude(windowX, windowY, windowZ):
	mag = []
	for i in range(20):
		mag.append(math.sqrt(windowX[i] * windowX[i] + windowY[i] * windowY[i] + windowZ[i] * windowZ[i]))

	return (numpy.mean(mag) / 40.0, numpy.std(mag) / 15.0, max(mag) - min(mag) / 40.0)

def getAngle(a, b, c):
	angle = []
	for i in range(20):
		angle.append(math.atan(a[i] / (b[i] * b[i] + c[i] * c[i])))

	norm_mean = numpy.mean(angle) - (math.pi / 2)  / math.pi
	norm_std = numpy.std(angle) - (math.pi/2) / math.pi

	return norm_mean, norm_std
def getLightSensor(start, end, lightRows):

	max_value = 100000
	if float(lightRows[0][0]) > end:
		return float(lightRows[0][1]) / max_value

	if float(lightRows[len(lightRows)-1][0]) < end:
		return float(lightRows[len(lightRows)-1][1]) /max_value

	for i in range(len(lightRows)):
		if float(lightRows[i][0]) < start:
			continue
		if float(lightRows[i][0]) > end:
			return float(lightRows[i-1][1]) / max_value

		return float(lightRows[i][1])
	return float(lightRows[len(lightRows)-1][1])  / max_value







