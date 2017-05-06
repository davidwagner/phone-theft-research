import os
import csv
import math
import numpy
from sklearn.externals import joblib
import Sensors as s

SIZE = 10

class Featurizer():
    """docstring for Featurizer"""
    def __init__(self):
        1+1

def getWindows(size, rows):
    windows = []
    for i in range(len(rows) - size):
        windowX = []
        windowY= []
        windowZ = []
        for j in range(size):
            windowX.append(float(rows[i + j][1]))
            windowY.append(float(rows[i + j][2]))
            windowZ.append(float(rows[i + j][3]))
            temp = windowX, windowY, windowZ
            windows.append(temp)

    return windows

k = 0

def getFeatures(path, isPositive):
    # features = numpy.array(10,10)
    # labels = []

    """Useful methods:
    a.resize(new_shape)

    """

    # path = '/some/path/to/file'
    allFeatures = []
    labelArray = []


    for filename in os.listdir(path):
        rows = []
        with open(path +'/' +  filename) as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        
        featureWindows = getWindows(10, rows)
        #featureWindows = getWindows(10, rows[k:SIZE + k + 1])

        """Calculate Features here"""
        for window in featureWindows:
            features = []
            xValues = window[0]
            yValues = window[1]
            zValues = window[2]

            """Features 1, 2, 3: averages of x, y, z"""
            
            avgX = numpy.nanmean(xValues)
            avgY = numpy.nanmean(yValues)
            avgZ = numpy.nanmean(zValues)

            features.append(avgX)
            features.append(avgY)
            features.append(avgZ)


            """Features 4, 5, 6: std. deviation if x, y, z"""
            stdDevX = numpy.std(xValues)
            stdDevY = numpy.std(yValues)
            stdDevZ = numpy.std(zValues)
            features.append(stdDevX)
            features.append(stdDevY)
            features.append(stdDevZ)

            """Feature 7: Magnitude"""
            magnitude = []
            for i in range(len(xValues)):
                mag = math.sqrt(xValues[i] * xValues[i] + yValues[i] * yValues[i] + zValues[i] * zValues[i])
                magnitude.append(mag)
            features.append(numpy.mean(magnitude))

            allFeatures.append(features)
            if isPositive:
                labelArray.append(1)
            else:
                labelArray.append(0)

            """a = np.array([[1,3,4],[1,2,3],[1,2,1]])
                b = np.array([10,20,30])
                c = np.hstack((a, np.atleast_2d(b).T))"""


    featureArray = numpy.empty([len(allFeatures), len(features)])
    for i in range(len(allFeatures)):
        featureArray[i] = allFeatures[i]

    return featureArray, labelArray

def getXYZWindows(rows):
    windows = []

    windowX = []
    windowY= []
    windowZ = []
    for j in range(SIZE):
        windowX.append(float(rows[j][1]))
        windowY.append(float(rows[j][2]))
        windowZ.append(float(rows[j][3]))
        temp = windowX, windowY, windowZ
        windows.append(temp)

    return windows


def dataToFeatures(windowOfData):
    # features = numpy.array(10,10)
    # labels = []

    """Useful methods:
    a.resize(new_shape)

    """

    # path = '/some/path/to/file'
    allFeatures = []
    labelArray = []


    # for filename in os.listdir(path):
    #     rows = []
    #     with open(path +'/' +  filename) as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             rows.append(row)
    #     featureWindows = getWindows(10, rows)

    #     """Calculate Features here"""
    #     for window in featureWindows:

    accelRows = windowOfData[s.ACCELEROMETER]
    # accelRows = windowOfData
    featureWindows = getXYZWindows(accelRows)

    for window in featureWindows:
        features = []
        xValues = window[0]
        yValues = window[1]
        zValues = window[2]

        """Features 1, 2, 3: averages of x, y, z"""
        
        avgX = numpy.nanmean(xValues)
        avgY = numpy.nanmean(yValues)
        avgZ = numpy.nanmean(zValues)

        features.append(avgX)
        features.append(avgY)
        features.append(avgZ)


        """Features 4, 5, 6: std. deviation if x, y, z"""
        stdDevX = numpy.std(xValues)
        stdDevY = numpy.std(yValues)
        stdDevZ = numpy.std(zValues)
        features.append(stdDevX)
        features.append(stdDevY)
        features.append(stdDevZ)

        """Feature 7: Magnitude"""
        magnitude = []
        for i in range(len(xValues)):
            mag = math.sqrt(xValues[i] * xValues[i] + yValues[i] * yValues[i] + zValues[i] * zValues[i])
            magnitude.append(mag)
        features.append(numpy.mean(magnitude))

        allFeatures.append(features)
        # if isPositive:
        #     labelArray.append(1)
        # else:
        #     labelArray.append(0)

        """a = np.array([[1,3,4],[1,2,3],[1,2,1]])
            b = np.array([10,20,30])
            c = np.hstack((a, np.atleast_2d(b).T))"""


    featureArray = numpy.empty([len(allFeatures), len(features)])
    for i in range(len(allFeatures)):
        featureArray[i] = allFeatures[i]

    return featureArray

def main():
    testFile = 'steadyStateTest/AppMon_0d609497-bd0d-4391-a29f-d96b5b20d309_BatchedAccelerometer_2016_09_28_15_12_51_.csv'
    expectedOutput = getFeatures('steadyStateTest/', True)
    #print(expectedOutput)
    expectedFeatures = expectedOutput[0]
    #print(expectedFeatures)

    rows = []
    with open(testFile) as f:
        reader = csv.reader(f)

        for row in reader:
            rows.append(row)


    windowOfData = []
    
    for j in range(k, SIZE + k):
        windowOfData.append(rows[j])
    actualFeatures = dataToFeatures(windowOfData)

    # for i in range(1, len(rows) - SIZE):
    #     windowOfData = []
    #     for j in range(SIZE):
    #         windowOfData.append(rows[i + j])

    #     features = dataToFeatures(windowOfData)
    #     # print(features)
    #     numpy.concatenate((actualFeatures, features), axis=0)

    # print("Success?: ", str(expectedFeatures == actualFeatures))
    
    #print(actualFeatures)

    CLASSIFIER_PATH = 'TrainedClassifier_PhoneSteadyState/PocketSteadyStateClassifier_Beta.pkl'
    clf = joblib.load(CLASSIFIER_PATH)

    expectedResults = clf.predict(expectedFeatures)
    actualResults = clf.predict(actualFeatures)
    print("EXPECTED")
    print(expectedResults)
    print("ACTUAL")
    print(actualResults)




if __name__ == '__main__':
    main()







