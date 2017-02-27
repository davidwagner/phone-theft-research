import os
import csv
import math
import numpy
import ClassifierLog

class Featurizer():
    """docstring for Featurizer"""
    def __init__(self):
        1+1

WINDOW_SIZE = 50
def getWindows(size, rows):
    windows = []
    for i in range(len(rows) - size + 1):
        windowX = []
        windowY= []
        windowZ = []
        windowTouch = []
        windowScreen = []
        windowLocked = []
        windowSignX = []
        windowSignY = []
        windowSignZ = []
        for j in range(size):
            windowX.append(float(rows[i + j][1]))
            windowY.append(float(rows[i + j][2]))
            windowZ.append(float(rows[i + j][3]))
            windowTouch.append(float(rows[i + j][4]))
            windowScreen.append(float(rows[i + j][5]))
            windowLocked.append(float(rows[i + j][6]))
            windowSignX.append(float(rows[i + j][7]))
            windowSignY.append(float(rows[i + j][8]))
            windowSignZ.append(float(rows[i + j][9]))
            
        temp = windowX, windowY, windowZ, windowTouch, windowScreen, windowLocked, windowSignX, windowSignY, windowSignZ
        windows.append(temp)

    return windows



def getFeatures(rows):
    # features = numpy.array(10,10)
    # labels = []

    """Useful methods:
    a.resize(new_shape)

    """

    # path = '/some/path/to/file'
    allFeatures = []
    labelArray = []



    featureWindows = getWindows(WINDOW_SIZE, rows)

    # print(len(rows))
    # print("Windows to process: " + str(len(featureWindows)))
    count = 0

    """Calculate Features here"""
    for window in featureWindows:
        features = []
        xValues = window[0]
        yValues = window[1]
        zValues = window[2]
        touchValues = window[3]
        screenValues = window[4]
        lockedValues = window[5]
        signXValues = window[6]
        signYValues = window[7]
        signZValues = window[8]

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
        
        """ Feature 8: Number of Touches, Screen State, Screen Locked """
        numTouches = numpy.sum(touchValues)
        screenState = numpy.nanmean(screenValues)
        lockedState = numpy.nanmean(lockedValues)
        features.append(numTouches)
        features.append(screenState)
        features.append(lockedState)
            
        """ Feature 9, 10, 11: Number of acceleration sign changes, X, Y, Z"""
        signChangesX = numpy.sum(signXValues)
        signChangesY = numpy.sum(signYValues)
        signChangesZ = numpy.sum(signZValues)
        features.append(signChangesX)
        features.append(signChangesY)
        features.append(signChangesZ)
        
        allFeatures.append(features)
        
        # count += 1
        # if count % 10000 == 0:
        #     print("Finished featurizing: " + str(count))
        """a = np.array([[1,3,4],[1,2,3],[1,2,1]])
            b = np.array([10,20,30])
            c = np.hstack((a, np.atleast_2d(b).T))"""


    featureArray = numpy.empty([len(allFeatures), len(features)])
    for i in range(len(allFeatures)):
        featureArray[i] = allFeatures[i]

    # print("Finished Featurizing!")
    return featureArray



