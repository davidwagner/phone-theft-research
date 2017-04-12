import csv
import os
import glob
import re
import datetime
import sys
import shutil
import argparse
# import matplotlib.pyplot as plt
# from matplotlib.dates import SecondLocator, MinuteLocator, HourLocator, DateFormatter, date2num
# import classifier
import Sensors as sensors
import Classifiers as classifiers 
import PossessionState
import pickle
import traceback

from configsettings import *
from collections import deque, Counter
from UnlockTimeChecker import computeUnlocks

DUMP_RESULTS = True

if DIRECTORY[-1] != '/':
    DIRECTORY += '/'

NOW = datetime.datetime.now()
# NOW_DAY = NOW.strftime('%Y_%m_%d')

YESTERDAY = (NOW - datetime.timedelta(days=1)).strftime('%Y_%m_%d')
# NOW_DAY = YESTERDAY
NOW_DAY = '2016_11_01'

# RELEVANT_SENSORS = set([])
# RELEVANT_SENSORS = [sensors.ACCELEROMETER, sensors.PHONE_ACTIVE_SENSORS]
RELEVANT_SENSORS = [sensors.ACCELEROMETER, sensors.PHONE_ACTIVE_SENSORS, sensors.LIGHT_SENSOR]
HEARTRATE_SENSOR = sensors.HEART_RATE
# BLUETOOTH_SENSOR = sensors.BLUETOOTH_CONNECTED
# WATCH_SENSORS = [HEARTRATE_SENSOR, BLUETOOTH_SENSOR]
WATCH_SENSORS = [HEARTRATE_SENSOR, sensors.CONNECTED_DEVICES]
YEAR_2000 = datetime.date(2000, 1, 1)

BOOT_TIME_DELTA = datetime.timedelta(hours=1)
BOOT_TIME_SENSOR = sensors.ACCELEROMETER
START_OF_TIME = datetime.datetime.min

SANITY_TEST = False
maxWindowSize = 100

START_TIME_FILTER = datetime.time(hour=8)
END_TIME_FILTER = datetime.time(hour=22)
# START_TIME_FILTER = None
# END_TIME_FILTER = None


def getUserFilesByDayAndInstrument(userID, instrument):
    query = DIRECTORY + 'AppMon_' + userID + '*_' + instrument + '_' + '*'
    userFiles = glob.glob(query)
    userFiles.sort()

    # TODO: Need to filter for sensors that need data files with matching times as other
    # sensors (e.g. accelerometer and step count for Theft Classifier)
    # print(userFiles)
    return userFiles


def dataFilesToDataList(userFiles, bootTimes, needsToComputeBootTime=False):
    dataList = []
    currentBootTime = START_OF_TIME
    nextFileTime = START_OF_TIME
    nextFileTimeIndex = 0

    # print("USER FILES")
    # print(userFiles)
    
    for dataFile in userFiles:
        with open(dataFile) as f:  
            reader = csv.reader(f)
            
            fileTime = timeStringToDateTime(getTimeFromFile(dataFile))

            firstRow = next(reader)
            firstTime = datetime.timedelta(milliseconds=int(firstRow[0]))

            if needsToComputeBootTime:
                bootTime = fileTime - firstTime

                difference = bootTime - currentBootTime if bootTime > currentBootTime else currentBootTime - bootTime

                if difference > BOOT_TIME_DELTA:
                    currentBootTime = bootTime 
                    bootTimes.append((fileTime, bootTime))
            
            else:
                # print("FileTime", str(fileTime))
                # print("NextFileTIme", str(nextFileTime))
                if fileTime > nextFileTime:
                    currentBootTime = bootTimes[nextFileTimeIndex][1] # boot time has changed, update
                    # print("Current Boot Time:", currentBootTime)
                    nextFileTimeIndex = nextFileTimeIndex + 1 if nextFileTimeIndex < len(bootTimes) - 1 else nextFileTimeIndex
                    nextFileTime = bootTimes[nextFileTimeIndex][0]

            firstRow[0] = convertToDateTime(firstRow[0], currentBootTime)
            minLength = len(firstRow)
            if len(firstRow) >= 2:
                if (START_TIME_FILTER == None or firstRow[0].time() >= START_TIME_FILTER) and (END_TIME_FILTER == None or firstRow[0].time() < END_TIME_FILTER):
                    dataList.append(firstRow)
            count = 1
            for row in reader:
                if len(row) >= 2 and len(row) >= minLength:
                    row[0] = convertToDateTime(row[0], currentBootTime)
                    if (START_TIME_FILTER == None or firstRow[0].time() >= START_TIME_FILTER):
                        if (END_TIME_FILTER == None or firstRow[0].time() < END_TIME_FILTER):
                            dataList.append(row)
                        else:
                            return dataList

                if SANITY_TEST:
                    count += 1
                    if count > 10000:
                        break
    # print("DATA LIST")
    # print(len(dataList))
    return dataList

def dataFilesToDataListAbsTime(userFiles):
    dataList = []
    for dataFile in userFiles:
        with open(dataFile) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1:
                    timestamp = int(row[1]) / 1000
                    row[0] = datetime.datetime.fromtimestamp(timestamp)
                    if (START_TIME_FILTER != None or firstRow[0].time() >= START_TIME_FILTER):
                        if (END_TIME_FILTER != None or firstRow[0].time() < END_TIME_FILTER):
                            dataList.append(row)
    # print "Number of heartrate files"
    # print len(dataList)
    return dataList

def getReferenceBootTimes(userID):
    userFiles = getUserFilesByDayAndInstrument(userID, BOOT_TIME_SENSOR)

    bootTimes = []
    currentBootTime = START_OF_TIME
    
    for dataFile in userFiles:
        with open(dataFile) as f:  
            reader = csv.reader(f)
            
            fileTime = timeStringToDateTime(getTimeFromFile(dataFile))

            firstRow = reader.next()
            firstTime = datetime.timedelta(milliseconds=int(firstRow[0]))

            bootTime = fileTime - firstTime

            difference = bootTime - currentBootTime if bootTime > currentBootTime else currentBootTime - bootTime

            if difference > BOOT_TIME_DELTA:
                currentBootTime = bootTime 
                bootTimes.append((fileTime, bootTime))

    return bootTimes



def getRelevantUserData(userID, logInfo=False, logFile=None):
    userData = {}
    bootTimes = []

    dataFiles = getUserFilesByDayAndInstrument(userID, BOOT_TIME_SENSOR)
    userData[BOOT_TIME_SENSOR] = dataFilesToDataList(dataFiles, bootTimes, True)

    for instrument in RELEVANT_SENSORS:
        if instrument != BOOT_TIME_SENSOR and instrument != sensors.PHONE_ACTIVE_SENSORS:
            
            dataFiles = getUserFilesByDayAndInstrument(userID, instrument)
            userData[instrument] = dataFilesToDataList(dataFiles, bootTimes)
    
    #print(len(userData[sensors.ACCELEROMETER]))
    userData[sensors.PHONE_ACTIVE_SENSORS], userData[sensors.KEYGUARD] = processPhoneActiveData(userID, userData[sensors.ACCELEROMETER])
    print("KEYGUARD", len(userData[sensors.KEYGUARD]))

    # print("GONNA TRY TO GET LIGHT SENSOR DATA")
    userData[sensors.LIGHT_SENSOR] = processLightSensorData(userData)
    userData[BOOT_TIME_SENSOR] = userData[BOOT_TIME_SENSOR][:-1]
    print("Length accel:", len(userData[BOOT_TIME_SENSOR]))
    print("Length active:", len(userData[sensors.PHONE_ACTIVE_SENSORS]))

    for instrument in WATCH_SENSORS:
        dataFiles = getUserFilesByDayAndInstrument(userID, instrument)
        # print "Heart Rate Files"
        # print dataFiles
        userData[instrument] = dataFilesToDataListAbsTime(dataFiles)

    if logInfo:
        logFile.write("Data Files Analyzed:\n")
        for filename in dataFiles:
            logFile.write(getTimeFromFile(filename) + "_.csv" + '\n')
        logFile.write("Boot Times Computed:\n")
        for bootTime in bootTimes:
            logFile.write("Files after " + str(formatTime(bootTime[0], withDate=True)) + ", have boot time: " + str(formatTime(bootTime[1], withDate=True)) + '\n')

    return userData

def processLightSensorData(userData):
    
    dataAccel = userData[sensors.ACCELEROMETER]
    if len(dataAccel) <= 1:
        return []
    dataLight = userData[sensors.LIGHT_SENSOR]
    dataLightProcessed = []
    firstAccelTime = dataAccel[0][0]
    
    firstLightTime = None
    firstLightValue = None

    currentLightIndex = -1
    startLightIndex = -1
    accelIndex = 0
    if len(dataLight) <= 1:
        return

    currentLightIndex = 0
    currentTime = dataLight[currentLightIndex][0]
    prevLightValue = None
    while currentTime < firstAccelTime: 
        currentLightIndex += 1
        if currentLightIndex >= len(dataLight):
            break
        currentTime = dataLight[currentLightIndex][0]
        prevLightValue = dataLight[currentLightIndex][1]

    startLightIndex = currentLightIndex
    firstLightTime = dataLight[currentLightIndex][0]
    firstLightValue = dataLight[currentLightIndex][1] if prevLightValue == None else prevLightValue

    # print("GOT OUT OF FIRST WHILE")
    currentAccelTime = dataAccel[accelIndex][0]
    while currentAccelTime < firstLightTime:
        lightRow = [currentAccelTime, firstLightValue]
        dataLightProcessed.append(lightRow)
        accelIndex += 1
        currentAccelTime = dataAccel[accelIndex][0]


    currentLightDate = dataLight[currentLightIndex][0]
    nextLightDate = dataLight[currentLightIndex + 1][0]

    # print("NOW ADDING DATA")
    for i in range(accelIndex, len(dataAccel) - 1):
        accelRow = dataAccel[i]
        accelRowNext = dataAccel[i + 1]
        
        accelDate = accelRow[0]
        accelDateNext = accelRowNext[0]
        
        currentLightVal = dataLight[currentLightIndex][1]

        if accelDate >= nextLightDate:
            if currentLightIndex + 1 < len(dataLight):
                currentLightIndex += 1
                currentLightDate = dataLight[currentLightIndex][0]
                if currentLightIndex + 1 < len(dataLight):
                    nextLightDate = dataLight[currentLightIndex + 1][0]
                
            lightRow = [accelDate, currentLightVal]
            dataLightProcessed.append(lightRow)
        
        else:
            lightRow = [accelDate, currentLightVal]
            dataLightProcessed.append(lightRow)

    return dataLightProcessed

def continuousWatchInterals(userID):
    watchData = getRelevantUserData(userID)
    delta = datetime.timedelta(seconds=60)
    allIntervals = {}

    for instrument in WATCH_SENSORS:
        startTime = -1
        prevTime = -1
        intervals = []
        prevState = -1
        watchTimes = watchData[instrument]

        for row in watchTimes:
            time = row[0]
            state = row[-1]
            if startTime == -1:
                startTime = time
            elif time - prevTime > delta or prevState != state:
                intervals.append((startTime, prevTime, prevState))
                startTime = time
            prevState = state
            prevTime = time

        intervals.append((startTime, prevTime, prevState))
        allIntervals[instrument] = intervals 
    return allIntervals

# returns interval and state (1 = phone is near, 0, phone is not near, -1 unknown state)
def stateFromWatchData(allIntervals, file):
    i = 0
    j = 0
    bluetoothIntervals = allIntervals[sensors.CONNECTED_DEVICES]
    heartRateIntervals = allIntervals[HEARTRATE_SENSOR]
    states = ["phoneNear", "phoneFar", "unknown"]
    allIntervals = []
    for h in heartRateIntervals:
        start, end, state = h
        allIntervals.append((start, end, "phoneNear"))
    basisPeakIntervals = []
    for b in bluetoothIntervals:
        start, end, state = b
        if str(state) == "Basis Peak":
            basisPeakIntervals.append((start, end))
    while i < len(heartRateIntervals) and j < len(basisPeakIntervals):
        hInterval = heartRateIntervals[i]
        bInterval = basisPeakIntervals[j]
        hStart, hEnd, hState = hInterval
        bStart, bEnd = bInterval
        while bEnd < hStart  and j < len(basisPeakIntervals):
            bStart, bEnd = basisPeakIntervals[j]
            allIntervals.append((bStart, bEnd, "unknown"))
            j += 1
        while hEnd < bStart and i < len(heartRateIntervals):
            hStart, hEnd, hState = heartRateIntervals[i]
            i += 1
        if bStart < hStart and bEnd > hEnd:
            allIntervals.append((bStart, hStart, "unknown"))
            if j < len(basisPeakIntervals):
                basisPeakIntervals[j] = ((hStart, bEnd))
        if bEnd < hEnd:
            j += 1
        if hEnd < bEnd:
            i += 1

    while j < len(basisPeakIntervals):
        bStart, bEnd = basisPeakIntervals[j]
        allIntervals.append((bStart, bEnd, "unknown"))
        j += 1
    result = []
    prevTime = -1
    for start, end in basisPeakIntervals:
        print(start, end)
        if prevTime == -1:
            prevTime = end
        elif start > prevTime:
            allIntervals.append((prevTime, start, "phoneFar"))
        prevTime = end
    allIntervals = sorted(allIntervals, key=lambda x: x[0])
    logString = ""
    result = {}
    for start, end, state in allIntervals:
        if state not in result:
            result[state] = [(start, end)]
        else:
            result[state].append((start, end))
        logString += state + " : ("  + str(start) + ", " + str(end) + ")"
        logString += "\n"
    file.write(logString)
    return result

def watchActivationStates(watchStates):
    activated = []
    deactivated = []
    activated = watchStates["phoneNear"]
    deactivated += watchStates["unknown"]
    deactivated += watchStates["phoneFar"]
    
    deactivated = sorted(deactivated, key=lambda x: x[0])
    print("DEACTIVATED WATCH:", deactivated)
    mergeAdjacentIntervals(deactivated)
    return activated, deactivated



def processPhoneActiveData(ID, posDataAccel):
    if len(posDataAccel) <= 1:
        return []

    firstAccelTime = posDataAccel[0][0]
    
    posFilesTouch = getUserFilesByDayAndInstrument(ID, 'TouchScreenAsEvent')
    rawPosDataTouch = dataFilesToDataListAbsTime(posFilesTouch)
    # # print("RAW DATA TOUCH")
    # # print(rawPosDataTouch)
    
    posFilesScreen = getUserFilesByDayAndInstrument(ID, 'TriggeredScreenState')
    rawPosDataScreen = dataFilesToDataListAbsTime(posFilesScreen)
    
    posFilesLocked = getUserFilesByDayAndInstrument(ID, 'TriggeredKeyguard')
    rawPosDataLocked = dataFilesToDataListAbsTime(posFilesLocked)

    currScreenDate = None
    nextScreenDate = None
    currScreenVal = None
    currLockedDate = None
    nextLockedDate = None
    currLockedVal = None
    
    touchIndex = -1
    if len(rawPosDataTouch) > 0:

        touchIndex = 0
        currentTime = rawPosDataTouch[touchIndex][0]
        while currentTime < firstAccelTime: 
            touchIndex += 1
            if touchIndex >= len(rawPosDataTouch):
                break
            currentTime = rawPosDataTouch[touchIndex][0]

        startTouchIndex = touchIndex
    
    screenIndex = -1
    if len(rawPosDataScreen) > 0:
        screenIndex = 0
        currentTime = rawPosDataScreen[screenIndex][0]
        while currentTime < firstAccelTime:
            screenIndex += 1
            if screenIndex >= len(rawPosDataScreen):
                break
            currentTime = rawPosDataScreen[screenIndex][0]
            # # print(currentTime)
            # # print(screenIndex)
        currScreenDate = rawPosDataScreen[screenIndex][0]
        currScreenVal = rawPosDataScreen[screenIndex][2]
        if len(rawPosDataScreen) > 1:
            nextScreenDate = rawPosDataScreen[screenIndex + 1][0]
    
    lockedIndex = -1
    if len(rawPosDataLocked) > 0:
        lockedIndex = 0
        currentTime = rawPosDataLocked[lockedIndex][0]
        while currentTime < firstAccelTime:
            lockedIndex += 1
            if lockedIndex >= len(rawPosDataLocked):
                break
            currentTime = rawPosDataLocked[lockedIndex][0]
        currLockedDate = rawPosDataLocked[lockedIndex][0]
        currLockedVal = rawPosDataLocked[lockedIndex][2]
        if len(rawPosDataLocked) > 1:
            nextLockedDate = rawPosDataLocked[lockedIndex + 1][0]
    
    posDataTouch = []
    posDataScreen = []
    posDataLocked = []
    
    # # print(firstAccelTime)
    # # print(screenIndex)
    
    truthToNum = lambda x : 0 if str(x) == 'false' else 1
    
    for i in range(len(posDataAccel) - 1):
        accelRow = posDataAccel[i]
        accelRowNext = posDataAccel[i + 1]
        
        accelDate = accelRow[0]
        accelDateNext = accelRowNext[0]
        
        # Calculate number of touch events starting at this row time and before next row time
        # touchDate >= firstAccelTime
        if len(rawPosDataTouch) == 0 or touchIndex >= len(rawPosDataTouch) or rawPosDataTouch[touchIndex][0] >= accelDateNext: # No touch events
            touchRow = [accelDate, 0]
            posDataTouch.append(touchRow)
            ## print("TOUCH DATE:" + str(touchDate))
            ## print("ACCEL DATE:" + str(accelDate))
            
        else: #touchDate < AccelDateNext
            numTouches = 0
            touchDate = rawPosDataTouch[touchIndex][0]
            while touchDate < accelDateNext and touchIndex < len(rawPosDataTouch):
                # # print("TOUCH RECOGNIZED!")
                numTouches += 1
                touchIndex += 1
                if touchIndex < len(rawPosDataTouch) - 1:
                    touchDate = rawPosDataTouch[touchIndex][0]
                
            touchRow = [accelDate, numTouches]
            posDataTouch.append(touchRow)
        
        
        # Calculate if screen on in this interval
        if currScreenDate == None or nextScreenDate == None:
            screenRow = [accelDate, 0]
            posDataScreen.append(screenRow)

        elif accelDate >= nextScreenDate:
            if screenIndex + 1 < len(rawPosDataScreen):
                screenIndex += 1
                currScreenDate = rawPosDataScreen[screenIndex][0]
                currScreenVal = rawPosDataScreen[screenIndex][2]
                if screenIndex + 1 < len(rawPosDataScreen):
                    nextScreenDate = rawPosDataScreen[screenIndex + 1][0]
                
            screenRow = [accelDate, truthToNum(currScreenVal)]
            posDataScreen.append(screenRow)
        
        else:
            screenRow = [accelDate, truthToNum(currScreenVal)]
            posDataScreen.append(screenRow)
        
        # Calculate if locked on in this interval

        if currLockedDate == None or nextLockedDate == None:
            screenRow = [accelDate, 0]
            posDataLocked.append(screenRow)
        elif accelDate >= nextLockedDate:
            if lockedIndex + 1 < len(rawPosDataLocked):
                lockedIndex += 1
                currLockedDate = rawPosDataLocked[lockedIndex][0]
                currLockedVal = rawPosDataLocked[lockedIndex][2]
                if lockedIndex + 1 < len(rawPosDataLocked):
                    nextLockedDate = rawPosDataLocked[lockedIndex + 1][0]
                
            lockedRow = [accelDate, truthToNum(currLockedVal)]
            posDataLocked.append(lockedRow)
        
        else:
            lockedRow = [accelDate, truthToNum(currLockedVal)]
            posDataLocked.append(lockedRow)
            
    posData = []
    curAccelSignX = float(posDataAccel[0][1]) > 0
    curAccelSignY = float(posDataAccel[0][2]) > 0
    curAccelSignZ = float(posDataAccel[0][3]) > 0
    
    curSigns = [curAccelSignX, curAccelSignY, curAccelSignZ]
    
    signsChanged = lambda now, cur : [1 if now[i] != cur[i] else 0 for i in range(len(now))]
    for i in range(len(posDataAccel) - 1):
        try:
            accelSignX = float(posDataAccel[i][1]) > 0
            accelSignY = float(posDataAccel[i][2]) > 0
            accelSignZ = float(posDataAccel[i][3]) > 0
            
            newSigns = [accelSignX, accelSignY, accelSignZ]
            accelSigns = signsChanged(newSigns, curSigns)
            curSigns = newSigns
            
            
            numTouches = posDataTouch[i][1]
            screenState = posDataScreen[i][1]
            lockedState = posDataLocked[i][1]
            
            row = [posDataAccel[i][0]] + [numTouches, screenState, lockedState] + accelSigns
            posData.append(row)

        except (ValueError,IndexError):
            print("BAD VALUE OF I:", i)
            numTouches = posDataTouch[i][1]
            screenState = posDataScreen[i][1]
            lockedState = posDataLocked[i][1]
            
            row = [posDataAccel[i][0]] + [numTouches, screenState, lockedState] + signsChanged(curSigns, curSigns)
            posData.append(row)
    
    return posData, rawPosDataLocked




def runClassifier(classifier, userData):
    windowSize = classifier.getWindowTime()
    instruments = classifier.getRelevantSensors()
    
    # numRows = min([len(userData[instrument]) for instrument in instruments])
    # # print(len(userData[sensors.ACCELEROMETER]))

    classifications = []

    for i in range(maxWindowSize // windowSize):
        windowOfData = {}
        for instrument in instruments:
            start = i * windowSize
            end = (i + 1) * windowSize
            windowOfData[instrument] = userData[instrument][start:end]
        classification = classifier.classify(windowOfData)
        classifications.append(classification)
    
    return classifications

# {windowStartTime : 0, 1}
# {7:30pm : 0}

# {0 : [list of times], 1 : [list of times]}

def getHeartRateTimes(userData):
    heartRateData = userData[HEARTRATE_SENSOR]
    intervals = []
    # # print "Heart rate data"
    # # print len(heartRateData)
    if len(heartRateData) <= 0:
        return intervals

    currentWindowStartTime = heartRateData[0][0]
    currentWindowEndTime = currentWindowStartTime
    THRESHOLD = datetime.timedelta(minutes=5)
    for row in heartRateData:
        time = row[0]
        if time - currentWindowEndTime > THRESHOLD:
            interval = (currentWindowStartTime, currentWindowEndTime)
            intervals.append(interval)
            currentWindowStartTime = time
            currentWindowEndTime = time
        else:
            currentWindowEndTime = time
    interval = (currentWindowStartTime, currentWindowEndTime)
    intervals.append(interval)

    return intervals



def runClassifiersOnUser(userID, csvWriter, resultsFile):
    if DUMP_RESULTS:
        resultsFile.write("###########################\n")
        resultsFile.write(str(userID) + '\n')
        resultsFile.write("###########################\n")
    # print(userID)
    userData = getRelevantUserData(userID, logInfo=True, logFile=resultsFile)
    heartRateTimes = getHeartRateTimes(userData)

    csvRow = [userID]
    results = {}
    pickleResults = {}

    for instrument in RELEVANT_SENSORS:
        print(instrument, ":", len(userData[instrument]))    

    numRows = min([len(userData[instrument]) for instrument in RELEVANT_SENSORS])

    classifications = []
    intervalsByClass = {}
    SMOOTHING_NUM = 10
    resultsBuffer = deque()
    resultsCounter = Counter()
    currentClass = -1
    firstTime = userData[sensors.ACCELEROMETER][0][0]
    currentInterval = (firstTime, firstTime)

    print("HELLO CLASSIFIERS")
    for c in classifiers.CLASSIFIERS:
        print(c)
        intervalsByClass[c] = []
    intervalsByClass["Unknown"] = []

    limit = numRows // maxWindowSize * maxWindowSize
    print("LIMIT", limit)

    possessionState = PossessionState.PossessionState(userData[sensors.PHONE_ACTIVE_SENSORS], userData[sensors.KEYGUARD], SMOOTHING_NUM)
    for i in range(0, limit, maxWindowSize):
        windowOfData = {}
        windowStartTime = 0
        for instrument in RELEVANT_SENSORS:
            data = userData[instrument][i:i + maxWindowSize] 
            windowOfData[instrument] = data
            windowStartTime = getWindowStartTime(data)
        
        if i % 50000 == 0:
            print("i:", i)
            print("NumRows:", numRows)
        classifierResults = {}
        for c in classifiers.CLASSIFIERS:
            classifier = classifiers.CLASSIFIERS[c]
            results = runClassifier(classifier, windowOfData)
            classifierResults[classifier] = results
            # if i % 50000 == 0 and c ==classifiers.HAND_CLASSIFIER:
            #     print("TYPE: ", type(results))

        logString = windowStartTime.strftime("%H:%M:%S") + "| " 

        windowClassification = classifierPolicy(classifierResults)
        logString += "___" + windowClassification[0] + "___ " 
        for c, results in classifierResults.items():
            r = Counter()
            for result in results:
                r[result] += 1
            r = dict(r)
            logString += '"' + c.getName()[0] + '"' + ":" + str(r) + "; "
        logString += "\n"
        resultsFile.write(logString)

        resultsBuffer.append((windowStartTime, windowClassification))
        resultsCounter[windowClassification] += 1
        # print("WINDOW START TIME:", windowStartTime)
        if len(resultsBuffer) >= SMOOTHING_NUM:
            middleWindow = resultsBuffer[SMOOTHING_NUM // 2]
            middleWindowStartTime = middleWindow[0]
            # print("MIDDLE WINDOW START TIME:", middleWindowStartTime)
            newClassification = resultsCounter.most_common(1)[0][0]
            
            possessionState.updateState(middleWindowStartTime, newClassification)
            if currentClass == -1:
                currentClass = newClassification
                # currentInterval = (middleWindowStartTime, middleWindowStartTime)
            elif currentClass != newClassification:
                classifications.append((currentInterval, currentClass))
                intervalsByClass[currentClass].append(currentInterval)
                interval = currentInterval
                currentInterval = (middleWindowStartTime, middleWindowStartTime)
                currentClass = newClassification
            else:
                currentInterval = (currentInterval[0], middleWindowStartTime)

            removed = resultsBuffer.popleft()
            removedClassification = removed[1]
            resultsCounter[removedClassification] -= 1

    print("HELLOOOOOOOOO")

    classifications.append((currentInterval, currentClass))
    print("CURRENT CLASS:", currentClass)
    intervalsByClass[currentClass].append(currentInterval)

    return classifications, intervalsByClass, possessionState


def logResultsToFile(classifierResults, classifier_name, resultsFile):
    resultsFile.write("-------------------------------------\n")
    resultsFile.write(str(classifier_name) + '\n')
    resultsFile.write("-------------------------------------\n")
    resultIntervals, resultIntervalsByValue = classifierResults
    resultsFile.write("Result Intervals\n")
    for interval in resultIntervals:
        interval = (formatTimeInterval(interval[0], withDate=True), interval[1])
        resultsFile.write(str(interval) + '\n')

    posTimes = resultIntervalsByValue[1]
    negTimes = resultIntervalsByValue[0]

    resultsFile.write("Positive Intervals\n")
    for interval in posTimes:
        resultsFile.write(formatTimeInterval(interval, withDate=True) + ' ; ' + formatTimeValue(intervalLength(interval)) + '\n')

    resultsFile.write("Negative Intervals\n")
    for interval in negTimes:
        resultsFile.write(formatTimeInterval(interval, withDate=True) + ' ; ' + formatTimeValue(intervalLength(interval)) + '\n')



def processTheftResults(results, writer, csvRow):
    resultIntervals, resultIntervalsByValue = results[0], results[1]

    posTimes = resultIntervalsByValue[1]
    negTimes = resultIntervalsByValue[0]
    numPos = len(posTimes)
    numNeg = len(negTimes)
    numTotal = numPos + numNeg

    posTimesString = "No false positive periods"
    longestPosIntervalString = "No false positive periods"

    if len(posTimes) > 0:
        posTimesString = intervalsToString(posTimes)
        longestPosInterval = posTimes[0]
        longestPosIntervalLength = intervalLength(posTimes[0])
        for interval in posTimes:
            length = intervalLength(interval)
            if length > longestPosIntervalLength:
                longestPosIntervalLength = length
                longestPosInterval = interval

        longestPosIntervalString = formatTimeInterval(longestPosInterval)

    csvRow += [longestPosIntervalString, posTimesString, str(numPos), str(numNeg), str(numTotal)]       

def processResults(results, writer, csvRow):
    # analyze results
    # write actionable output to writer
    resultIntervals, resultIntervalsByValue = results[0], results[1]
    
    negativeIntervals = sorted(resultIntervalsByValue[0], key=intervalLength) 
    positiveIntervals = sorted(resultIntervalsByValue[1], key=intervalLength)

    negStats = getIntervalStats(negativeIntervals)
    posStats = getIntervalStats(positiveIntervals)

    negTime = negStats["totalTimeSpent"].total_seconds()
    posTime = posStats["totalTimeSpent"].total_seconds()
    totalTime = negTime + posTime

    negTimePercentage = negTime / totalTime if totalTime > 0 else 0
    posTimePercentage = posTime / totalTime if totalTime > 0 else 0

    stats = ["totalTimeSpent", "medianLength", "avgLength", "longestInterval", "shortestInterval"]

    csvRow.append(posTimePercentage)
    for stat in stats:
        val = posStats[stat]
        csvRow.append(formatTimeValue(val))
    
    csvRow.append(negTimePercentage)
    for stat in stats:
        val = negStats[stat]
        csvRow.append(formatTimeValue(val))
        

def getIntervalStats(intervals):
    stats = {}
    intervalLengths = [intervalLength(interval) for interval in intervals]
    print(intervalLengths)
    totalTimeSpent = datetime.timedelta(seconds=0)
    for interval in intervalLengths:
        totalTimeSpent += interval

    medianLength = "N/A"
    avgLength = "N/A"
    longestInterval = "N/A"
    shortestInterval = "N/A"

    if len(intervals) > 0:
        medianLength = intervalLength(intervals[len(intervals) // 2])
        avgLength = totalTimeSpent / len(intervalLengths)
        longestInterval = intervals[-1]
        shortestInterval = intervals[0]


    stats["totalTimeSpent"] = totalTimeSpent
    stats["medianLength"] = medianLength
    stats["avgLength"] = avgLength
    stats["longestInterval"] = longestInterval
    stats["shortestInterval"] = shortestInterval
    

    return stats

def getIntervalStatHeaders(classifier_name):
    headers = ["% Time Positive", "Total Time Positive", "Median Period Length", "Average Period Length",
               "Longest Positive Period", "Shortest Positive Period", "% Time Negatve", "Total Time Negative", "Median Period Length", "Average Period Length",
               "Longest Negative Period", "Shortest Negative Period",]

    classifier = " (" + classifier_name  + ")"
    return [header + classifier for header in headers]



def processAllClassifierResults(results, csvRow):
    conflicitingClassifications = findConflictingClassifications(results, False)
    # print "These classifications conflict"
    # print conflicitingClassifications
    if len(conflicitingClassifications) > 0:
        csvRow += [intervalsToString(conflicitingClassifications)]
    else:
        csvRow += ["No times when multiple classifiers output 1"]

    conflicitingClassificationsIncludingTheft = findConflictingClassifications(results, True)
    if len(conflicitingClassifications) > 0:
        csvRow += [intervalsToString(conflicitingClassificationsIncludingTheft)]
    else:
        csvRow += ["No times when multiple classifiers output 1"]


def findConflictingClassifications(results, includeTheft):
    conflictingVal = 1
    conflicitingClassifications = []
    for classifier in results:
        if includeTheft or classifier != classifiers.THEFT_CLASSIFIER:
            intervals = results[classifier][1][conflictingVal]
            conflicitingClassifications = findCommonIntervals(conflicitingClassifications, intervals)

    return conflicitingClassifications

def mergeAdjacentIntervalsByValue(intervals):
    i = 0
    while i + 1 < len(intervals):
        curr = intervals[i]
        next = intervals[i + 1]
        if curr[1] == next[1]:
            intervals[i] = ((curr[0][0], next[0][1]), curr[1])
            del intervals[i + 1]
        else:
            i += 1   

def mergeAdjacentIntervals(intervals):
    i = 0
    while i + 1 < len(intervals):
        curr = intervals[i]
        next = intervals[i + 1]
        if curr[1] == next[0]:
            intervals[i] = (curr[0], next[1])
            del intervals[i + 1]
        else:
            i += 1   


def filterSpikesFromIntervals(intervals, intervalsByValue):
    spikeLength = datetime.timedelta(seconds=1)
    i = 1
    indexAddedToIntervalsByValue = -1
    while i < len(intervals) - 1:
        interval, intervalBefore, intervalAfter = intervals[i], intervals[i - 1], intervals[i + 1]

        timeInterval = interval[0]

        if timeInterval[1] - timeInterval[0] <= spikeLength:
            newTimeInterval = (intervalBefore[0][0], intervalAfter[0][1])
            intervals[i - 1] = (newTimeInterval, intervalBefore[1])
            del intervals[i:i+2]
        else:
            timeIntervalBefore = intervalBefore[0]
            classification = intervalBefore[1]
            intervalsByValue[classification].append(timeIntervalBefore)
            indexAddedToIntervalsByValue = i - 1
            i += 1

    for j in range(indexAddedToIntervalsByValue + 1, len(intervals)):
        interval = intervals[j]
        timeInterval = interval[0]
        classification = interval[1]
        intervalsByValue[classification].append(timeInterval)


def findCommonIntervalsByValue(intervals1, intervals2, value):
    # print("Finding common intervals!")
    # print intervals1
    # print intervals2

    if len(intervals1) == 0 and len(intervals2) == 0:
        return []
    if len(intervals1) == 0:
        return intervals2
    if len(intervals2) == 0:
        return intervals1 

    def advance(intervals, i, value):
        while i < len(intervals) and intervals[i][1] != value:
            # print(i)
            i += 1
        return i 

    i1 = advance(intervals1, 0, value) 
    i2 = advance(intervals2, 0, value)
    # print i1, i2 
    
    commonIntervals = []
    while i1 < len(intervals1) and i2 < len(intervals2):
        interval1 = intervals1[i1][0]
        interval2 = intervals2[i2][0]
        # # print(i1, i2)
        laterStartingInterval, earlierStartingInterval = None, None
        later_i, earlier_i = None, None

        if interval1[0] >= interval2[0]:
            laterStartingInterval, earlierStartingInterval = interval1, interval2
            later_i, earlier_i = i1, i2
        else:
            laterStartingInterval, earlierStartingInterval = interval2, interval1
            later_i, earlier_i = i2, i1

        if laterStartingInterval[0] >= earlierStartingInterval[1]:
            if earlier_i == i1:
                i1 = advance(intervals1, i1, value)
            else:
                i2 = advance(intervals2, i2, value)
        
        else:
            earlierEndingInterval = earlierStartingInterval if earlierStartingInterval[1] <= laterStartingInterval[1] else laterStartingInterval

            commonIntervals.append((laterStartingInterval[0], earlierEndingInterval[1]))
            # print commonIntervals

            if earlierStartingInterval[1] == laterStartingInterval[1]:
                # print "End times are equal"
                i1 = advance(intervals1, i1, value)
                i2 = advance(intervals2, i2, value)

            elif earlierStartingInterval[1] < laterStartingInterval[1]:
                # print "Early start ends earlier, advance early"
                if earlier_i == i1:
                    i1 = advance(intervals1, i1, value)
                else:
                    i2 = advance(intervals2, i2, value)
                # print i1, i2
            else:
                # print "Early start ends later, advance later"
                if later_i == i1:
                    i1 = advance(intervals1, i1, value)
                else:
                    i2 = advance(intervals2, i2, value)
                # print i1, i2

    return commonIntervals

def findCommonIntervals(intervals1, intervals2):
    # print("Finding common intervals!")
    # print intervals1
    # print intervals2

    if len(intervals1) == 0 and len(intervals2) == 0:
        return []
    if len(intervals1) == 0:
        return []
    if len(intervals2) == 0:
        return []

    i1 = 0
    i2 = 0
    # print "Starting"
    # print i1, i2 
    
    commonIntervals = []
    while i1 < len(intervals1) and i2 < len(intervals2):
        interval1 = intervals1[i1]
        interval2 = intervals2[i2]
        # # print(i1, i2)
        laterStartingInterval, earlierStartingInterval = None, None
        later_i, earlier_i = None, None

        if interval1[0] >= interval2[0]:
            laterStartingInterval, earlierStartingInterval = interval1, interval2
            later_i, earlier_i = i1, i2
        else:
            laterStartingInterval, earlierStartingInterval = interval2, interval1
            later_i, earlier_i = i2, i1

        if laterStartingInterval[0] >= earlierStartingInterval[1]:
            # print("GOODBYE")
            if earlier_i == i1:
                i1 += 1
            else:
                i2 += 1
        
        else:
            # print("HELLO")
            earlierEndingInterval = earlierStartingInterval if earlierStartingInterval[1] <= laterStartingInterval[1] else laterStartingInterval

            commonIntervals.append((laterStartingInterval[0], earlierEndingInterval[1]))
            # print commonIntervals

            if earlierStartingInterval[1] == laterStartingInterval[1]:
                # print "End times are equal"
                i1 += 1
                i2 += 1

            elif earlierStartingInterval[1] < laterStartingInterval[1]:
                # print "Early start ends earlier, advance early"
                if earlier_i == i1:
                    i1 += 1
                else:
                    i2 += 1
                # print i1, i2
            else:
                # print "Early start ends later, advance later"
                if later_i == i1:
                    i1 += 1
                else:
                    i2 += 1
                # print i1, i2

    return commonIntervals



def plotIntervals(intervals):
    times = []
    values = []

    for interval in intervals:
        time = interval[0]
        times.append(time[0])
        times.append(time[1])
        values.append(interval[1])
        values.append(interval[1])

    times = date2num(times)

    seconds = SecondLocator()   # every year
    minutes = MinuteLocator()  # every month
    hours = HourLocator()
    hoursFmt = DateFormatter('%H:%M')
    minutesFmt = DateFormatter('%H:%M:%S')

    fig, ax = plt.subplots()
    ax.plot_date(times, values, '-')

    # format the ticks
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(hoursFmt)
    ax.xaxis.set_minor_locator(minutes)
    ax.autoscale_view()


    # format the coords message box
    ax.fmt_xdata = DateFormatter('%H:%M')
    ax.grid(True)

    axes = plt.gca()
    axes.set_ylim([-0.25, 1.25])

    fig.autofmt_xdate()
    plt.show()


def intervalLength(interval):
    return interval[1] - interval[0]


def classifierPolicy(classifiedWindow):
    averagedClassifications = []
    for c, labels in classifiedWindow.items():
        positives = labels.count(1)
        negatives = labels.count(0)
        if positives > negatives:
            averagedClassifications.append(c)
    if len(averagedClassifications) == 1:
        return averagedClassifications[0].getName()
    #use policy (most dangerous) among conflicting classifications
    c = classifiers.CLASSIFIERS
    if c[classifiers.TABLE_CLASSIFIER] in averagedClassifications:
        return classifiers.TABLE_CLASSIFIER
    elif c[classifiers.STEADY_BAG_CLASSIFIER] in averagedClassifications:
        return classifiers.POCKET_BAG_CLASSIFIER
    elif c[classifiers.POCKET_BAG_CLASSIFIER] in averagedClassifications:
        return classifiers.POCKET_BAG_CLASSIFIER
    elif c[classifiers.HAND_CLASSIFIER] in averagedClassifications:
        return classifiers.HAND_CLASSIFIER
    else: 
        return "Unknown"

###### Utilities #######

def filesToTimesToFilesDict(files, userID, instrument):
    timesToFiles = {}
    for f in files:
        time = getTimeFromFile(f, userID, instrument, True)
        timesToFiles[time] = f 
    return timesToFiles

def timeStringToDateTime(timestring):
    return datetime.datetime.strptime(timestring, '%Y_%m_%d_%H_%M_%S')

def timeStringsToDateTimes(timeStrings):
    return [timeStringToDateTime(timeString) for timeString in timeStrings]

def formatTime(dateTime, withDate=False):
    if withDate:
        return dateTime.strftime('%b %d|%H:%M:%S')
    return dateTime.strftime('%H:%M:%S')

def formatTimeDelta(timeDelta):
    totalSeconds = timeDelta.total_seconds()
    return formatTotalSeconds(totalSeconds)

def formatTotalSeconds(totalSeconds):
    hours = totalSeconds // 3600
    minutes = (totalSeconds % 3600) // 60
    seconds = totalSeconds % 60
    return str(hours) + 'h:' + str(minutes) + 'm:' + str(seconds) + 's' 

def formatTimeInterval(timeInterval, withDate=False):
    if withDate:
        return '(' + formatTime(timeInterval[0], withDate=True) + '--' + formatTime(timeInterval[1], withDate=True) + ')'
    else:
        return '(' + formatTime(timeInterval[0]) + '--' + formatTime(timeInterval[1]) + ')' 

def formatTimeValue(timeValue):
    if type(timeValue) is str:
        return timeValue 
    if type(timeValue) is datetime.datetime:
        return formatTime(timeValue)
    elif type(timeValue) is datetime.timedelta:
        return formatTimeDelta(timeValue)
    else:
        # must be an interval
        return formatTimeInterval(timeValue)


def getTimeFromFile(filename, userID, instrument):
    query = DIRECTORY + 'AppMon' + '_' + userID + '.*_' + instrument + '_' + \
        '(?P<time>.*)' + '_.csv'
    match = re.match(query, filename)
    return match.group('time')

def getTimeFromFile(filename):
    query = DIRECTORY + 'AppMon' + '_*_' + '*_' + \
        '(?P<time>.*)' + '_.csv'
    match = re.match(query, filename)
    time = match.group('time')[-19:]
    
    return time

def getFileExtension(isDecrypted):
    if isDecrypted:
        return '_.csv'
    else:
        return '_.zip.encrypted'

def removePath(filename):
    i = -1
    while i >= -1 * len(filename) and filename[i] != '/':
        i -= 1

    return filename[i + 1:]

def removeFilesFromDir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def replaceCommasWithSemicolons(string):
    return string.replace(",", ";")

def intervalsToString(intervals):
    resultsString = ""
    for interval in intervals:
        intervalString = "(" + formatTime(interval[0]) + "--" + formatTime(interval[1]) + "); "
        resultsString += intervalString
    return resultsString 
 
def convertToDateTime(timestring, bootTime):
    epochTimeString = float(timestring) / 1000.0
    timeAsEpochTime = datetime.date.fromtimestamp(epochTimeString)
    isAbsoluteTimeStamp = timeAsEpochTime > YEAR_2000
    if isAbsoluteTimeStamp:
        return timeAsEpochTime
    else:
        return bootTime + datetime.timedelta(milliseconds=int(timestring))

def getWindowStartTime(windowOfDataRows):
    return windowOfDataRows[0][0] #time of first data row in window

def compileRelevantSensors():
    for c in classifiers.CLASSIFIERS:
        classifier = classifiers.CLASSIFIERS[c]
        for sensor in classifier.getRelevantSensors():
            RELEVANT_SENSORS.add(sensor)

def calculateBootTime(userFiles):
    return 


def getUserFileTimes(userFiles):
    return [timeStringToDateTime(getTimeFromFile(filename)) in filename in userFiles]



def main():

    now = datetime.datetime.now()
    dashboardFileName = DASHBOARDDIR + "Dashboard-" + now.strftime('%Y_%m_%d_%H_%M') + ".csv"
    resultsFileName = DASHBOARDDIR + "Dashboard-" + now.strftime('%Y_%m_%d_%H_%M') + ".txt"

    dashboardFile = open(dashboardFileName, 'wb')
    dashboardWriter = csv.writer(dashboardFile, delimiter = ',')

    resultsFile = open(resultsFileName, 'wb')

    columnHeaders = ["User ID"]

    columnHeaders += ["Longest False Positive Time Period", "False Positive Theft Times", "# of Theft Positives", "# of Theft Negatives", "Total Theft Classifications"]

    for c in classifiers.CLASSIFIERS:
        # print c
        if c != classifiers.THEFT_CLASSIFIER:
            columnHeaders += getIntervalStatHeaders(c)

    columnHeaders += ["Periods when multiple (non-theft) classifiers were both positive", "Periods when multiple classifiers were both positive"]

    dashboardWriter.writerow(columnHeaders)

    # compileRelevantSensors()

    for userID in USERS:
        datarow = [userID]
        runClassifiersOnUser(userID, dashboardWriter, resultsFile)
        #runClassifiersOnUser(userID, tempResultsFile)

    dashboardFile.close()
    resultsFile.close()
    # print("Dashboard results generated in: " + dashboardFileName)

if __name__ == '__main__':
    # main()
    # USER_ID = '6fdda897'

    NOW = datetime.datetime.now()
    NOW_TIME = NOW.strftime('_%m_%d_%H_%M')
    DIRECTORY_PATH = DIRECTORY

    DATA_DAY = 'FULL_STUDY_RUN'
    file = open('testing-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
    watchFile = open('watch-testing-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
    results = open('testing-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
    watchResults = open('watch-testing-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
    resultsSummary = open('testing-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
    watchSummary = open('watch-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
    resultsSummaryWriter = csv.writer(resultsSummary)
    watchSummaryWriter = csv.writer(watchSummary)
    resultsSummaryWriter.writerow(["Day","User", "Classifier", "Percentage of Time"])
    watchSummaryWriter.writerow(["Day", "User", "State", "Percentage of Time", "Hours", "Total Hours"])

    for DATA_DAY in DATA_DATES:
        print("DIRECTORY started as:", DIRECTORY)
        DIRECTORY = DIRECTORY_PATH + DATA_DAY + "/"
        print("DIRECTORY now:", DIRECTORY)
        if not FULL_STUDY_RUN:
            file = open('testing-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
            watchFile = open('watch-testing-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
            results = open('testing-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
            watchResults = open('watch-testing-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
            resultsSummary = open('testing-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
            watchSummary = open('watch-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
            resultsSummaryWriter = csv.writer(resultsSummary)
            watchSummaryWriter = csv.writer(watchSummary)
            resultsSummaryWriter.writerow(["Day", "User", "Classifier", "Percentage of Time"])
            watchSummaryWriter.writerow(["Day", "User", "State", "Percentage of Time", "Hours", "Total Hours"])

        count = 0
        for USER_ID in USERS:
            count += 1
            print("Number of users processed:", count)
            print("Currently on:", USER_ID)
            
            activatedIntervalsWatch = None
            activatedIntervalsPhone = None

            if not RUN_CLASSIFIERS_ONLY:
                watchResults.write("#########" + USER_ID + "#######\n")
                try:
                    watchState = stateFromWatchData(continuousWatchInterals(USER_ID), watchFile)
                    activatedWatchStates = watchActivationStates(watchState)
                    activatedIntervalsWatch = {PossessionState.PHONE_ACTIVATED: activatedWatchStates[0], PossessionState.PHONE_DEACTIVATED: activatedWatchStates[1]}
                    # HERE STEVEN :)
                    timeSpentByWatchState = {}
                    # print(watchState)
                    for state in watchState:
                        watchResults.write("----" + str(state) + "-----" + "\n")
                        intervals = watchState[state]
                        stats = getIntervalStats(intervals)
                        for stat, val in stats.items():
                            watchResults.write(str(stat) + "\t\t\t" + str(formatTimeValue(val)) + "\n")
                            if stat == "totalTimeSpent":
                                timeSpentByWatchState[state] = val.total_seconds()

                    totalTime = 0
                    for c, time in timeSpentByWatchState.items():
                        totalTime += time

                    watchResults.write("-----Percentage of Time for each State ------" + "\n")
                    for c, time in timeSpentByWatchState.items():
                        percentage = time / totalTime
                        percentageString = str(c) + "\t\t\t\t" + str(percentage * 100) + "%\n"
                        watchResults.write(percentageString)
                        percentageRow = [DATA_DAY, USER_ID, str(c), str(percentage * 100)]
                        watchSummaryWriter.writerow(percentageRow)
                except:
                    tb = traceback.format_exc()
                    print(tb)
                    watchResults.write("******EXCEPTION (while computing watch state)*******\n")
                    watchResults.write(tb)
                    watchResults.write("\n")
            
            if not RUN_WATCH_ONLY:
                results.write("#########" + USER_ID + "#######\n")
                try: 
                    classifications, intervalsByClass, possessionState = runClassifiersOnUser(USER_ID, None, file)
                    activatedIntervalsPhone = possessionState.getIntervalsByState()
                    # print("ACTIVATED REPORT:")
                    # for interval in possessionState.getIntervals():
                    #     print(interval)
                    # print("END ACTIVATED REPORT")
                    timeSpentByClass = {}

                    for c in intervalsByClass:
                        results.write("----" + str(c) + "-----\n")
                        intervals = intervalsByClass[c]
                        stats = getIntervalStats(intervals)
                        for stat, value in stats.items():
                            results.write(str(stat) + "\t\t\t" + str(value) + "\n")
                            if stat == "totalTimeSpent":
                                timeSpentByClass[c] = value.total_seconds()

                    totalTime = 0
                    for c, time in timeSpentByClass.items():
                        totalTime += time

                    results.write("-----Percentage of Time for each classifier------\n")
                    for c, time in timeSpentByClass.items():
                        percentage = time / totalTime
                        results.write(str(c) + "\t\t\t\t" + str(percentage * 100) + "%\n")
                        # resultsSummary.write(str(c) + "\t\t\t\t" + str(percentage * 100) + "%\n")
                        resultsSummaryWriter.writerow([DATA_DAY, USER_ID, str(c), str(percentage * 100), formatTotalSeconds(time), formatTotalSeconds(totalTime)])

                    results.write("-----Classifications over Time-------\n")
                    for c in classifications:
                        interval = c[0]
                        duration = formatTimeValue(interval[1] - interval[0])
                        classification = c[1]
                        intervalString = "(" + formatTime(interval[0]) + "--" + formatTime(interval[1]) + "); "
                        results.write(intervalString + ' ' + duration + '; ' + str(classification) + "\n")
                except:
                    tb = traceback.format_exc()
                    print(tb)
                    results.write("******EXCEPTION (while computing classifications)*******\n")
                    results.write(tb)
                    results.write("\n")

            activatedFile = open('activated-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
            if activatedIntervalsWatch == None or activatedIntervalsPhone == None:
                activatedFile.write("Check: " + 'watch-testing-results-' + DATA_DAY + NOW_TIME + '.txt')

            else:
                bothActivated = findCommonIntervals(activatedIntervalsPhone["activated"], activatedIntervalsWatch["activated"])
                bothDeactivated = findCommonIntervals(activatedIntervalsPhone["deactivated"], activatedIntervalsWatch["deactivated"])
                onlyPhoneActivated = findCommonIntervals(activatedIntervalsPhone["activated"], activatedIntervalsWatch["deactivated"])
                onlyWatchActivated = findCommonIntervals(activatedIntervalsPhone["deactivated"], activatedIntervalsWatch["activated"])
                
                ### CALCULATE UNLOCKS ###

                unlockData = possessionState.unlockData
                activatedIntervals = activatedIntervalsPhone["activated"]
                pickle.dump(unlockData, open( "unlock_test_data.pkl", "wb" ) )
                pickle.dump(activatedIntervals, open( "unlock_test_intervals.pkl", "wb" ) )

                numUnlocksSaved, numUnlocksTotal, unlockTimes = computeUnlocks(unlockData, activatedIntervals)
                print("UNLOCK DATA:", str(numUnlocksSaved), str(numUnlocksTotal), str(unlockTimes))
                ##########################

                totalActivatedTestTimes = 0
                stateTimes = {}
                for stateP in activatedIntervalsPhone:
                    for stateW in activatedIntervalsWatch:
                        state = "Phone: " + stateP + " Watch: " + stateW
                        print(state)
                        activatedFile.write(str(state) + '\n')
                        # print("Phone Intervals:", activatedIntervalsPhone[stateP])
                        # print("Watch Intervals:", activatedIntervalsWatch[stateW])
                        commonIntervals = findCommonIntervals(activatedIntervalsPhone[stateP], activatedIntervalsWatch[stateW])
                        # print("COMMON INTERVALS:", commonIntervals)
                        stats = getIntervalStats(commonIntervals)
                        print(stats["totalTimeSpent"])
                        activatedFile.write(str(stats["totalTimeSpent"]) + '\n')

                        timeSeconds = stats["totalTimeSpent"].total_seconds()
                        totalActivatedTestTimes += timeSeconds
                        stateTimes[state] = timeSeconds


                print("ACTIVATION PERCENTAGES")
                activatedFile.write("ACTIVATION PERCENTAGES\n")
                for state, time in stateTimes.items():
                    percentage = time / totalActivatedTestTimes if totalActivatedTestTimes > 0 else 0
                    print(state, "-->", percentage)
                    activatedFile.write(str(state) + " --> " + str(percentage * 100) + '\n')

                for stateP, intervals in activatedIntervalsPhone.items():
                    activatedFile.write(str(stateP) + "\n")
                    activatedFile.write(str([formatTimeInterval(interval) for interval in intervals]) + "\n")
                
                for stateW in activatedIntervalsWatch:
                    activatedFile.write(str(stateW) + "\n")
                    activatedFile.write(str([formatTimeInterval(interval) for interval in intervals]) + "\n")    

        if not FULL_STUDY_RUN:
            file.close()
            watchFile.close()
            results.close()
            watchResults.close()
    print("Yay I finished!")

