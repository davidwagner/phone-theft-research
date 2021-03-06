import csv
import os
import glob
import re
import datetime
import sys
import shutil
import argparse
import time as TIMER
# import matplotlib.pyplot as plt
# from matplotlib.dates import SecondLocator, MinuteLocator, HourLocator, DateFormatter, date2num
# import classifier
import Sensors as sensors
import Classifiers as classifiers 
import PossessionState
import pickle
import argparse
import traceback
import math

from configsettings import *
from collections import deque, Counter, OrderedDict
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

# START_TIME_FILTER = datetime.time(hour=8)
# END_TIME_FILTER = datetime.time(hour=22)
RESULTS_DIRECTORY = './' + 'RESULTS/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
START_TIME_FILTER = None
END_TIME_FILTER = None

# SAFE_PERIOD = 3

SAFE_PERIOD = 3
USE_CACHED_DATA = False

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
                    if (START_TIME_FILTER == None or row[0].time() >= START_TIME_FILTER):
                        if (END_TIME_FILTER == None or row[0].time() < END_TIME_FILTER):
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

def continuousWatchInterals(userID, userData={}):
    if len(userData) == 0:
        for instrument in WATCH_SENSORS:
            dataFiles = getUserFilesByDayAndInstrument(userID, instrument)
            # print "Heart Rate Files"
            # print dataFiles
            userData[instrument] = dataFilesToDataListAbsTime(dataFiles)
    watchData = userData
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

        if prevTime != -1 and startTime != -1 and prevTime != -1:
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
    noHeartIntervals = []
    prevTime = -1
    for start, end, state in heartRateIntervals:
        if prevTime == -1:
            prevTime = end
        elif start > prevTime:
            noHeartIntervals.append((prevTime, start))
        prevTime = end

    while i < len(noHeartIntervals) and j < len(basisPeakIntervals):
        hInterval = noHeartIntervals[i]
        bInterval = basisPeakIntervals[j]
        hStart, hEnd = hInterval
        bStart, bEnd = bInterval

        if hEnd < bStart:
            allIntervals.append((hStart, hEnd, "phoneFar"))
            i += 1
        elif bEnd < hStart:
            j += 1
        elif hStart < bStart:
            allIntervals.append((hStart, bStart, "phoneFar"))
            if bEnd < hEnd:
                allIntervals.append((bStart, bEnd, "unknown"))
                j += 1
                noHeartIntervals[i] = (bEnd, hEnd)
            else: 
                allIntervals.append((bStart, hEnd, "unknown"))
                i += 1
                basisPeakIntervals[j] = (hEnd, bEnd)
        else:
            bStart = hStart
            if hEnd < bEnd:
                allIntervals.append((bStart, hEnd, "unknown"))
                i += 1
                basisPeakIntervals[j] = (hEnd, bEnd)
            else:
                allIntervals.append((bStart, bEnd, "unknown"))
                j += 1 
                noHeartIntervals[i] = (bEnd, hEnd)

    while i < len(noHeartIntervals):
        hInterval = noHeartIntervals[i]
        hStart, hEnd = hInterval
        allIntervals.append((hStart, hEnd, "phoneFar"))
        i += 1

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
    return result, allIntervals

def watchActivationStates(watchStates):
    activated = []
    deactivated = []
    delta = datetime.timedelta(minutes=3)
    for start, end, state in watchStates:
        if state == "phoneNear":
            activated.append((start, end))
        else:
            if end - start > delta:
                deactivated.append((start, end))
            else:
                activated.append((start, end))
    # if "phoneNear" in watchStates:
    #     activated = watchStates["phoneNear"]
    # if "unknown" in watchStates:
    #     deactivated += watchStates["unknown"]
    # if "phoneFar" in watchStates:
    #     deactivated += watchStates["phoneFar"]
    
    activated = sorted(activated, key=lambda x: x[0])
    deactivated = sorted(deactivated, key=lambda x: x[0])
    # print("DEACTIVATED WATCH:", deactivated)
    mergeAdjacentIntervals(deactivated)
    mergeAdjacentIntervals(activated)
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



def runClassifiersOnUser(userID, csvWriter, resultsFile, userData={}):
    if DUMP_RESULTS:
        resultsFile.write("###########################\n")
        resultsFile.write(str(userID) + '\n')
        resultsFile.write("###########################\n")
    # print(userID)
    if len(userData) == 0:
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

    for c in classifiers.CLASSIFIERS:
        # print(c)
        intervalsByClass[c] = []
    intervalsByClass["Unknown"] = []

    limit = numRows // maxWindowSize * maxWindowSize
    # print("LIMIT", limit)

    print("########")
    print("SAFE PERIOD:", SAFE_PERIOD)
    possessionState = PossessionState.PossessionState(userData, userData[sensors.PHONE_ACTIVE_SENSORS], userData[sensors.KEYGUARD], SMOOTHING_NUM, safe_period=SAFE_PERIOD)
    aggregateClassifierResults = []
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
            # aggregateClassifierResults[c].append(((windowStartTime, windowStartTime + datetime.timedelta(seconds=1)), results))
            classifier = classifiers.CLASSIFIERS[c]
            results = runClassifier(classifier, windowOfData)
            classifierResults[classifier] = results
            # if i % 50000 == 0 and c ==classifiers.HAND_CLASSIFIER:
            #     print("TYPE: ", type(results))


        logString = windowStartTime.strftime("%H:%M:%S") + "| " 
        aggregateWindow = {}
        for c in classifierResults:
            aggregateWindow[c.getName()] = classifierResults[c]
        aggregateClassifierResults.append(((windowStartTime, windowStartTime + datetime.timedelta(seconds = 1)), aggregateWindow))
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


    classifications.append((currentInterval, currentClass))
    intervalsByClass[currentClass].append(currentInterval)

    return classifications, intervalsByClass, possessionState, aggregateClassifierResults


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
    # print(intervalLengths)
    totalTimeSpent = datetime.timedelta(seconds=0)
    for interval in intervalLengths:
        if type(interval) is int:
            continue
        totalTimeSpent += interval

    medianLength = "N/A"
    avgLength = "N/A"
    longestInterval = "N/A"
    shortestInterval = "N/A"

    if totalTimeSpent.total_seconds() < 0:
        print("WTF!!!")
        for interval in intervals:
            print(formatTimeInterval(interval))

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

def compareIntervals(intervals1, intervals2):
    i1 = 0
    i2 = 0
    interval1 = intervals1[i1][0]
    interval2 = intervals2[i2][0]
    class1 = intervals1[i1][1]
    class2 = intervals1[i2][1]

    startTime = interval1[0] if interval1[0] > interval2[0] else interval2[0]
    endTime = None

    comparedIntervals = []
    matchingIntervals = []
    conflictingIntervals = []
    while i1 < len(intervals1) and i2 < len(intervals2):
        interval1 = intervals1[i1][0]
        interval2 = intervals2[i2][0]
        class1 = intervals1[i1][1]
        class2 = intervals2[i2][1]

        if interval1[1] == interval2[1]:
            endTime = interval1[1]
            i1 += 1
            i2 += 1
        elif interval1[1] < interval2[1]:
            endTime = interval1[1]
            i1 += 1
        else:
            endTime = interval2[1]
            i2 += 1

        comparedClass = None
        matchingClasses = False
        if class1 == class2:
            comparedClass = class1
            matchingClasses = True
        else:
            comparedClass = str(class1) + " | " + str(class2)

        comparedInterval = ((startTime, endTime), comparedClass, matchingClasses)
        comparedIntervals.append(comparedInterval)

        if matchingClasses:
            matchingIntervals.append(comparedInterval)
        else:
            conflictingIntervals.append(comparedInterval)

        startTime = endTime

    return comparedIntervals, matchingIntervals, conflictingIntervals


def totalTimeOfIntervals(intervals):
    timeConnected = datetime.timedelta(seconds=00)
    prevState = -1
    for interval, classified, state in intervals:
        start = interval[0]
        end = interval[1]
        timeInBetween = end - start
        timeConnected += timeInBetween
        prevState = end

    return timeConnected


def getExpectedIntervals(file):
    intervals = []
    with open(file) as f:  
        reader = csv.reader(f)
        prevTime = -1
        prevState = -1
        for row in reader: 
            startTime = datetime.datetime.strptime(row[0] + " " + row[1], "%m/%d/%y %H:%M")
            if prevTime != -1:
                intervals.append(((prevTime, startTime), prevState))
            prevTime = startTime
            prevState = row[2]
    return intervals



### Joanna Finish #####
# actualIntervals is a list of intervals, classifications like [((startTime, endTime), "table"), ((start, end), "pocket")]
def checkClassifications(actualIntervals, DATA_DAY, NOW_TIME, expectedIntervals=None, expectedNoSteady = None, normalIntervals=None):
    # However you want to load the expectedIntervals, maybe parse a text file?
    # Just make sure to load them as a list with each item formatted as ((startDateTime, endDateTime), classification)

    classificationIntervals = {}
    noSteadyState = combineSteadyState(actualIntervals)
    # print(noSteadyState)
    accuracyLengths = classificationAccWithoutPolicy(actualIntervals, expectedIntervals)
    noSteadyAccuracyLenghts = classificationAccWithoutPolicy(noSteadyState, expectedNoSteady)

    # print(accuracyLengths)
    comparedIntervals, matchingIntervals, conflictingIntervals = compareIntervals(normalIntervals, expectedIntervals)
    policyNoSteady = filterSteadyState(comparedIntervals);
    # print("STEADY STATE FILTERING: " );
    # print(intervalsWithoutSteady);
    classificationAccuracy = getClassificationAccuracy(comparedIntervals)
    # print(classificationAccuracy)
    print("Policy Confusion Matrix with Steady State")
    # print(classifierConfusionMatrix(classificationAccuracy))
    noSteadyAccuracy = getClassificationAccuracy(policyNoSteady)
    print("Policy Confusion Matrix with NO Steady State")
    # print(classifierConfusionMatrix(noSteadyAccuracy))

    for c in classifiers.CLASSIFIERS:
        # print(actualIntervals[c])
        # if c != classifiers.BACKPACK_CLASSIFIER:
        #     continue
        print(c)


        # comparedIntervals, matchingIntervals, conflictingIntervals = classificationAccWithoutPolicy(actualIntervals[c], expectedIntervals, c)
        # classificationIntervals[c] = (comparedIntervals, matchingIntervals, conflictingIntervals)
        # accuracyLengths = classificationAccWithoutPolicy(actualIntervals, expectedIntervals, c)


        # print(accuracyLengths)

        # print(c)
        # print(matchingIntervals)
        # return

        # file = open(RESULTS_DIRECTORY + '/' + 'diary-study-stats-' + DATA_DAY + NOW_TIME + '_' + c +'.txt', 'w+')

        # file.write("############ DIARY STUDY COMPARISON ############## \n")

        # interval1, classification1, ismatch1 = comparedIntervals[0]
        # interval2, classification2, ismatch2 = comparedIntervals[-1]
        # totalTime = interval2[1] - interval1[0]
        # matchingTime = totalTimeOfIntervals(matchingIntervals)
        # conflictingTime = totalTimeOfIntervals(conflictingIntervals)
        # print(comparedIntervals)
        # return

        # print("BEFORE FILTERING");
        # print(comparedIntervals);
        # comparedIntervals = filterSteadyState(comparedIntervals);
        # print("STEADY STATE FILTERING: " );
        # print(intervalsWithoutSteady);
        # classificationAccuracy = getClassificationAccuracy(comparedIntervals, c);
        # print(classificationAccuracy)
        ccm = classifierConfusionMatrix(accuracyLengths[c])
        print("Classifier Confusion matrix with Steady State")
        print(ccm)
        ccm_no = classifierConfusionMatrix(noSteadyAccuracyLenghts[c])
        print("Classifier Confusion Matrix with NO Steady State")
        print(ccm_no)

        # accuracyNoSteady = classificationAccWithoutPolicy(noSteadyState[c], expectedIntervals, c)
        # ccm_no = classifierConfusionMatrix(accuracyNoSteady)
        # print("Confusion matrix with Steady State")
        # print(ccm_no)
        # return


        # file.write("Total Time: " + formatTimeValue(totalTime) + "\n")
        # file.write("Total time matching: " + formatTimeValue(matchingTime) +"\n")
        # file.write("% of time matched: " + str(1.0 * matchingTime/totalTime) + "\n")
        # file.write("Total time conflicting: " + formatTimeValue(conflictingTime) + "\n")
        # file.write("% of time conflicted: " + str(1.0 * conflictingTime/totalTime) + "\n")

        # file.write("\n")
        # file.write("\n")
        # filterConflicting = getConflictingIntervals(intervalsWithoutSteady)

        # file.write("All conflicting intervals: \n")
        # for interval, classificationString, isMatching in comparedIntervals:
        #     file.write(formatTimeValue(interval, withDate=True) + ": " + classificationString + "\n")

        # file.write("Conflicting intervals by length:\n")
        # for interval, classificationString, isMatching in sorted(comparedIntervals, key=lambda x: intervalLength(x[0]), reverse=True):
        #     file.write(formatTimeValue(interval, withDate=True) + ": " + formatTimeValue(intervalLength(interval))[:15] + ": " + classificationString + "\n")

        # file.close()
    for c in classifiers.CLASSIFIERS:
        print(c)
        if c not in classificationAccuracy:
            continue
        print("Policy Confusion Matrix with Steady State")
        print(classifierConfusionMatrix(classificationAccuracy[c]))
        noSteadyAccuracy = getClassificationAccuracy(policyNoSteady)
        print("Policy Confusion Matrix with NO Steady State")
        print(classifierConfusionMatrix(noSteadyAccuracy[c]))

    # Write the results to some file, probably also calculate some stats on what % of time we match/don't match
    # All of comparedIntervals, matchingIntervals, and conflictingIntervals have the following format:
    # ((startDateTime, endDateTime), classificationString, isMatchingClassifications)
    # the classificationString is either one classifier if the expected/actual matched, else two classifier names

# classifier = [true positive, false positive, false negtive, total time]

def classificationAccWithoutPolicy(intervals1, intervals2):
    i1 = 0
    i2 = 0
    
    interval1 = intervals1[i1][0]
    interval2 = intervals2[i2][0]
    class2 = intervals1[i2][1]
    classifierToVal = intervals1[i1][1]

    startTime = interval1[0] if interval1[0] > interval2[0] else interval2[0]
    endTime = None

    comparedIntervals = []
    matchingIntervals = []
    conflictingIntervals = []
    accuracyLengths = {}
    while i1 < len(intervals1) and i2 < len(intervals2):
        interval1 = intervals1[i1][0]
        interval2 = intervals2[i2][0]
        class2 = intervals2[i2][1]
        classifierToVal = intervals1[i1][1]

        if interval1[1] == interval2[1]:
            endTime = interval1[1]
            i1 += 1
            i2 += 1
        elif interval1[1] < interval2[1]:
            endTime = interval1[1]
            i1 += 1
        else:
            endTime = interval2[1]
            i2 += 1

        comparedClass = None
        # matchingClasses = False
        length = intervalLength((startTime, endTime))
        for c in classifierToVal:
            # print(c)
            classification = classifierToVal[c]
            if c not in accuracyLengths:
                accuracyLengths[c] = [datetime.timedelta(seconds=0)] * 4
            if c == class2:
                if sum(classification) * 1.0 / len(classification) >= 0.5:
                    accuracyLengths[c][0] += length
                    # print("TRUE POSITIVES")
                    # print(class1)
                    # print(class2)
                    # print(classification)
                    # print("\n")
                else:
                    # if class2 == classifiers.HAND_CLASSIFIER:

                    #     print("FALSE NEGATIVES")
                    #     # print(class1)
                    #     print(class2)
                    #     print(classifierToVal)
                    #     print((startTime, endTime))

                    #     print("\n")


                    accuracyLengths[c][2] += length
                    #false negatives
            else:
                if sum(classification) * 1.0 / len(classification) > 0.5:
                 #false positive
                    # print("FALSE POSITIVE")
                    # print(class1)
                    # print(class2)
                    # print(classification)
                    # print(sum(classification) * 1.0 / len(classification))
                    # print((startTime, endTime))
                    # print("\n")

                    accuracyLengths[c][1] += length

                else :
                    # print("TRUE NEGATIVES")
                    # print(classification)
                    # print(class1)
                    # print(class2)
                    # print((startTime, endTime))
                    # print("\n")
                    accuracyLengths[c][3] += length



        startTime = endTime
    return accuracyLengths


def combineSteadyState(actualIntervals):
    results = []
    i = 0

    interval1 = actualIntervals[i][0]
    classifierToVal = actualIntervals[i][1]

    prev = {}
    while i < 5:
        interval1 = actualIntervals[i][0]
        classifierToVal = actualIntervals[i][1]
        for c in classifiers.CLASSIFIERS:
            if c not in prev:
                prev[c] = [0]* 5
            prev[c][i%5] = sum(classifierToVal[c]) * 1.0 / len(classifierToVal[c])
        results.append((interval1, classifierToVal))
        i+=1

    while i < len(actualIntervals):
        interval1 = actualIntervals[i][0]
        classifierToVal = actualIntervals[i][1]
        prevVal = 0
        prevClass = classifiers.TABLE_CLASSIFIER
        for c in prev:
            if sum(prev[c]) > prevVal:
                prevClass=c
                prevVal = sum(prev[c])
            prev[c][i%5] = sum(classifierToVal[c]) * 1.0 / len(classifierToVal[c])
        condition = sum(classifierToVal[prevClass]) * 1.0 / len(classifierToVal[prevClass]) < 0.5 and sum(classifierToVal[classifiers.STEADY_STATE_CLASSIFIER]) * 1.0 / len(classifierToVal[classifiers.STEADY_STATE_CLASSIFIER]) >= 0.5
        if  condition:
            classifierToVal[prevClass] = [1]

        results.append((interval1, classifierToVal))
        i+=1
    return results



def filterSteadyState(comparedIntervals):
    result = []
    i = 0
    previous = classifiers.STEADY_STATE_CLASSIFIER;
    while (i < len(comparedIntervals)):
        interval, classicationString, isMatching = comparedIntervals[i]
        tokens = classicationString.split("|")
        chosen = tokens[0]
        chosen = chosen.strip()
        if not isMatching:
            expected = tokens[1]
            expected = expected.strip()
        if chosen == classifiers.STEADY_STATE_CLASSIFIER:
            chosen = previous
        else:
            previous = chosen
        if not isMatching and expected == chosen:
            isMatching = True
            result.append((interval, chosen, isMatching));
        else:
            result.append((interval, chosen + "|" + expected, isMatching))
        i+=1
    
    return result


def getClassificationAccuracy(comparedIntervals):
    confusionList = {}
    for interval, classificationString, isMatching in comparedIntervals:
        tokens = classificationString.split("|");
        chosen = tokens[0]
        chosen = chosen.strip()
        # if chosen  == classifiers.STEADY_STATE_CLASSIFIER:
        #     continue
        if not isMatching:
            expected = tokens[1]
            expected = expected.strip()
            # if expected == classifiers.STEADY_STATE_CLASSIFIER:
            #     continue
            if expected not in confusionList:
                confusionList[expected] = [datetime.timedelta(seconds=0)] * 4
        if chosen not in confusionList:
            confusionList[chosen] = [datetime.timedelta(seconds=0)] * 4
        length = intervalLength(interval)
        if isMatching:
            confusionList[chosen][0] += length
        else:
            confusionList[chosen][1] += length
            confusionList[expected][2] += length
            confusionList[expected][3] += length
        confusionList[chosen][3] += length

    totalTime = totalTimeOfIntervals(comparedIntervals)    
    for c in confusionList:
        tp, fp, fn, tt = confusionList[c]
        tn = totalTime - tp - fp -fn
        confusionList[c][3] = tn
    return confusionList

def classifierConfusionMatrix(confusionList):
    tp, fp, fn, tn = confusionList
    posTotal = tp + fn
    negTotal = tn + fp
    if posTotal == datetime.timedelta(seconds=0):
        tp = "N/A"
        fn = "N/A"
    else:
        tp = 1.0 * tp/posTotal
        fn = 1.0 * fn/posTotal
    confusionMatrix = [tp, 1.0 * fp/negTotal, fn, 1.0 * (tn/negTotal)]
    return confusionMatrix



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

        laterStartingInterval, earlierStartingInterval = None, None
        later_i, earlier_i = None, None

        if interval1[0] >= interval2[0]:
            # print("Interval1 starts after Interval2")
            laterStartingInterval, earlierStartingInterval = interval1, interval2
            later_i, earlier_i = "i1", "i2"
        else:
            # print("Interval2 starts after Interval1")
            laterStartingInterval, earlierStartingInterval = interval2, interval1
            later_i, earlier_i = "i2", "i1"

        if laterStartingInterval[0] >= earlierStartingInterval[1]:
            # print("GOODBYE")
            # print("Later starting interval starts completely after early interval")
            if earlier_i == "i1":
                i1 += 1
            else:
                i2 += 1
        
        else:
            # print("HELLO")
            earlierEndingInterval = earlierStartingInterval if earlierStartingInterval[1] <= laterStartingInterval[1] else laterStartingInterval
            # print("Earlier ending interval:", formatTimeInterval(earlierEndingInterval))
            
            commonIntervals.append((laterStartingInterval[0], earlierEndingInterval[1]))
            # print("Common Intervals:")
            # for interval in commonIntervals:
            #     print(formatTimeInterval(interval))


            if earlierStartingInterval[1] == laterStartingInterval[1]:
                # print("End times are equal")
                i1 += 1
                i2 += 1

            elif earlierStartingInterval[1] < laterStartingInterval[1]:
                # print("Early start ends earlier, advance early")
                if earlier_i == "i1":
                    i1 += 1
                else:
                    i2 += 1
                # print i1, i2
            else:
                # print("Early start ends later, advance later")
                if later_i == "i1":
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
    try:
        return interval[1] - interval[0]
    except:
        return datetime.timedelta(seconds=0)


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
    elif c[classifiers.STEADY_STATE_CLASSIFIER] in averagedClassifications:
        return classifiers.STEADY_STATE_CLASSIFIER
    elif c[classifiers.BACKPACK_CLASSIFIER] in averagedClassifications:
        return classifiers.BACKPACK_CLASSIFIER
    elif c[classifiers.BAG_CLASSIFIER] in averagedClassifications:
        return classifiers.BAG_CLASSIFIER
    elif c[classifiers.POCKET_CLASSIFIER] in averagedClassifications:
        return classifiers.POCKET_CLASSIFIER
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
    if type(dateTime) is not datetime.datetime:
        return str(datetime)
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

def formatTimeValue(timeValue, withDate=False):
    if type(timeValue) is str or type(timeValue) is int:
        return str(timeValue) 
    if type(timeValue) is datetime.datetime:
        return formatTime(timeValue, withDate)
    elif type(timeValue) is datetime.timedelta:
        return formatTimeDelta(timeValue)
    else:
        # must be an interval
        return formatTimeInterval(timeValue, withDate)


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
    return [timeStringToDateTime(getTimeFromFile(filename)) for filename in userFiles]

def continousIntervalsBleConnected(userData):
    startTime = -1
    prevTime = -1
    intervals = []
    prevState = -1
    delta = datetime.timedelta(seconds=60)

    for row in userData:
        time = row[0]
        state = row[-1]
        if startTime == -1:
            startTime = time
        elif time - prevTime > delta or prevState != state:
            intervals.append((startTime, prevTime, prevState))
            startTime = time
        prevState = state
        prevTime = time

    if prevTime != -1 and startTime != -1 and prevTime != -1:
        intervals.append((startTime, prevTime, prevState))
    return intervals



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

def toTime(t):
    if type(t) == datetime.datetime:
        return t.time()
    else:
        return t

def filterConsistentData(userData, consistentIntervals=[(START_TIME_FILTER, END_TIME_FILTER)]):
    consistentDataChunks = OrderedDict()

    if len(consistentIntervals) <= 0:
        return consistentDataChunks

    if not FILTER_ONLY_CONSISTENT_DATA:
        consistentDataChunks[(START_TIME_FILTER, END_TIME_FILTER)] = userData
        return consistentDataChunks

    print("Consistent Intervals:")
    for interval in consistentIntervals:
        print(formatTimeValue(interval, withDate=True))

    for sensor, sensorData in userData.items():
        i = 0
        currentInterval = consistentIntervals[i]
        currentStart, currentEnd = currentInterval
        currentDataChunk = []
        if currentInterval not in consistentDataChunks:
            consistentDataChunks[currentInterval] = {}

        consistentDataChunks[currentInterval][sensor] = currentDataChunk

        for rowOfData in sensorData:
            time = rowOfData[0].time()

            if time < toTime(currentStart):
                continue
            elif time < toTime(currentEnd):
                currentDataChunk.append(rowOfData)
            else:
                i += 1
                if i >= len(consistentIntervals):
                    break

                currentInterval = consistentIntervals[i]
                currentStart, currentEnd = currentInterval
                print("Data chunk (%s)" % sensor, formatTimeValue(consistentIntervals[i - 1], withDate=True), str(len(currentDataChunk)))
                currentDataChunk = []

                if currentInterval not in consistentDataChunks:
                    consistentDataChunks[currentInterval] = {}

                consistentDataChunks[currentInterval][sensor] = currentDataChunk

        for interval, chunk in consistentDataChunks.items():
            print(formatTimeInterval(interval), len(chunk))

    return consistentDataChunks

def filterConsistentIntervals(USER_ID, START_OF_TIME, END_OF_TIME, userData={}):
    if START_OF_TIME == None:
        START_OF_TIME = datetime.time(hour=0)

    if END_OF_TIME == None:
        END_OF_TIME = datetime.time(hour=23, minute=59)

    if len(userData) > 0 and len(userData[sensors.ACCELEROMETER]) > 0:
        accelData = userData[sensors.ACCELEROMETER]
        accelStart, accelEnd = accelData[0][0], accelData[-1][0]
        START_OF_TIME = max(accelStart, datetime.datetime.combine(accelStart.date(), START_OF_TIME))
        END_OF_TIME = min(accelEnd, datetime.datetime.combine(accelEnd.date(), END_OF_TIME))

    allIntervals = continuousWatchInterals(USER_ID, userData)

    bluetoothIntervals = allIntervals[sensors.CONNECTED_DEVICES]
    heartRateIntervals = allIntervals[HEARTRATE_SENSOR]

    def filterInRange(intervals):
        filtered_intervals = []
        i = 0
        for interval in intervals:
            start, end = interval[0], interval[1]
            if end < START_OF_TIME:
                i += 1
            elif start < START_OF_TIME:
                i += 1
                if len(interval) <= 2:
                    filtered_intervals.append((START_OF_TIME, end))
                else:
                    interval = list(interval)
                    interval[0] = START_OF_TIME
                    filtered_intervals.append(tuple(interval))
            elif start > END_OF_TIME:
                i += 1
            elif end > END_OF_TIME:
                i += 1
                if len(interval) <= 2:
                    filtered_intervals.append((start, END_OF_TIME))
                else:
                    interval = list(interval)
                    interval[1] = END_OF_TIME
                    filtered_intervals.append(tuple(interval))
            else:
                filtered_intervals.append(interval)

        # filtered_intervals.extend(intervals[i:])
        return filtered_intervals

    bluetoothIntervals = filterInRange(bluetoothIntervals)
    heartRateIntervals = filterInRange(heartRateIntervals)

    basisPeakIntervals = []
    for b in bluetoothIntervals:
        start, end, state = b
        if str(state) == "Basis Peak":
            basisPeakIntervals.append((start, end))

    noBasisPeakIntervals = inverseIntervals(basisPeakIntervals, START_OF_TIME, END_OF_TIME)
    noHeartIntervals = inverseIntervals(heartRateIntervals, START_OF_TIME, END_OF_TIME)

    heartAndBasisPeakIntervals = findCommonIntervals(heartRateIntervals, basisPeakIntervals)
    noHeartNoBasicPeakIntervals = findCommonIntervals(noHeartIntervals, noBasisPeakIntervals)

    consistentIntervals = merge_sorted_lists(heartAndBasisPeakIntervals, noHeartNoBasicPeakIntervals, lambda x, y: x[0] < y[0])

    # no heart rate, but watch connected (user not wearing watch) OR heart rate, but watch not connected (impossible)
    inconsistentIntervals = inverseIntervals(consistentIntervals, START_OF_TIME, END_OF_TIME)
    return heartAndBasisPeakIntervals, noHeartNoBasicPeakIntervals, consistentIntervals, inconsistentIntervals




def inverseIntervals(intervals, START_OF_TIME, END_OF_TIME):
    inverseIntervals = []
    start = START_OF_TIME
    # start = datetime.datetime.combine(intervals[0][0].date(), START_OF_TIME)
    # print(intervals[0])
    for interval in intervals:
        end = interval[0]
        next_start = interval[1]
        inverseIntervals.append((start, end))
        start = next_start
    end = END_OF_TIME
    # end = datetime.datetime.combine(start.date(), END_OF_TIME)
    inverseIntervals.append((start, end))
    return inverseIntervals

def merge_sorted_lists(l1, l2, cmp_func):
    merged = []
    i1 = 0
    i2 = 0
    while i1 < len(l1) and i2 < len(l2):
        item1 = l1[i1]
        item2 = l2[i2]

        if cmp_func(item1, item2):
            merged.append(item1)
            i1 += 1
        else:
            merged.append(item2)
            i2 += 2

    if i1 < len(l1):
        merged.extend(l1[i1:])
    else:
        merged.extend(l2[i2:])

    return merged

def main_filter_consistent():
    headers = ["% Time", "Total Time", "Median Period Length", "Average Period Length",
               "Longest Period Length", "Shortest Period Length", "Longest Period", "Shortest Period"]

    headers = ["User"] + [header + " (Near)" for header in headers] + [header + " (Far)" for header in headers] + [header + " (Inconsistent)" for header in headers]
    START_TIME = START_TIME_FILTER
    END_TIME = END_TIME_FILTER
    # TOTAL_DAY_TIME = (END_TIME - START_TIME).total_seconds()

    start_time = TIMER.time()

    NOW = datetime.datetime.now()
    NOW_TIME = NOW.strftime('_%m_%d_%H_%M')
    global DIRECTORY
    DIRECTORY_PATH = DIRECTORY

    for DATA_DAY in DATA_DATES:
        print("DIRECTORY started as:", DIRECTORY)
        DIRECTORY = DIRECTORY_PATH + DATA_DAY + "/"
        print("DIRECTORY now:", DIRECTORY)
        count = 0

        f = open('consistent-data-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
        writer = csv.writer(f)
        writer.writerow(headers)

        for USER_ID in USERS:
            print("Computing for user:", USER_ID)
            count += 1

            try:
                userData = getRelevantUserData(USER_ID)
                accelData = userData[sensors.ACCELEROMETER]
                TOTAL_DAY_TIME = (accelData[-1][0] - accelData[0][0]).total_seconds()
                print(str(accelData[-1][0]), str(accelData[0][0]), str(accelData[-1][0] - accelData[0][0]))
                nearIntervals, farIntervals, consistentIntervals, inconsistentIntervals = filterConsistentIntervals(USER_ID, START_TIME, END_TIME, userData=userData)

                row = [USER_ID]

                for intervals in [nearIntervals, farIntervals, inconsistentIntervals]:
                    stats = getIntervalStats(intervals)
                    row.append(stats["totalTimeSpent"].total_seconds() / TOTAL_DAY_TIME)
                    row.append(formatTimeValue(stats["totalTimeSpent"]))
                    row.append(formatTimeValue(stats["medianLength"]))
                    row.append(formatTimeValue(stats["avgLength"]))
                    print("long:", type(stats["longestInterval"]), stats["longestInterval"])
                    print("short:", type(stats["shortestInterval"]), stats["shortestInterval"])
                    row.append(formatTimeValue(intervalLength(stats["longestInterval"])))
                    row.append(formatTimeValue(intervalLength(stats["shortestInterval"])))
                    row.append(formatTimeValue(stats["longestInterval"]))
                    row.append(formatTimeValue(stats["shortestInterval"]))

                
                writer.writerow(row)
            except:
                tb = traceback.format_exc()
                print(tb)

        f.close()

    print("--- %s seconds ---" % (TIMER.time() - start_time))
    print("Yay I finished!")

def logConsistentIntervals(userData, USER_ID, consistentDataFile):
    headers = ["% Time", "Total Time", "Median Period Length", "Average Period Length",
               "Longest Period Length", "Shortest Period Length", "Longest Period", "Shortest Period"]

    headers = ["User"] + [header + " (Near)" for header in headers] + [header + " (Far)" for header in headers] + [header + " (Inconsistent)" for header in headers]
    START_TIME = START_TIME_FILTER
    END_TIME = END_TIME_FILTER

    f = consistentDataFile
    writer = csv.writer(f)
    writer.writerow(headers)

    try:
        accelData = userData[sensors.ACCELEROMETER]
        TOTAL_DAY_TIME = (accelData[-1][0] - accelData[0][0]).total_seconds()
        print(str(accelData[-1][0]), str(accelData[0][0]), str(accelData[-1][0] - accelData[0][0]))
        nearIntervals, farIntervals, consistentIntervals, inconsistentIntervals = filterConsistentIntervals(USER_ID,
                                                                                                            START_TIME,
                                                                                                            END_TIME,
                                                                                                            userData=userData)

        row = [USER_ID]

        for intervals in [nearIntervals, farIntervals, inconsistentIntervals]:
            stats = getIntervalStats(intervals)
            row.append(stats["totalTimeSpent"].total_seconds() / TOTAL_DAY_TIME)
            row.append(formatTimeValue(stats["totalTimeSpent"]))
            row.append(formatTimeValue(stats["medianLength"]))
            row.append(formatTimeValue(stats["avgLength"]))
            print("long:", type(stats["longestInterval"]), stats["longestInterval"])
            print("short:", type(stats["shortestInterval"]), stats["shortestInterval"])
            row.append(formatTimeValue(intervalLength(stats["longestInterval"])))
            row.append(formatTimeValue(intervalLength(stats["shortestInterval"])))
            row.append(formatTimeValue(stats["longestInterval"]))
            row.append(formatTimeValue(stats["shortestInterval"]))

        writer.writerow(row)
    except:
        tb = traceback.format_exc()
        print(tb)

def runWatchFunctions(USER_ID, watchResults, watchSummaryWriter, watchFile, DATA_DAY, userData={}):
    watchResults.write("#########" + USER_ID + "#######\n")
    try:
        watchState, continousWatchState = stateFromWatchData(continuousWatchInterals(USER_ID, userData=userData), watchFile)
        activatedWatchStates = watchActivationStates(continousWatchState)
        activatedIntervalsWatch = {PossessionState.PHONE_ACTIVATED: activatedWatchStates[0],
                                   PossessionState.PHONE_DEACTIVATED: activatedWatchStates[1]}
        # HERE STEVEN :)
        timeSpentByWatchState = {}
        # print(watchState)
        for state in watchState:
            watchResults.write("----" + str(state) + "-----" + "\n")
            intervals = watchState[state]
            # print("WTF!", state)
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
            percentage = time / totalTime if totalTime != 0 else 0
            percentageString = str(c) + "\t\t\t\t" + str(percentage * 100) + "%\n"
            watchResults.write(percentageString)
            percentageRow = [DATA_DAY, USER_ID, str(c), str(percentage * 100)]
            watchSummaryWriter.writerow(percentageRow)

        return activatedIntervalsWatch

    except:
        tb = traceback.format_exc()
        print(tb)
        watchResults.write("******EXCEPTION (while computing watch state)*******\n")
        watchResults.write(tb)
        watchResults.write("\n")
        return {PossessionState.PHONE_ACTIVATED: [],
                                   PossessionState.PHONE_DEACTIVATED: []}

def runClassifierFunctions(USER_ID, log_file, results, resultsSummaryWriter, DATA_DAY, NOW_TIME, userData={}):
    print("RUNNING CLASSIFIER FUNCTIONS")
    results.write("#########" + USER_ID + "#######\n")
    try:
        classifications, intervalsByClass, possessionState, aggregateClassifierResults = runClassifiersOnUser(USER_ID, None, log_file, userData=userData)

        # print("TRANSITION REASONS")
        # print(possessionState.transitionTimes)
        # print(possessionState.toActivatedTimes)
        # print(possessionState.toDeactivatedTimes)
        if DIARY_STUDY:
            print("Checking against Diary Study")
            expectedIntervalsDiary = getExpectedIntervals(DIARY_STUDY_FILE)
            noSteadyExpectedIntervals = getExpectedIntervals(DIARY_STUDY_NO_STEADY_FILE)
            checkClassifications(aggregateClassifierResults, DATA_DAY, NOW_TIME, expectedIntervalsDiary, noSteadyExpectedIntervals, classifications)

        activatedIntervalsPhone = possessionState.getIntervalsByState()

        ### START LOGGING ###
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
            resultsSummaryWriter.writerow([DATA_DAY, USER_ID, str(c), str(percentage * 100), formatTotalSeconds(time),
                                           formatTotalSeconds(totalTime)])

        results.write("-----Classifications over Time-------\n")
        for c in classifications:
            interval = c[0]
            duration = formatTimeValue(interval[1] - interval[0])
            classification = c[1]
            intervalString = "(" + formatTime(interval[0]) + "--" + formatTime(interval[1]) + "); "
            results.write(intervalString + ' ' + duration + '; ' + str(classification) + "\n")
        ### END LOGGING ###

        classifierFunctionResults = {
            'activatedIntervalsPhone' : activatedIntervalsPhone,
            'possessionState' : possessionState,
            'classifications' : classifications
        }

        return classifierFunctionResults

    except:
        tb = traceback.format_exc()
        print(tb)
        print("******EXCEPTION (while computing classifications)*******\n")
        results.write("******EXCEPTION (while computing classifications)*******\n")
        results.write(tb)
        results.write("\n")

        return {}

def runSmartUnlockFunctions(possessionState, activatedIntervalsPhone, smartUnlockFile, smartUnlockSummaryWriter, DATA_DAY, USER_ID):
    try:
        unlockData = possessionState.unlockData
        userData = possessionState.allData
        activatedIntervals = activatedIntervalsPhone["activated"]

        activatedIntervalsBle = continousIntervalsBleConnected(userData[sensors.CONNECTED_DEVICES])

        # pickle.dump(unlockData, open( "unlock_test_data.pkl", "wb" ) )
        # pickle.dump(activatedIntervals, open( "unlock_test_intervals.pkl", "wb" ) )

        numUnlocksSaved, numUnlocksTotal, unlockTimes = computeUnlocks(unlockData, activatedIntervals)

        numUnlocksSavedBle, numUnlocksTotalBle, unlockTimesBle = computeUnlocks(unlockData, activatedIntervalsBle)

        print("UNLOCK DATA:", str(numUnlocksSaved), str(numUnlocksTotal), str(unlockTimes))
        smartUnlockFile.write("#####UNLOCKS SAVED#######\n")
        smartUnlockFile.write("Unlocks saved: " + str(numUnlocksSaved) + "\n")
        smartUnlockFile.write("Total Unlocks: " + str(numUnlocksTotal) + "\n")

        smartUnlockFile.write("#####UNLOCKS SAVED (BLE) #######\n")
        smartUnlockFile.write("Unlocks saved: " + str(numUnlocksSavedBle) + "\n")
        smartUnlockFile.write("Total Unlocks: " + str(numUnlocksTotalBle) + "\n")

        ##########################

        activatedRow = [DATA_DAY, USER_ID, numUnlocksSaved, numUnlocksSavedBle, numUnlocksTotal]
        smartUnlockSummaryWriter.writerow(activatedRow)

        smartUnlockFile.write("####### VERBOSE INFO #########\n")
        smartUnlockFile.write("######## SAVED UNLOCK TIMES #######\n")

        for time in unlockTimes:
            smartUnlockFile.write(formatTimeValue(time) + "\n")

        smartUnlockFile.write("######## SAVED UNLOCK TIMES (BLE) #######\n")

        for time in unlockTimesBle:
            smartUnlockFile.write(formatTimeValue(time) + "\n")

    except:
        tb = traceback.format_exc()
        print(tb)
        smartUnlockFile.write("******EXCEPTION (while computing activations)*******\n")
        smartUnlockFile.write(tb)
        smartUnlockFile.write("\n")

def calculateUnlocksMetrics(possessionState, activatedIntervalsPhone):
    if activatedIntervalsPhone != None:
        unlockData = possessionState.unlockData
        activatedIntervals = activatedIntervalsPhone["activated"]

        numUnlocksSaved, numUnlocksTotal, unlockTimes = computeUnlocks(unlockData, activatedIntervals)
        unlockMetrics = {
            'numUnlocksSaved' : numUnlocksSaved,
            'numUnlocksTotal' : numUnlocksTotal,
            'unlockTimes' : unlockTimes
        }
        return unlockMetrics
    return {}

def matchTimesWithIntervals(timesWithDescriptions, intervals, classifications, logFile):
    if len(intervals) <= 0:
        return

    i = 0
    currInterval = intervals[i]
    start, end = currInterval

    print("Times with descriptions:")
    for time, description in timesWithDescriptions.items():
        print("Time:", formatTimeValue(time), "Description:", description)

    print("The bad intervals yo")
    for interval in intervals:
        print(formatTimeValue(interval))


    j = 0
    currClassification = None
    cInterval = None
    if len(classifications) > 0:
        cInterval, currClassification = classifications[j]
        cStart, cEnd = cInterval

    for time, description in timesWithDescriptions.items():
        while time >= end:
            if time < start:
                continue
            elif time <= end:
                print("TIME LESS THAN END")
                # while j < len(classifications) and cEnd < start:
                #     j += 1
                #     cInterval, currClassification = classifications[j]
                #     cStart, cEnd = cInterval

                logFile.write(formatTimeValue(intervals[i]) + "\n")

                logString = formatTimeValue(time) + "\t" + description + "\n"
                logFile.write(logString)
            else:
                i += 1
                if i >= len(intervals):
                    break
                start, end = intervals[i]
        if i >= len(intervals):
            break


def runActivationFunctions(activatedIntervalsWatch, activatedIntervalsPhone, toActivatedTimes, toDeactivatedTimes, classifications, unlockMetrics, activatedFile, activatedSummaryWriter, activationsLogFile, DATA_DAY, USER_ID, NOW_TIME, tag=""):
    print(tag)
    try:
        activatedFile.write("***********" + USER_ID + "*************" + "\n")
        if activatedIntervalsWatch == None or activatedIntervalsPhone == None:
            activatedFile.write("Check: " + 'watch-testing-results-' + DATA_DAY + NOW_TIME + '.txt')

        else:
            numUnlocksSaved, numUnlocksTotal, unlockTimes = unlockMetrics['numUnlocksSaved'], unlockMetrics['numUnlocksTotal'], unlockMetrics['unlockTimes']
            print("UNLOCK DATA:", str(numUnlocksSaved), str(numUnlocksTotal), str(unlockTimes))
            activatedFile.write("#####UNLOCKS SAVED#######\n")
            activatedFile.write("Unlocks saved: " + str(numUnlocksSaved) + "\n")
            activatedFile.write("Total Unlocks: " + str(numUnlocksTotal) + "\n")

            ##########################

            activatedRow = [DATA_DAY, USER_ID, numUnlocksSaved, numUnlocksTotal]

            totalActivatedTestTimes = 0
            stateTimes = {}
            for stateP in activatedIntervalsPhone:
                for stateW in activatedIntervalsWatch:

                    state = "Phone: " + stateP + " Watch: " + stateW
                    print(state)
                    # activatedFile.write(str(state) + '\n')
                    # print("Phone Intervals:", activatedIntervalsPhone[stateP])
                    # print("Watch Intervals:", activatedIntervalsWatch[stateW])
                    commonIntervals = findCommonIntervals(activatedIntervalsPhone[stateP],
                                                          activatedIntervalsWatch[stateW])

                    try:
                        if stateP == PossessionState.PHONE_ACTIVATED and stateW == PossessionState.PHONE_DEACTIVATED:
                            #TODO: Find all transition changes within the common intervals
                            matchTimesWithIntervals(toActivatedTimes, commonIntervals, classifications, activationsLogFile)
                            print("False positives!")
                            # print(possessionState.transitionTimes)
                        elif stateP == PossessionState.PHONE_DEACTIVATED and stateW == PossessionState.PHONE_ACTIVATED:
                            matchTimesWithIntervals(toDeactivatedTimes, commonIntervals, classifications, activationsLogFile)
                            print("False negatives!")
                            # print(possessionState.transitionTimes)
                    except:
                        tb = traceback.format_exc()
                        print(tb)


                    # print("COMMON INTERVALS:", commonIntervals)
                    stats = getIntervalStats(commonIntervals)
                    # print(stats["totalTimeSpent"])
                    # activatedFile.write(str(stats["totalTimeSpent"]) + '\n')

                    timeSeconds = stats["totalTimeSpent"].total_seconds()
                    totalActivatedTestTimes += timeSeconds
                    stateTimes[state] = stats["totalTimeSpent"]
                    
            print("ACTIVATION PERCENTAGES")
            print(str(stateTimes))
            activatedFile.write("######ACTIVATION CONFUSION MATRIX#######\n")
            header = " " * 15 + "\t" + "Watch Activated\t\t" + "Watch Deactivated\n"
            activatedFile.write(header)
            percentRow = []
            timeRow = []
            for stateP in ["activated", "deactivated"]:
                state1 = "Phone: " + stateP + " Watch: " + "activated"
                time1 = stateTimes[state1].total_seconds()
                percentage1 = time1 / totalActivatedTestTimes if totalActivatedTestTimes > 0 else 0
                print("Time1:", time1)

                state2 = "Phone: " + stateP + " Watch: " + "deactivated"
                time2 = stateTimes[state2].total_seconds()
                print("Time2:", time2)
                percentage2 = time2 / totalActivatedTestTimes if totalActivatedTestTimes > 0 else 0

                if stateP == "activated":
                    stateP += '\t'

                datum = "Phone " + stateP + "\t" + str(percentage1 * 100)[:6] + '%' + "\t\t" + str(percentage2 * 100)[
                                                                                               :6] + '%' + "\n"
                times = " " * 20 + "\t" + str(stateTimes[state1])[:7] + "\t\t" + str(stateTimes[state2])[:7] + "\n"

                # print(state, "-->", percentage)
                # activatedFile.write(str(state) + " --> " + str(percentage * 100) + '\n')
                activatedFile.write(datum)
                activatedFile.write(times)
                percentRow += [percentage1, percentage2]
                timeRow += [str(stateTimes[state1]), str(stateTimes[state2])]

            activatedRow += percentRow
            activatedRow += timeRow

            activatedSummaryWriter.writerow(activatedRow)

            activatedFile.write("####### VERBOSE INFO #########\n")
            activatedFile.write("######## SAVED UNLOCK TIMES #######\n")

            for time in unlockTimes:
                activatedFile.write(formatTimeValue(time) + "\n")

            activatedFile.write("######## PHONE INTERVALS #######\n")
            for state, intervals in activatedIntervalsPhone.items():
                activatedFile.write("#########" + str(state).upper() + "########" + "\n")
                for interval in intervals:
                    activatedFile.write(formatTimeInterval(interval) + "\n")

            activatedFile.write("######## WATCH INTERVALS #######\n")
            for state, intervals in activatedIntervalsWatch.items():
                activatedFile.write("#########" + str(state).upper() + "########" + "\n")
                for interval in intervals:
                    activatedFile.write(formatTimeInterval(interval) + "\n")
    except:
        tb = traceback.format_exc()
        print(tb)
        activatedFile.write("******EXCEPTION (while computing activations)*******\n")
        activatedFile.write(tb)
        activatedFile.write("\n")

def mergeActivationIntervals(a1, a2):
    a1[PossessionState.PHONE_ACTIVATED].extend(a2[PossessionState.PHONE_ACTIVATED])
    a1[PossessionState.PHONE_DEACTIVATED].extend(a2[PossessionState.PHONE_DEACTIVATED])

def mergeUnlockMetrics(m1, m2):
    m1['numUnlocksSaved'] += m2['numUnlocksSaved']
    m1['numUnlocksTotal'] += m2['numUnlocksTotal']
    m1['unlockTimes'].extend(m2['unlockTimes'])

def main():
    print("SAFE PERIOD:", SAFE_PERIOD)
    # main_filter_consistent()
    # USER_ID = '6fdda897'
    start_time = TIMER.time()
    # print("Start:", start_time)
    global DIRECTORY

    NOW = datetime.datetime.now()
    NOW_TIME = NOW.strftime('_%m_%d_%H_%M')
    DIRECTORY_PATH = DIRECTORY

    if not os.path.exists(RESULTS_DIRECTORY):
        os.makedirs(RESULTS_DIRECTORY)
    else:
        print("RESULTS DIR ALREADY EXISTS:", RESULTS_DIRECTORY)

    for DATA_DAY in DATA_DATES:
        print("DIRECTORY started as:", DIRECTORY)
        DIRECTORY = DIRECTORY_PATH + DATA_DAY + "/"
        print("DIRECTORY now:", DIRECTORY)
        # if not FULL_STUDY_RUN:
        file = open(RESULTS_DIRECTORY + '/' + 'testing-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
        watchFile = open(RESULTS_DIRECTORY + '/' + 'watch-testing-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
        results = open(RESULTS_DIRECTORY + '/' + 'testing-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
        watchResults = open(RESULTS_DIRECTORY + '/' + 'watch-testing-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
        resultsSummary = open(RESULTS_DIRECTORY + '/' + 'testing-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
        watchSummary = open(RESULTS_DIRECTORY + '/' + 'watch-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
        resultsSummaryWriter = csv.writer(resultsSummary)
        watchSummaryWriter = csv.writer(watchSummary)
        resultsSummaryWriter.writerow(["Day", "User", "Classifier", "Percentage of Time"])
        watchSummaryWriter.writerow(["Day", "User", "State", "Percentage of Time", "Hours", "Total Hours"])
        activatedFile = open(RESULTS_DIRECTORY + '/' + 'activated-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
        activatedSummary = open(RESULTS_DIRECTORY + '/' + 'activated-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
        activatedSummaryWriter = csv.writer(activatedSummary)
        activatedSummaryWriter.writerow(["Day", "User", "Unlocks Saved", "Unlocks Total", "Percent Both Activated", "Percent Only Phone Activated", "Percent Only Watch Activated", "Percent Both Deactivated", "Both Activated", "Only Phone Activated", "Only Watch Activated", "Both Deactivated"])

        activatedRawFile = open(RESULTS_DIRECTORY + '/' + 'activated-raw-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
        activatedRawSummary = open(RESULTS_DIRECTORY + '/' + 'activated-raw-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
        activatedRawSummaryWriter = csv.writer(activatedRawSummary)
        activatedRawSummaryWriter.writerow(
            ["Day", "User", "Unlocks Saved", "Unlocks Total", "Percent Both Activated", "Percent Only Phone Activated",
             "Percent Only Watch Activated", "Percent Both Deactivated", "Both Activated", "Only Phone Activated",
             "Only Watch Activated", "Both Deactivated"])

        activationsLogFile = open(RESULTS_DIRECTORY + '/' + 'activated-log-'+ DATA_DAY + NOW_TIME + '.txt', 'w+')
        activationsLogRawFile = open(RESULTS_DIRECTORY + '/' + 'activated-raw-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')

        consistentDataFile = open(RESULTS_DIRECTORY + '/' + 'consistent-data-' + DATA_DAY + NOW_TIME + '.csv', 'w+')

        smartUnlockSummary = open(RESULTS_DIRECTORY + '/' + 'smart-unlock-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
        smartUnlockSummaryWriter = csv.writer(smartUnlockSummary)
        smartUnlockFile = open(RESULTS_DIRECTORY + '/' + 'smart-unlock-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')

        count = 0
        for USER_ID in USERS:
            try:
                count += 1
                print("Number of users processed:", count)
                print("Currently on:", USER_ID)

                if USE_CACHED_DATA:
                    pickle_file_name = './' + 'DATA/' + USER_ID + "_data_full.pkl"
                    pickle_file = open(pickle_file_name, 'rb')

                    userData = pickle.load(pickle_file)
                else:
                    userData = getRelevantUserData(USER_ID)


                nearIntervals, farIntervals, consistentIntervals, inconsistentIntervals = filterConsistentIntervals(USER_ID,
                                                                                                                    START_TIME_FILTER,
                                                                                                                    END_TIME_FILTER,
                                                                                                                    userData=userData)

                logConsistentIntervals(userData, USER_ID, consistentDataFile)

                consistentDataSegments = filterConsistentData(userData, consistentIntervals)

                activatedIntervalsPhoneAggregate = {PossessionState.PHONE_ACTIVATED : [], PossessionState.PHONE_DEACTIVATED : []}
                activatedIntervalsWatchAggregate = {PossessionState.PHONE_ACTIVATED : [], PossessionState.PHONE_DEACTIVATED : []}
                activatedTransitionTimes = OrderedDict()
                deactivatedTransitionTimes = OrderedDict()

                unlockMetricsAggregate = {
                'numUnlocksSaved' : 0,
                'numUnlocksTotal' : 0,
                'unlockTimes' : []
                }

                for consistentInterval, userData in consistentDataSegments.items():
                    activatedIntervalsWatch = None
                    activatedIntervalsPhone = None

                    if not RUN_CLASSIFIERS_ONLY:
                        activatedIntervalsWatch = runWatchFunctions(USER_ID, watchResults, watchSummaryWriter, watchFile, DATA_DAY, userData=userData)
                        mergeActivationIntervals(activatedIntervalsWatchAggregate, activatedIntervalsWatch)

                    if not RUN_WATCH_ONLY:
                        functionResults = runClassifierFunctions(USER_ID, file, results, resultsSummaryWriter, DATA_DAY, NOW_TIME, userData=userData)
                        if len(functionResults) > 0:
                            activatedIntervalsPhone, possessionState, classifications = functionResults['activatedIntervalsPhone'], functionResults['possessionState'], functionResults['classifications']
                            mergeActivationIntervals(activatedIntervalsPhoneAggregate, activatedIntervalsPhone)
                            activatedTransitionTimes.update(possessionState.toActivatedTimes)
                            deactivatedTransitionTimes.update(possessionState.toDeactivatedTimes)

                    if activatedIntervalsPhone != None:
                        # runSmartUnlockFunctions(possessionState, activatedIntervalsPhone, smartUnlockFile, smartUnlockSummaryWriter, DATA_DAY, USER_ID)
                        unlockMetrics = calculateUnlocksMetrics(possessionState, activatedIntervalsPhone)
                        if len(unlockMetrics) > 0:
                            mergeUnlockMetrics(unlockMetricsAggregate, unlockMetrics)
                
                print("Run with watchActivation")
                runActivationFunctions(activatedIntervalsWatchAggregate, activatedIntervalsPhoneAggregate, activatedTransitionTimes, deactivatedTransitionTimes, classifications, unlockMetricsAggregate, activatedFile, activatedSummaryWriter, activationsLogFile, DATA_DAY, USER_ID, NOW_TIME, tag="Activation (Filtered)")
                
                activatedIntervalsWatchFromConsistency = {
                    PossessionState.PHONE_ACTIVATED : nearIntervals,
                    PossessionState.PHONE_DEACTIVATED : farIntervals
                }
                
                print("Run with raw")
                runActivationFunctions(activatedIntervalsWatchFromConsistency, activatedIntervalsPhoneAggregate, activatedTransitionTimes, deactivatedTransitionTimes, classifications, unlockMetricsAggregate, activatedRawFile, activatedRawSummaryWriter, activationsLogRawFile, DATA_DAY, USER_ID, NOW_TIME, tag="Activation (RAW)")
            except:
                tb = traceback.format_exc()
                print(tb)

        if not FULL_STUDY_RUN:
            file.close()
            watchFile.close()
            results.close()
            watchResults.close()

    print("--- %s seconds ---" % (TIMER.time() - start_time))
    print("Yay I finished!")

if __name__ == '__main__':
    # print("HELLO")
    # global SAFE_PERIOD,f

    parser = argparse.ArgumentParser()
    parser.add_argument("safeperiod", help="Safe period for policy",
                        type=int)
    parser.add_argument("--cached", help="Use cached data files in ./DATA/",
                        action="store_true")
    args = parser.parse_args()

    SAFE_PERIOD = args.safeperiod
    USE_CACHED_DATA = args.cached

    main()
    # main_filter_consistent()

