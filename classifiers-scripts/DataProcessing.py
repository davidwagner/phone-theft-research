import Sensors as sensors
import Intervals
import TimeFileUtils

import glob
import re
import datetime

from configsettings import *
from collections import deque, Counter

BOOT_TIME_DELTA = datetime.timedelta(hours=1)
BOOT_TIME_SENSOR = sensors.ACCELEROMETER
START_OF_TIME = datetime.datetime.min

START_TIME_FILTER = datetime.time(hour=0)
END_TIME_FILTER = datetime.time(hour=23, minute=59)
# START_TIME_FILTER = None
# END_TIME_FILTER = None


def getUserFilesByDayAndInstrument(userID, DIRECTORY, instrument):
    query = DIRECTORY + 'AppMon_' + userID + '*_' + instrument + '_' + '*'
    userFiles = glob.glob(query)
    userFiles.sort()
    # TODO: Need to filter for sensors that need data files with matching times as other
    # sensors (e.g. accelerometer and step count for Theft Classifier)
    print("QUERY:", query)
    return userFiles

def getUserFileTimes(userFiles):
    return [TimeFileUtils.timeStringToDateTime(getTimeFromFile(filename)) in filename in userFiles]

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
            
            fileTime = TimeFileUtils.timeStringToDateTime(TimeFileUtils.getTimeFromFile(dataFile))

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

            firstRow[0] = TimeFileUtils.convertToDateTime(firstRow[0], currentBootTime)
            minLength = len(firstRow)
            if len(firstRow) >= 2:
                if (START_TIME_FILTER == None or firstRow[0].time() >= START_TIME_FILTER) and (END_TIME_FILTER == None or firstRow[0].time() < END_TIME_FILTER):
                    dataList.append(firstRow)
            count = 1
            for row in reader:
                if len(row) >= 2 and len(row) >= minLength:
                    row[0] = TimeFileUtils.convertToDateTime(row[0], currentBootTime)
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

def getReferenceBootTimes(userID, DIRECTORY):
    userFiles = getUserFilesByDayAndInstrument(userID, DIRECTORY, BOOT_TIME_SENSOR)

    bootTimes = []
    currentBootTime = START_OF_TIME
    
    for dataFile in userFiles:
        with open(dataFile) as f:  
            reader = csv.reader(f)
            
            fileTime = TimeFileUtils.timeStringToDateTime(TimeFileUtils.getTimeFromFile(dataFile))

            firstRow = reader.next()
            firstTime = datetime.timedelta(milliseconds=int(firstRow[0]))

            bootTime = fileTime - firstTime

            difference = bootTime - currentBootTime if bootTime > currentBootTime else currentBootTime - bootTime

            if difference > BOOT_TIME_DELTA:
                currentBootTime = bootTime 
                bootTimes.append((fileTime, bootTime))

    return bootTimes



def getRelevantUserData(userID, DIRECTORY, logInfo=False, logFile=None):
    userData = {}
    bootTimes = []

    dataFiles = getUserFilesByDayAndInstrument(userID, DIRECTORY, BOOT_TIME_SENSOR)
    userData[BOOT_TIME_SENSOR] = dataFilesToDataList(dataFiles, bootTimes, True)

    for instrument in sensors.RELEVANT_SENSORS:
        if instrument != BOOT_TIME_SENSOR and instrument != sensors.PHONE_ACTIVE_SENSORS:
            
            dataFiles = getUserFilesByDayAndInstrument(userID, DIRECTORY, instrument)
            userData[instrument] = dataFilesToDataList(dataFiles, bootTimes)
    
    #print(len(userData[sensors.ACCELEROMETER]))
    userData[sensors.PHONE_ACTIVE_SENSORS], userData[sensors.KEYGUARD] = processPhoneActiveData(userID, DIRECTORY, userData[sensors.ACCELEROMETER])
    print("KEYGUARD", len(userData[sensors.KEYGUARD]))

    # print("GONNA TRY TO GET LIGHT SENSOR DATA")
    userData[sensors.LIGHT_SENSOR] = processLightSensorData(userData)
    userData[BOOT_TIME_SENSOR] = userData[BOOT_TIME_SENSOR][:-1]
    print("Length accel:", len(userData[BOOT_TIME_SENSOR]))
    print("Length active:", len(userData[sensors.PHONE_ACTIVE_SENSORS]))

    for instrument in sensors.WATCH_SENSORS:
        dataFiles = getUserFilesByDayAndInstrument(userID, DIRECTORY, instrument)
        # print "Heart Rate Files"
        # print dataFiles
        userData[instrument] = dataFilesToDataListAbsTime(dataFiles)

    if logInfo:
        logFile.write("Data Files Analyzed:\n")
        for filename in dataFiles:
            logFile.write(TimeFileUtils.getTimeFromFile(filename) + "_.csv" + '\n')
        logFile.write("Boot Times Computed:\n")
        for bootTime in bootTimes:
            logFile.write("Files after " + str(TimeFileUtils.formatTime(bootTime[0], withDate=True)) + ", have boot time: " + str(TimeFileUtils.formatTime(bootTime[1], withDate=True)) + '\n')

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

def processPhoneActiveData(ID, DIRECTORY, posDataAccel):
    if len(posDataAccel) <= 1:
        return [], []

    firstAccelTime = posDataAccel[0][0]
    
    posFilesTouch = getUserFilesByDayAndInstrument(ID, DIRECTORY, 'TouchScreenAsEvent')
    rawPosDataTouch = dataFilesToDataListAbsTime(posFilesTouch)
    # # print("RAW DATA TOUCH")
    # # print(rawPosDataTouch)
    
    posFilesScreen = getUserFilesByDayAndInstrument(ID, DIRECTORY,'TriggeredScreenState')
    rawPosDataScreen = dataFilesToDataListAbsTime(posFilesScreen)
    
    posFilesLocked = getUserFilesByDayAndInstrument(ID, DIRECTORY,'TriggeredKeyguard')
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