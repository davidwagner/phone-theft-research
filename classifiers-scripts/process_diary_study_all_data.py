import pandas as pd
import datetime
import re
import numpy as np
import sys
import argparse
import glob
import csv


### CONFIGURE HERE ###

# Directory where the diary study data is
DIR = '../data/diary-study-11-14-15/diary_data/'
# Path to diary file
DIARY_FILE = '../data/diary-study-11-14-15/diary_state_no_steady.txt'
# User ID of data
USER_ID = 'd792b61e'

"""
Configure the variables above, then call getAllUserData().
Returns:
1) numpy matrix of all acceleromter, touch, screen state, unlock data, and class according to the diary study
2) names of the coluns of the numpy matrix
"""
def getAllUserData(data_dir, diary_file, user_id, as_matrix=True, calibrate_accel=False):
    files = getUserFilesByInstrument(data_dir, user_id, 'BatchedAccelerometer')
    x = compileAccelData(files, diary_file, as_matrix=True, calibrate=calibrate_accel)
    accelDf = pd.DataFrame(data=x, columns = ["Timestamp", "X accel.", "Y accel.", "Z accel.", "Other accel", "Class"])

    posData, rawPosDataLocked = processPhoneActiveData(data_dir, user_id, x)
    posDataFrame = pd.DataFrame(data=posData, columns=["Timestamp", "Num. Touches", "Screen State", "isUnlocked", "X direction changed", "Y direction changed", "Z direction changed"])


    ALL_DATA = accelDf.set_index("Timestamp").join(posDataFrame.set_index("Timestamp"), how='inner', lsuffix='d1', rsuffix='d2')
    ALL_DATA_MATRIX = ALL_DATA.reset_index().values
    ALL_DATA_COLS = ALL_DATA.columns

    return ALL_DATA_MATRIX if as_matrix else ALL_DATA, ALL_DATA_COLS


def convertToDateTime(timestring, bootTime):
    epochTimeString = float(timestring) / 1000.0
    timeAsEpochTime = datetime.date.fromtimestamp(epochTimeString)
    isAbsoluteTimeStamp = timeAsEpochTime > datetime.date(2000, 1, 1)
    if isAbsoluteTimeStamp:
        return timeAsEpochTime
    else:
        # print(datetime.timedelta(milliseconds=int(timestring)))
        return bootTime + datetime.timedelta(milliseconds=int(timestring))

    
def getTimeFromFile(filename):
    filename = filename.split("/")[-1]
    # print("filename", filename)
    query = '.*' + 'AppMon' + '_' + '.*_' + '[a-zA-z]+_' + '(?P<time>.*)' + '_.csv'
    match = re.match(query, filename)
    return match.group('time')

def getUserFilesByInstrument(directory, user, instrument):
    query = directory + 'AppMon_' + user + '*_' + instrument + '_' + '*'
    # print(query)
    userFiles = glob.glob(query)
    userFiles.sort()
    return userFiles

def getBootTimestampedData(f):
    data = pd.read_csv(filepath_or_buffer=f,
                       header=None)
    fileTime = getTimeFromFile(f)
    fileTime = datetime.datetime.strptime(fileTime, '%Y_%m_%d_%H_%M_%S')
    firstTimestamp = data.iloc[0][0]
    
    bootTime = fileTime - datetime.timedelta(milliseconds=int(firstTimestamp))
    
    data[0] = data[0].map(lambda timestamp : convertToDateTime(timestamp, bootTime))
    
    return data

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

def get_accel_calibration_constants(accel_data, is_mean=True):
    X_Y_STILL_THRESHOLD_MEAN = 0.75
    X_Y_STILL_THRESHOLD_STD = 0.75
    
    data = pd.DataFrame(data=accel_data, columns=["Time", "X", "Y", "Z", "N/A", "Class"])
    d = data.set_index("Time")
    d_roll = d[["X", "Y", "Z"]].rolling(6000)
    d_agg = d_roll.mean().join(d_roll.std(), how='inner', lsuffix='mean', rsuffix='std')
    
    mask = (d_agg["Xmean"].abs() < X_Y_STILL_THRESHOLD_MEAN) & (d_agg["Xstd"].abs() < X_Y_STILL_THRESHOLD_STD) \
        & (d_agg["Ymean"].abs() < X_Y_STILL_THRESHOLD_MEAN) & (d_agg["Ystd"].abs() < X_Y_STILL_THRESHOLD_STD) \
        & (d_agg["Zmean"].abs() > 8) & (d_agg["Zmean"].abs() < 12) & (d_agg["Zstd"] < 1)
    
    table_data = d_agg[mask]
    x, y, z = table_data.median()[["Xmean", "Ymean", "Zmean"]]
    print(table_data.std()[["Xmean", "Ymean", "Zmean"]])
    
    if is_mean:
        x, y, z = table_data.mean()[["Xmean", "Ymean", "Zmean"]]
        
    
    x_offset, y_offset, z_offset = 0 - x, 0 - y, 9.8 - z
    return x_offset, y_offset, z_offset

def calibrate_accel(accel_table):
    cols = accel_table.columns

    x_offset, y_offset, z_offset = get_accel_calibration_constants(accel_table.as_matrix())
    
    accel_table[cols[1]] += x_offset
    accel_table[cols[2]] += y_offset
    accel_table[cols[3]] += z_offset

    return accel_table


def compileAccelData(accel_files, diary_study_f=None, as_matrix=True, calibrate=False):
    data = pd.concat([getBootTimestampedData(f) for f in accel_files])
    data = data.sort_values(0)
    
    new_col = len(data.columns)
    data[new_col] = ''
    
    if diary_study_f != None:
        expectedIntervals = getExpectedIntervals(diary_study_f)

        for interval, label in expectedIntervals:
            start, end = interval
            mask = (data[0] >= start) & (data[0] < end)
            data.loc[mask, new_col] = label

    data = calibrate_accel(data)
    
    data = data.as_matrix() if as_matrix else data
    return data
    
def processPhoneActiveData(data_dir, ID, posDataAccel):
    if len(posDataAccel) <= 1:
        return []

    firstAccelTime = posDataAccel[0][0]
    
    posFilesTouch = getUserFilesByInstrument(data_dir, ID, 'TouchScreenAsEvent')
    rawPosDataTouch = dataFilesToDataListAbsTime(posFilesTouch)
    # # print("RAW DATA TOUCH")
    # # print(rawPosDataTouch)
    
    posFilesScreen = getUserFilesByInstrument(data_dir, ID, 'TriggeredScreenState')
    rawPosDataScreen = dataFilesToDataListAbsTime(posFilesScreen)
    
    posFilesLocked = getUserFilesByInstrument(data_dir, ID, 'TriggeredKeyguard')
    rawPosDataLocked = dataFilesToDataListAbsTime(posFilesLocked)



    currScreenDate = None
    nextScreenDate = None
    currScreenVal = None
    currLockedDate = None
    nextLockedDate = None
    currLockedVal = None
    
    touchIndex = -1
    if rawPosDataTouch.shape[0] > 0:
        touchIndex = 0
        currentTime = rawPosDataTouch[touchIndex][0]
        while currentTime < firstAccelTime: 
            touchIndex += 1
            if touchIndex >= rawPosDataTouch.shape[0]:
                break
            currentTime = rawPosDataTouch[touchIndex][0]

        startTouchIndex = touchIndex
    
    screenIndex = -1
    if rawPosDataScreen.shape[0] > 0:
        screenIndex = 0
        currentTime = rawPosDataScreen[screenIndex][0]
        while currentTime < firstAccelTime:
            screenIndex += 1
            if screenIndex >= rawPosDataScreen.shape[0]:
                break
            currentTime = rawPosDataScreen[screenIndex][0]
            # # print(currentTime)
            # # print(screenIndex)
        currScreenDate = rawPosDataScreen[screenIndex][0]
        currScreenVal = rawPosDataScreen[screenIndex][2]
        if rawPosDataScreen.shape[0] > 1:
            nextScreenDate = rawPosDataScreen[screenIndex + 1][0]
    
    lockedIndex = -1
    if rawPosDataLocked.shape[0] > 0:
        lockedIndex = 0
        currentTime = rawPosDataLocked[lockedIndex][0]
        while currentTime < firstAccelTime:
            lockedIndex += 1
            if lockedIndex >= rawPosDataLocked.shape[0]:
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
    
    for i in range(posDataAccel.shape[0] - 1):
        accelRow = posDataAccel[i]
        accelRowNext = posDataAccel[i + 1]
        
        accelDate = accelRow[0]
        accelDateNext = accelRowNext[0]
        
        # Calculate number of touch events starting at this row time and before next row time
        # touchDate >= firstAccelTime
        if rawPosDataTouch.shape[0] == 0 or touchIndex >= rawPosDataTouch.shape[0] or rawPosDataTouch[touchIndex][0] >= accelDateNext: # No touch events
            touchRow = [accelDate, 0]
            posDataTouch.append(touchRow)
            ## print("TOUCH DATE:" + str(touchDate))
            ## print("ACCEL DATE:" + str(accelDate))
            
        else: #touchDate < AccelDateNext
            numTouches = 0
            touchDate = rawPosDataTouch[touchIndex][0]
            while touchDate < accelDateNext and touchIndex < rawPosDataTouch.shape[0]:
                # # print("TOUCH RECOGNIZED!")
                numTouches += 1
                touchIndex += 1
                if touchIndex < rawPosDataTouch.shape[0] - 1:
                    touchDate = rawPosDataTouch[touchIndex][0]
                
            touchRow = [accelDate, numTouches]
            posDataTouch.append(touchRow)
        
        
        # Calculate if screen on in this interval
        if currScreenDate == None or nextScreenDate == None:
            screenRow = [accelDate, 0]
            posDataScreen.append(screenRow)

        elif accelDate >= nextScreenDate:
            if screenIndex + 1 < rawPosDataScreen.shape[0]:
                screenIndex += 1
                currScreenDate = rawPosDataScreen[screenIndex][0]
                currScreenVal = rawPosDataScreen[screenIndex][2]
                if screenIndex + 1 < rawPosDataScreen.shape[0]:
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
            if lockedIndex + 1 < rawPosDataLocked.shape[0]:
                lockedIndex += 1
                currLockedDate = rawPosDataLocked[lockedIndex][0]
                currLockedVal = rawPosDataLocked[lockedIndex][2]
                if lockedIndex + 1 < rawPosDataLocked.shape[0]:
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
    for i in range(posDataAccel.shape[0] - 1):
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


def dataFilesToDataListAbsTime(userFiles):
    dataLists = [csvToTable(dataFile) for dataFile in userFiles]
    dataListP = pd.concat(dataLists)

    dataListP[0] = dataListP[1].map(lambda timestamp : datetime.datetime.fromtimestamp(int(timestamp) / 1000))
    # dataListP = timeFilteredData(dataListP)
    
    # checkListAndDataframeAreEqual(dataList, dataListP)
    return dataListP.as_matrix()

def csvToTable(csvFile):
    return pd.read_csv(filepath_or_buffer=csvFile,
                       header=None)

def main():
    data, cols = getAllUserData(DIR, DIARY_FILE, USER_ID)
    print(cols)
    print(data[:100,:])


if __name__ == '__main__':
    main()







