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
def getAllUserData(data_dir, diary_file, user_id, as_matrix=True, calibrate_accel=False, no_boundary=False):
    print("Compiling data from:", data_dir)
    files = getUserFilesByInstrument(data_dir, user_id, 'BatchedAccelerometer')

    print("Getting Accel Data")
    x = compileAccelData(files, diary_file, as_matrix=True, calibrate=calibrate_accel, no_boundary_data=no_boundary)
    accelDf = pd.DataFrame(data=x, columns = ["Timestamp", "X accel.", "Y accel.", "Z accel.", "Class"])

    print("Getting Phone Active Data")
    ALL_DATA, rawPosDataLocked = processPhoneActiveData(data_dir, user_id, accelDf)
    # posDataFrame = pd.DataFrame(data=posData, columns=["Timestamp", "Num. Touches", "Screen State", "isUnlocked", "X direction changed", "Y direction changed", "Z direction changed"])


    # ALL_DATA = accelDf.set_index("Timestamp").join(posDataFrame.set_index("Timestamp"), how='inner', lsuffix='d1', rsuffix='d2')
    ALL_DATA_MATRIX = ALL_DATA.values
    ALL_DATA_COLS = ALL_DATA.columns

    print("Finished compiling data")

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
    
    d = pd.DataFrame(data=accel_data, columns=["X", "Y", "Z", "Class"])

    # print("Calc calibration any null?")
    # print(data[data.isnull().any(axis=1)])

    # d = data.set_index("Time")
    d_roll = d[["X", "Y", "Z"]].rolling(6000)
    # print("ROLLING")
    # print(d_roll.mean())
    # print(d_roll.std())
    d_agg = d_roll.mean().join(d_roll.std(), how='inner', lsuffix='mean', rsuffix='std')

    # print("AGGREGATE TABLE")
    # print(d_agg)
    
    mask = (d_agg["Xmean"].abs() < X_Y_STILL_THRESHOLD_MEAN) & (d_agg["Xstd"].abs() < X_Y_STILL_THRESHOLD_STD) \
        & (d_agg["Ymean"].abs() < X_Y_STILL_THRESHOLD_MEAN) & (d_agg["Ystd"].abs() < X_Y_STILL_THRESHOLD_STD) \
        & (d_agg["Zmean"].abs() > 8) & (d_agg["Zmean"].abs() < 12) & (d_agg["Zstd"] < 1)
    
    table_data = d_agg[mask]
    # print("MATCHING TO TABLE DATA")
    # print(table_data)

    x, y, z = table_data.median()[["Xmean", "Ymean", "Zmean"]]
    # print(table_data.std()[["Xmean", "Ymean", "Zmean"]])
    
    if is_mean:
        x, y, z = table_data.mean()[["Xmean", "Ymean", "Zmean"]]
        
    
    x_offset, y_offset, z_offset = 0 - x, 0 - y, 9.8 - z
    print("CALIBRATION CONSTANTS:", x_offset, y_offset, z_offset)
    return x_offset, y_offset, z_offset

def calibrate_accel(accel_table):
    cols = accel_table.columns
    print("ACCEL COLS:?", cols[1:4])

    accel_cols = ["X accel.", "Y accel.", "Z accel.", "Class"]
    x_offset, y_offset, z_offset = get_accel_calibration_constants(accel_table[accel_cols].as_matrix())
    
    accel_table["X accel."] += x_offset
    accel_table["Y accel."] += y_offset
    accel_table["Z accel."] += z_offset

    return accel_table


def compileAccelData(accel_files, diary_study_f=None, as_matrix=True, calibrate=False, no_boundary_data=False):
    data = pd.concat([getBootTimestampedData(f) for f in accel_files])
    data = data.sort_values(0)
    
    # Drop N/A column
    data = data.drop(4, axis=1)

    # Drop rows with nans
    data = data.dropna(axis=0)

    # mask = (data[1] != np.nan) & (data[2] != np.nan) & (data[3] != np.nan)
    # data = data[mask]
    
    new_col = len(data.columns)
    data[new_col] = ''
    
    if diary_study_f != None:
        expectedIntervals = getExpectedIntervals(diary_study_f)
        # print(data)
        for interval, label in expectedIntervals:
            start, end = interval
            mask = (data[0] >= start) & (data[0] < end)
            print("CHECK INTERVAL", interval, label)
            print(data[mask].shape)
            data.loc[mask, new_col] = label

            if no_boundary_data:
                print("Before boundary filter:", data.shape)
                mask = ~((data[0] >= start) & (data[0] < start + datetime.timedelta(minutes=1)))
                data = data[mask]
                print("After boundary filter:", data.shape)

    # print("UNCALIBRATED NULL")
    # print(data[data.isnull().any(axis=1)])

    # mask = data[new_col] == ''
    # print("DID NOT FALL IN INTERVALS")
    # print(data[mask])
    
    data = data.as_matrix() if as_matrix else data
    return data


def processPhoneActive(data_dir, ID, posDataAccel):
    # print("Adding screen state")
    posFilesScreen = getUserFilesByInstrument(data_dir, ID, 'TriggeredScreenState')
    addActiveData(posFilesScreen, posDataAccel, "Screen State", zero_val='false')

    # print("Adding unlocks")
    posFilesLocked = getUserFilesByInstrument(data_dir, ID, 'TriggeredKeyguard')
    addActiveData(posFilesLocked, posDataAccel, "isUnlocked", zero_val='true')

    # print("Adding touches")
    posFilesTouch = getUserFilesByInstrument(data_dir, ID, 'TouchScreenAsEvent')
    rawPosDataTouch = dataFilesToDataListAbsTime(posFilesTouch)
    rawPosDataTouch = pd.DataFrame(data=rawPosDataTouch)
    processTouches(rawPosDataTouch, posDataAccel)

    return posDataAccel

def processTouches(touchDf, accelDf):
    accelDf['Num. Touches'] = 0

    curTouchIndex = 0
    curAccelIndex = 1

    numAccelRows = accelDf.shape[0]
    numTouchRows = touchDf.shape[0]

    accel = accelDf.iloc
    while curAccelIndex < numAccelRows - 1 and curTouchIndex < numTouchRows:
        startTime, endTime = accel[curAccelIndex, 0], accel[curAccelIndex + 1, 0]
        touchTime = touchDf.iloc[curTouchIndex, 0]

        if touchTime < startTime:
            curTouchIndex += 1
        elif touchTime >= startTime and touchTime < endTime:
            accelDf.loc[curAccelIndex, 'Num. Touches'] += 1

            curTouchIndex += 1
        else:
            curAccelIndex += 1
    print("curAccel:", curAccelIndex, "numAccel:", numAccelRows, "curTouch:", curTouchIndex, "numTouch:", numTouchRows)

    return accelDf


def addActiveData(csv_files, accelDf, column, zero_val='false'):
    convertTime = lambda timestamp : datetime.datetime.fromtimestamp(int(timestamp) / 1000)
    truthToNum = lambda x : 0 if str(x).lower() == zero_val else 1

    accelDf[column] = 0

    currVal = 0
    startTime, endTime = accelDf.iloc[0][0], None
    trueMask = None
    for f in csv_files:
        # print(f)
        reader = csv.reader(open(f, 'r'))

        for row in reader:
            if len(row) < 3:
                # print("Bad row:", row)
                continue
            time, val = convertTime(row[1]), truthToNum(row[2])
            
            endTime = time
            
            if val != currVal:
                # if currVal is True and val is False, then a True interval is ending
                if currVal == 1 and val == 0:
                    # print("Appending true interval:", startTime, endTime)
                    mask = ((accelDf['Timestamp'] >= startTime) & (accelDf['Timestamp'] < endTime))
                    if trueMask is None:
                        trueMask = mask
                    else:
                        trueMask = trueMask | mask
                       
                # start new interval
                startTime = time
                currVal = val
                
                # print("Starting new interval:", currVal, startTime)
            
    
    if currVal == True:
        mask = ((accelDf['Timestamp'] >= startTime) & (accelDf['Timestamp'] < endTime))
        if trueMask is None:
            trueMask = mask
        else:
            trueMask = trueMask | mask
            
    accelDf.loc[trueMask, column] = 1

    return accelDf

    
def processPhoneActiveData(data_dir, ID, posDataAccel):
    if len(posDataAccel) <= 1:
        return [], []

    print("Adding screen state")
    posFilesScreen = getUserFilesByInstrument(data_dir, ID, 'TriggeredScreenState')
    addActiveData(posFilesScreen, posDataAccel, "Screen State", zero_val='false')

    print("Adding unlocks")
    posFilesLocked = getUserFilesByInstrument(data_dir, ID, 'TriggeredKeyguard')
    rawPosDataLocked = dataFilesToDataListAbsTime(posFilesLocked)
    addActiveData(posFilesLocked, posDataAccel, "isUnlocked", zero_val='true')

    print("Adding touches")
    posFilesTouch = getUserFilesByInstrument(data_dir, ID, 'TouchScreenAsEvent')
    rawPosDataTouch = dataFilesToDataListAbsTime(posFilesTouch)
    rawPosDataTouch = pd.DataFrame(data=rawPosDataTouch)
    processTouches(rawPosDataTouch, posDataAccel)

    allData = posDataAccel
    # posDataAccel = posDataAccel.values
    # print("Adding signs")

    # allData["X direction changed"] = 0
    # allData["Y direction changed"] = 0
    # allData["Z direction changed"] = 0

    # curAccelSignX = float(posDataAccel[0][1]) > 0
    # curAccelSignY = float(posDataAccel[0][2]) > 0
    # curAccelSignZ = float(posDataAccel[0][3]) > 0
    
    # curSigns = [curAccelSignX, curAccelSignY, curAccelSignZ]
    
    # signsChanged = lambda now, cur : [1 if now[i] != cur[i] else 0 for i in range(len(now))]
    # for i in range(posDataAccel.shape[0] - 1):
    #     try:
    #         accelSignX = float(posDataAccel[i][1]) > 0
    #         accelSignY = float(posDataAccel[i][2]) > 0
    #         accelSignZ = float(posDataAccel[i][3]) > 0
            
    #         newSigns = [accelSignX, accelSignY, accelSignZ]
    #         accelSigns = signsChanged(newSigns, curSigns)
    #         curSigns = newSigns

    #         allData.loc[i, ["X direction changed", "Y direction changed", "Z direction changed"]] = accelSigns

    #     except (ValueError,IndexError):
    #         print("BAD VALUE OF I:", i)
    #         accelSigns = signsChanged(curSigns, curSigns)
    #         allData.loc[i, ["X direction changed", "Y direction changed", "Z direction changed"]] = accelSigns
    
    return allData, rawPosDataLocked


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







