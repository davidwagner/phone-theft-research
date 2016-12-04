import csv
import os
import glob
import re
import datetime
import sys
import shutil
import matplotlib.pyplot as plt
from matplotlib.dates import SecondLocator, MinuteLocator, HourLocator, DateFormatter, date2num
# import classifier
import Sensors as sensors
import Classifiers as classifiers 

NOW = datetime.datetime.now()
# NOW_DAY = NOW.strftime('%Y_%m_%d')
NOW_DAY = '2016_11_10'


# Replace with your own
# DATA_DIR = "/home/daw/Dropbox/phone_data/Dashboard_results/"
DASHBOARDDIR = './Classifier_Results_TEST/'
DATA_DIR = "../../../Dropbox/phone_data/Sensor_Research/"
DECRYPTED_DIR = "./Decrypted_Data/"
DIRECTORY = DECRYPTED_DIR + NOW_DAY + "/"

# USERS_TO_IDS_FILE = "./users_to_ids_test.csv"
USERS_TO_IDS_FILE = "./users_r2_test.csv"
ids = []
usersToIdsFile = open(USERS_TO_IDS_FILE, 'rU')
try:
    reader = csv.reader(usersToIdsFile)
    for row in reader:
        userID = row[1]
        ids.append(userID)
finally:
    usersToIdsFile.close()

USERS = set(ids)

RELEVANT_SENSORS = set([])
YEAR_2000 = datetime.date(2000, 1, 1)

BOOT_TIME_DELTA = datetime.timedelta(hours=1)
BOOT_TIME_SENSOR = sensors.ACCELEROMETER
START_OF_TIME = datetime.datetime.min


def getUserFilesByDayAndInstrument(userID, instrument):
    query = DECRYPTED_DIR + NOW_DAY + '/' + 'AppMon_' + userID + '*_' + instrument + '_' + NOW_DAY + '*'
    # print(query)
    userFiles = glob.glob(query)

    # TODO: Need to filter for sensors that need data files with matching times as other
    # sensors (e.g. accelerometer and step count for Theft Classifier)
    
    return userFiles


def dataFilesToDataList(userFiles, bootTimes, needsToComputeBootTime=False):
    dataList = []
    currentBootTime = START_OF_TIME
    nextFileTime = START_OF_TIME
    nextFileTimeIndex = 0
    
    for dataFile in userFiles:
        with open(dataFile) as f:  
            reader = csv.reader(f)
            
            fileTime = timeStringToDateTime(getTimeFromFile(dataFile))

            firstRow = reader.next()
            firstTime = datetime.timedelta(milliseconds=int(firstRow[0]))

            if needsToComputeBootTime:
                bootTime = fileTime - firstTime

                difference = bootTime - currentBootTime if bootTime > currentBootTime else currentBootTime - bootTime

                if difference > BOOT_TIME_DELTA:
                    currentBootTime = bootTime 
                    bootTimes.append((fileTime, bootTime))
            
            else:
                
                if fileTime > nextFileTime:
                    currentBootTime = bootTimes[nextFileTimeIndex][1] # boot time has changed, update
                    nextFileTimeIndex = nextFileTimeIndex + 1 if nextFileTimeIndex < len(bootTimes) - 1 else nextFileTimeIndex
                    nextFileTime = bootTimes[nextFileTimeIndex][0]

            firstRow[0] = convertToDateTime(firstRow[0], currentBootTime)
            dataList.append(firstRow)
            for row in reader:
                row[0] = convertToDateTime(row[0], currentBootTime)
                dataList.append(row)
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



def getRelevantUserData(userID):
    userData = {}
    bootTimes = []

    dataFiles = getUserFilesByDayAndInstrument(userID, BOOT_TIME_SENSOR)
    userData[BOOT_TIME_SENSOR] = dataFilesToDataList(dataFiles, bootTimes, True)

    for instrument in RELEVANT_SENSORS:
        if instrument != BOOT_TIME_SENSOR:
            dataFiles = getUserFilesByDayAndInstrument(userID, instrument)
            
            userData[instrument] = dataFilesToDataList(dataFiles, bootTimes)

    return userData 


def runClassifier(classifier, userData):
    windowSize = classifier.getWindowTime()
    instruments = classifier.getRelevantSensors()
    
    numRows = min([len(userData[instrument]) for instrument in instruments])
    #print(len(userData[sensors.ACCELEROMETER]))
    
    # {instrument: windowOfRawData}
    results = {0 : [], 1 : []}
    resultIntervals = []
    resultTimes = []

    if numRows == 0:
        return results, resultIntervals, resultTimes
    print(userData[instrument][0])
    firstTime = userData[instrument][0][0]
    currentInterval = (firstTime, firstTime)
    currentClass = -1

    for row in range(0, numRows - windowSize):
        windowOfData = {}
        for instrument in instruments:
            data = userData[instrument][row:row + windowSize] 
            windowOfData[instrument] = data
            windowStartTime = getWindowStartTime(data)

        # print(windowStartTime)
        classification = classifier.classify(windowOfData)
        # print(classification)
        # resultTimes.append((windowStartTime, classification))

        # Adjust the interval
        if currentClass == -1:
            currentClass = classification
        elif currentClass != classification:
            resultIntervals.append((currentInterval, currentClass))
            interval = currentInterval
            print((formatTime(interval[0]), formatTime(interval[1])), currentClass)
            currentInterval = (windowStartTime, windowStartTime)
            currentClass = classification
        else:
            currentInterval = (currentInterval[0], windowStartTime)


    resultIntervals.append((currentInterval, currentClass))
    # filterSpikesFromIntervals(resultIntervals)

        # results[classification].append(windowStartTime)
    return results, resultIntervals, resultTimes

# {windowStartTime : 0, 1}
# {7:30pm : 0}

# {0 : [list of times], 1 : [list of times]}

def runClassifiersOnUser(userID, csvWriter):
    print(userID)
    userData = getRelevantUserData(userID)
  
    csvRow = [userID]

    csvWriter.write(str(userID) + '\n')
    for c in classifiers.CLASSIFIERS:
        classifier = classifiers.CLASSIFIERS[c]
        print(classifiers.CLASSIFIERS)
        csvWriter.write(str(c) + '\n')
        results, intervals, times = runClassifier(classifier, userData)
        for interval in intervals:
            intervalString = (formatTime(interval[0][0]), formatTime(interval[0][1]))
            result = (intervalString, interval[1])
            csvWriter.write(str(result) + '\n')

        csvWriter.write('#######\n')

        # for time in times:
        #     result = (formatTime(time[0]), time[1])
        #     csvWriter.write(str(result) + '\n')

        plotIntervals(intervals)        
        #print(results)
        # processResults(results, csvWriter, csvRow)

# TODO:
def processResults(results, writer, csvrow):
    # analyze results
    # write actionable output to writer
    positives = results[0]
    negatives = results[1]
    if len(negatives) > 0:
        return 1
    return 0

def filterSpikesFromIntervals(intervals):
    spikeLength = datetime.timedelta(seconds=1)
    i = 1
    while i < len(intervals) - 1:
        interval, intervalBefore, intervalAfter = intervals[i], intervals[i - 1], intervals[i + 1]

        timeInterval = interval[0]

        if timeInterval[1] - timeInterval[0] <= spikeLength:
            newTimeInterval = (intervalBefore[0][0], intervalAfter[0][1])
            intervals[i - 1] = (newTimeInterval, intervalBefore[1])
            del intervals[i:i+2]
        else:
            i += 1



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

def formatTime(dateTime):
    return dateTime.strftime('%H:%M:%S:%f')

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

def removeFilesFromDir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def replaceCommasWithSemicolons(string):
    return string.replace(",", ";")
 
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

    dashboardFile = open(dashboardFileName, 'wb')
    dashboardWriter = csv.writer(dashboardFile, delimiter = ',')

    columnHeaders = ["User ID", 
                     ]

    dashboardWriter.writerow(columnHeaders)

    compileRelevantSensors()

    tempResultsName = DASHBOARDDIR + "Dashboard-" + now.strftime('%Y_%m_%d_%H_%M') + ".txt"
    tempResultsFile = open(tempResultsName, 'wb')
    for userID in USERS:
        datarow = [userID]
        # runClassifiersOnUser(userID, dashboardWriter)
        runClassifiersOnUser(userID, tempResultsFile)
    tempResultsFile.close()

    print("Dashboard results generated in: " + dashboardFileName)

if __name__ == '__main__':
    main()

