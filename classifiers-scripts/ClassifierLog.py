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

##### CONFIGURE FOR OWN MACHINE ###############
# Replace with your own
DASHBOARDDIR = '../data/Classifier_Results_TEST/'
USERS_TO_IDS_FILE = "../data/users_r2_test.csv"
DIRECTORY = "../data/Decrypted_Data/2016_11_01/"
###############################################
DUMP_RESULTS = True

if DIRECTORY[-1] != '/':
    DIRECTORY += '/'

NOW = datetime.datetime.now()
# NOW_DAY = NOW.strftime('%Y_%m_%d')

YESTERDAY = (NOW - datetime.timedelta(days=1)).strftime('%Y_%m_%d')
# NOW_DAY = YESTERDAY
NOW_DAY = '2016_11_01'

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

# RELEVANT_SENSORS = set([])
RELEVANT_SENSORS = [sensors.ACCELEROMETER]
YEAR_2000 = datetime.date(2000, 1, 1)

BOOT_TIME_DELTA = datetime.timedelta(hours=1)
BOOT_TIME_SENSOR = sensors.ACCELEROMETER
START_OF_TIME = datetime.datetime.min


def getUserFilesByDayAndInstrument(userID, instrument):
    query = DIRECTORY + 'AppMon_' + userID + '*_' + instrument + '_' + '*'
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
            if len(firstRow) >= 5:
                dataList.append(firstRow)
            count = 1
            for row in reader:
                if len(row) >= 5:
                    row[0] = convertToDateTime(row[0], currentBootTime)
                    dataList.append(row)
                # count += 1
                # if count > 100000:
                #     break
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
    print(len(userData[sensors.ACCELEROMETER]))

    # {instrument: windowOfRawData}
    resultIntervalsByValue = ([], [])
    resultIntervals = []

    if numRows == 0:
        return resultIntervals, resultIntervalsByValue

    firstTime = userData[instrument][0][0]
    currentInterval = (firstTime, firstTime)
    currentClass = -1

    if windowSize == classifiers.DAY_INTERVAL:
        windowSize = numRows - 1
        for row in range(0, numRows - windowSize):
            windowOfData = {}
            for instrument in instruments:
                data = userData[instrument][row:row + windowSize] 
                windowOfData[instrument] = data
                windowStartTime = getWindowStartTime(data)

        classifications = classifier.classify(windowOfData)
        if classifications == None or classifications == [(0, 0)]:
            return resultIntervals, resultIntervalsByValue

        resultIntervals = classifications
        # mergeAdjacentIntervals(resultIntervals)
        # filterSpikesFromIntervals(resultIntervals, resultIntervalsByValue)
        for c in classifications:
            label = c[1]
            timeInterval = c[0]
            resultIntervalsByValue[label].append(timeInterval)


        print("THEFT CLASSIFICATIONS")
        print(classifications)

        return resultIntervals, resultIntervalsByValue

    else:
        print("Number of samples: " + str(numRows - windowSize))
        for row in range(0, numRows - windowSize):
            windowOfData = {}
            for instrument in instruments:
                data = userData[instrument][row:row + windowSize] 
                windowOfData[instrument] = data
                windowStartTime = getWindowStartTime(data)

            # if row % 1000 == 0:
            #     print("# of windows classified: " + str(row))
            #     print(formatTime(windowStartTime))

            classification = classifier.classify(windowOfData)

            # Adjust the interval
            if currentClass == -1:
                currentClass = classification
            elif currentClass != classification:
                resultIntervals.append((currentInterval, currentClass))
                # results[currentClass].append(currentInterval)
                interval = currentInterval
                # print((formatTime(interval[0]), formatTime(interval[1])), currentClass)
                currentInterval = (windowStartTime, windowStartTime)
                currentClass = classification
            else:
                currentInterval = (currentInterval[0], windowStartTime)

        # results[currentClass].append(currentInterval)
        resultIntervals.append((currentInterval, currentClass))
        print(resultIntervals)
        filterSpikesFromIntervals(resultIntervals, resultIntervalsByValue)

    return resultIntervals, resultIntervalsByValue

# {windowStartTime : 0, 1}
# {7:30pm : 0}

# {0 : [list of times], 1 : [list of times]}

def runClassifiersOnUser(userID, csvWriter, resultsFile):
    print(userID)
    userData = getRelevantUserData(userID)
  
    csvRow = [userID]

    # csvWriter.write(str(userID) + '\n')
    results = {}
    theft = classifiers.THEFT_CLASSIFIER
    print(theft)
    classifier = classifiers.CLASSIFIERS[theft]
    classifierResults = runClassifier(classifier, userData)
    processTheftResults(classifierResults, csvWriter, csvRow)
    results[theft] = classifierResults
    print ("Results computed for: " + theft)

    if DUMP_RESULTS:
        resultsFile.write("###########################\n")
        resultsFile.write(str(userID) + '\n')
        logResultsToFile(classifierResults, theft, resultsFile)

    for c in classifiers.CLASSIFIERS:
        if c == classifiers.THEFT_CLASSIFIER:
            continue
        classifier = classifiers.CLASSIFIERS[c]
        print(c)
        # csvWriter.write(str(c) + '\n')
        classifierResults = runClassifier(classifier, userData)
        processResults(classifierResults, csvWriter, csvRow)
        results[c] = classifierResults
        print("Results computed for: " + c)
        if DUMP_RESULTS:
            logResultsToFile(classifierResults, c, resultsFile)

    processAllClassifierResults(results, csvRow)

    csvWriter.writerow(csvRow)

def logResultsToFile(classifierResults, classifier_name, resultsFile):
    resultsFile.write(str(classifier_name) + '\n')
    resultsFile.write("-------------------------------------\n")
    resultIntervals, resultIntervalsByValue = classifierResults
    resultsFile.write("Result Intervals\n")
    for interval in resultIntervals:
        interval = (formatTimeInterval(interval[0]), interval[1])
        resultsFile.write(str(interval) + '\n')

    posTimes = resultIntervalsByValue[1]
    negTimes = resultIntervalsByValue[0]

    resultsFile.write("Positive Intervals\n")
    for interval in posTimes:
        resultsFile.write(formatTimeInterval(interval) + '\n')

    resultsFile.write("Negative Intervals\n")
    for interval in negTimes:
        resultsFile.write(formatTimeInterval(interval) + '\n')



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
    print "These classifications conflict"
    print conflicitingClassifications
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

def mergeAdjacentIntervals(intervals):
    i = 0
    while i + 1 < len(intervals):
        curr = intervals[i]
        next = intervals[i + 1]
        if curr[1] == next[1]:
            intervals[i] = ((curr[0][0], next[0][1]), curr[1])
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
    print("Finding common intervals!")
    print intervals1
    print intervals2

    if len(intervals1) == 0 and len(intervals2) == 0:
        return []
    if len(intervals1) == 0:
        return intervals2
    if len(intervals2) == 0:
        return intervals1 

    def advance(intervals, i, value):
        while i < len(intervals) and intervals[i][1] != value:
            print(i)
            i += 1
        return i 

    i1 = advance(intervals1, 0, value) 
    i2 = advance(intervals2, 0, value)
    print i1, i2 
    
    commonIntervals = []
    while i1 < len(intervals1) and i2 < len(intervals2):
        interval1 = intervals1[i1][0]
        interval2 = intervals2[i2][0]
        # print(i1, i2)
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
            print commonIntervals

            if earlierStartingInterval[1] == laterStartingInterval[1]:
                print "End times are equal"
                i1 = advance(intervals1, i1, value)
                i2 = advance(intervals2, i2, value)

            elif earlierStartingInterval[1] < laterStartingInterval[1]:
                print "Early start ends earlier, advance early"
                if earlier_i == i1:
                    i1 = advance(intervals1, i1, value)
                else:
                    i2 = advance(intervals2, i2, value)
                print i1, i2
            else:
                print "Early start ends later, advance later"
                if later_i == i1:
                    i1 = advance(intervals1, i1, value)
                else:
                    i2 = advance(intervals2, i2, value)
                print i1, i2

    return commonIntervals

def findCommonIntervals(intervals1, intervals2):
    print("Finding common intervals!")
    print intervals1
    print intervals2

    if len(intervals1) == 0 and len(intervals2) == 0:
        return []
    if len(intervals1) == 0:
        return intervals2
    if len(intervals2) == 0:
        return intervals1 

    i1 = 0
    i2 = 0
    print "Starting"
    print i1, i2 
    
    commonIntervals = []
    while i1 < len(intervals1) and i2 < len(intervals2):
        interval1 = intervals1[i1]
        interval2 = intervals2[i2]
        # print(i1, i2)
        laterStartingInterval, earlierStartingInterval = None, None
        later_i, earlier_i = None, None

        if interval1[0] >= interval2[0]:
            laterStartingInterval, earlierStartingInterval = interval1, interval2
            later_i, earlier_i = i1, i2
        else:
            laterStartingInterval, earlierStartingInterval = interval2, interval1
            later_i, earlier_i = i2, i1

        if laterStartingInterval[0] >= earlierStartingInterval[1]:
            print("GOODBYE")
            if earlier_i == i1:
                i1 += 1
            else:
                i2 += 1
        
        else:
            print("HELLO")
            earlierEndingInterval = earlierStartingInterval if earlierStartingInterval[1] <= laterStartingInterval[1] else laterStartingInterval

            commonIntervals.append((laterStartingInterval[0], earlierEndingInterval[1]))
            print commonIntervals

            if earlierStartingInterval[1] == laterStartingInterval[1]:
                print "End times are equal"
                i1 += 1
                i2 += 1

            elif earlierStartingInterval[1] < laterStartingInterval[1]:
                print "Early start ends earlier, advance early"
                if earlier_i == i1:
                    i1 += 1
                else:
                    i2 += 1
                print i1, i2
            else:
                print "Early start ends later, advance later"
                if later_i == i1:
                    i1 += 1
                else:
                    i2 += 1
                print i1, i2

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
    return dateTime.strftime('%H:%M:%S')

def formatTimeDelta(timeDelta):
    totalSeconds = timeDelta.total_seconds()
    hours = totalSeconds // 3600
    minutes = (totalSeconds % 3600) // 60
    seconds = totalSeconds % 60
    return str(hours) + 'h:' + str(minutes) + 'm:' + str(seconds) + 's' 

def formatTimeInterval(timeInterval):
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
        print c
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
    print("Dashboard results generated in: " + dashboardFileName)

if __name__ == '__main__':
    main()

