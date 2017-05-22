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
import traceback
import Intervals
import TimeFileUtils
import DataProcessing

from configsettings import *
from collections import deque, Counter
from UnlockTimeChecker import computeUnlocks

DUMP_RESULTS = True

if DIRECTORY[-1] != '/':
    DIRECTORY += '/'

maxWindowSize = 100

NOW = datetime.datetime.now()
# NOW_DAY = NOW.strftime('%Y_%m_%d')

YESTERDAY = (NOW - datetime.timedelta(days=1)).strftime('%Y_%m_%d')
# NOW_DAY = YESTERDAY
NOW_DAY = '2016_11_01'

YEAR_2000 = datetime.date(2000, 1, 1)

def continuousWatchInterals(userID):
    userData = {}
    for instrument in sensors.WATCH_SENSORS:
        dataFiles = getUserFilesByDayAndInstrument(userID, DIRECTORY, instrument)
        # print "Heart Rate Files"
        # print dataFiles
        userData[instrument] = DataProcessing.dataFilesToDataListAbsTime(dataFiles)
    watchData = userData
    delta = datetime.timedelta(seconds=60)
    allIntervals = {}

    for instrument in sensors.WATCH_SENSORS:
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
    heartRateIntervals = allIntervals[sensors.HEARTRATE_SENSOR]
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
    Intervals.mergeAdjacentIntervals(deactivated)
    Intervals.mergeAdjacentIntervals(activated)
    return activated, deactivated

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
    heartRateData = userData[sensors.HEARTRATE_SENSOR]
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
    userData = DataProcessing.getRelevantUserData(userID, DIRECTORY, logInfo=True, logFile=resultsFile)
    heartRateTimes = getHeartRateTimes(userData)

    csvRow = [userID]
    results = {}
    pickleResults = {}

    for instrument in sensors.RELEVANT_SENSORS:
        print(instrument, ":", len(userData[instrument]))    

    numRows = min([len(userData[instrument]) for instrument in sensors.RELEVANT_SENSORS])

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

    possessionState = PossessionState.PossessionState(userData, userData[sensors.PHONE_ACTIVE_SENSORS], userData[sensors.KEYGUARD], SMOOTHING_NUM)
    for i in range(0, limit, maxWindowSize):
        windowOfData = {}
        windowStartTime = 0
        for instrument in sensors.RELEVANT_SENSORS:
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


    classifications.append((currentInterval, currentClass))
    intervalsByClass[currentClass].append(currentInterval)

    return classifications, intervalsByClass, possessionState


def logResultsToFile(classifierResults, classifier_name, resultsFile):
    resultsFile.write("-------------------------------------\n")
    resultsFile.write(str(classifier_name) + '\n')
    resultsFile.write("-------------------------------------\n")
    resultIntervals, resultIntervalsByValue = classifierResults
    resultsFile.write("Result Intervals\n")
    for interval in resultIntervals:
        interval = (TimeFileUtils.formatTimeInterval(interval[0], withDate=True), interval[1])
        resultsFile.write(str(interval) + '\n')

    posTimes = resultIntervalsByValue[1]
    negTimes = resultIntervalsByValue[0]

    resultsFile.write("Positive Intervals\n")
    for interval in posTimes:
        resultsFile.write(TimeFileUtils.formatTimeInterval(interval, withDate=True) + ' ; ' + TimeFileUtils.formatTimeValue(intervalLength(interval)) + '\n')

    resultsFile.write("Negative Intervals\n")
    for interval in negTimes:
        resultsFile.write(TimeFileUtils.formatTimeInterval(interval, withDate=True) + ' ; ' + TimeFileUtils.formatTimeValue(intervalLength(interval)) + '\n')



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
        posTimesString = TimeFileUtils.intervalsToString(posTimes)
        longestPosInterval = posTimes[0]
        longestPosIntervalLength = intervalLength(posTimes[0])
        for interval in posTimes:
            length = intervalLength(interval)
            if length > longestPosIntervalLength:
                longestPosIntervalLength = length
                longestPosInterval = interval

        longestPosIntervalString = TimeFileUtils.formatTimeInterval(longestPosInterval)

    csvRow += [longestPosIntervalString, posTimesString, str(numPos), str(numNeg), str(numTotal)]       

def processResults(results, writer, csvRow):
    # analyze results
    # write actionable output to writer
    resultIntervals, resultIntervalsByValue = results[0], results[1]
    
    negativeIntervals = sorted(resultIntervalsByValue[0], key=intervalLength) 
    positiveIntervals = sorted(resultIntervalsByValue[1], key=intervalLength)

    negStats = Intervals.getIntervalStats(negativeIntervals)
    posStats = Intervals.getIntervalStats(positiveIntervals)

    negTime = negStats["totalTimeSpent"].total_seconds()
    posTime = posStats["totalTimeSpent"].total_seconds()
    totalTime = negTime + posTime

    negTimePercentage = negTime / totalTime if totalTime > 0 else 0
    posTimePercentage = posTime / totalTime if totalTime > 0 else 0

    stats = ["totalTimeSpent", "medianLength", "avgLength", "longestInterval", "shortestInterval"]

    csvRow.append(posTimePercentage)
    for stat in stats:
        val = posStats[stat]
        csvRow.append(TimeFileUtils.formatTimeValue(val))
    
    csvRow.append(negTimePercentage)
    for stat in stats:
        val = negStats[stat]
        csvRow.append(TimeFileUtils.formatTimeValue(val))

def processAllClassifierResults(results, csvRow):
    conflicitingClassifications = findConflictingClassifications(results, False)
    # print "These classifications conflict"
    # print conflicitingClassifications
    if len(conflicitingClassifications) > 0:
        csvRow += [TimeFileUtils.intervalsToString(conflicitingClassifications)]
    else:
        csvRow += ["No times when multiple classifiers output 1"]

    conflicitingClassificationsIncludingTheft = findConflictingClassifications(results, True)
    if len(conflicitingClassifications) > 0:
        csvRow += [TimeFileUtils.intervalsToString(conflicitingClassificationsIncludingTheft)]
    else:
        csvRow += ["No times when multiple classifiers output 1"]


def findConflictingClassifications(results, includeTheft):
    conflictingVal = 1
    conflicitingClassifications = []
    for classifier in results:
        if includeTheft or classifier != classifiers.THEFT_CLASSIFIER:
            intervals = results[classifier][1][conflictingVal]
            conflicitingClassifications = Intervals.findCommonIntervals(conflicitingClassifications, intervals)

    return conflicitingClassifications    

def totalTimeOfIntervals(intervals):
    timeConnected = datetime.timedelta(seconds=00)
    prevState = -1
    print("Calculating Total Time:")
    for interval, classified, state in intervals:
        start = interval[0]
        end = interval[1]
        timeInBetween = end - start
        print(str(start), str(end), str(timeInBetween))
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
def checkClassifications(actualIntervals, expectedIntervals=None):
    # However you want to load the expectedIntervals, maybe parse a text file?
    # Just make sure to load them as a list with each item formatted as ((startDateTime, endDateTime), classification)

    comparedIntervals, matchingIntervals, conflictingIntervals = Intervals.compareIntervals(actualIntervals, expectedIntervals)

    file = open('diary-study-stats' + DATA_DAY + NOW_TIME + '.txt', 'w+')

    file.write("############ DIARY STUDY COMPARISON ############## \n")

    interval1, classification1, ismatch1 = comparedIntervals[0]
    interval2, classification2, ismatch2 = comparedIntervals[-1]
    totalTime = interval2[1] - interval1[0]
    matchingTime = totalTimeOfIntervals(matchingIntervals)
    conflictingTime = totalTimeOfIntervals(conflictingIntervals)

    file.write("Total Time: " + TimeFileUtils.formatTimeValue(totalTime) + "\n")
    file.write("Total time matching: " + TimeFileUtils.formatTimeValue(matchingTime) +"\n")
    file.write("% of time matched: " + str(1.0 * matchingTime/totalTime) + "\n")
    file.write("Total time conflicting: " + TimeFileUtils.formatTimeValue(conflictingTime) + "\n")
    file.write("% of time conflicted: " + str(1.0 * conflictingTime/totalTime) + "\n")

    file.write("\n")
    file.write("\n")

    file.write("All conflicting intervals: \n")
    for interval, classificationString, isMatching in conflictingIntervals:
        file.write(TimeFileUtils.formatTimeValue(interval) + ": " + classificationString + "\n")

    file.close()

    # Write the results to some file, probably also calculate some stats on what % of time we match/don't match
    # All of comparedIntervals, matchingIntervals, and conflictingIntervals have the following format:
    # ((startDateTime, endDateTime), classificationString, isMatchingClassifications)
    # the classificationString is either one classifier if the expected/actual matched, else two classifier names


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

def getWindowStartTime(windowOfDataRows):
    return windowOfDataRows[0][0] #time of first data row in window

def compileRelevantSensors():
    for c in classifiers.CLASSIFIERS:
        classifier = classifiers.CLASSIFIERS[c]
        for sensor in classifier.getRelevantSensors():
            sensors.RELEVANT_SENSORS.add(sensor)

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
            columnHeaders += Intervals.getIntervalStatHeaders(c)

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
    start_time = TIMER.time()
    # print("Start:", start_time)

    NOW = datetime.datetime.now()
    NOW_TIME = NOW.strftime('_%m_%d_%H_%M')
    DIRECTORY_PATH = DIRECTORY

    # DATA_DAY = 'FULL_STUDY_RUN'
    # file = open('testing-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
    # watchFile = open('watch-testing-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
    # results = open('testing-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
    # watchResults = open('watch-testing-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
    # resultsSummary = open('testing-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
    # watchSummary = open('watch-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
    # resultsSummaryWriter = csv.writer(resultsSummary)
    # watchSummaryWriter = csv.writer(watchSummary)
    # resultsSummaryWriter.writerow(["Day","User", "Classifier", "Percentage of Time"])
    # watchSummaryWriter.writerow(["Day", "User", "State", "Percentage of Time", "Hours", "Total Hours"])
    # activatedFile = open('activated-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
    # activatedSummary = open('activated-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
    # activatedSummaryWriter = csv.writer(activatedSummary)
    # activatedSummaryWriter.writerow(["Day", "User", "Unlocks Saved", "Unlocks Total", "Percent Both Activated", "Percent Only Phone Activated", "Percent Only Watch Activated", "Percent Both Deactivated", "Both Activated", "Only Phone Activated", "Only Watch Activated", "Both Deactivated"])

    for DATA_DAY in DATA_DATES:
        print("DIRECTORY started as:", DIRECTORY)
        DIRECTORY = DIRECTORY_PATH + DATA_DAY + "/"
        print("DIRECTORY now:", DIRECTORY)
        # if not FULL_STUDY_RUN:
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
        activatedFile = open('activated-results-' + DATA_DAY + NOW_TIME + '.txt', 'w+')
        activatedSummary = open('activated-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
        activatedSummaryWriter = csv.writer(activatedSummary)
        activatedSummaryWriter.writerow(["Day", "User", "Unlocks Saved", "Unlocks Total", "Percent Both Activated", "Percent Only Phone Activated", "Percent Only Watch Activated", "Percent Both Deactivated", "Both Activated", "Only Phone Activated", "Only Watch Activated", "Both Deactivated"])

        smartUnlockSummary = open('smart-unlock-summary-' + DATA_DAY + NOW_TIME + '.csv', 'w+')
        smartUnlockSummaryWriter = csv.writer(smartUnlockSummary)
        smartUnlockFile = open('smart-unlock-log-' + DATA_DAY + NOW_TIME + '.txt', 'w+')

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
                    watchState, continousWatchState = stateFromWatchData(continuousWatchInterals(USER_ID), watchFile)
                    activatedWatchStates = watchActivationStates(continousWatchState)
                    activatedIntervalsWatch = {PossessionState.PHONE_ACTIVATED: activatedWatchStates[0], PossessionState.PHONE_DEACTIVATED: activatedWatchStates[1]}
                    # HERE STEVEN :)
                    timeSpentByWatchState = {}
                    # print(watchState)
                    for state in watchState:
                        watchResults.write("----" + str(state) + "-----" + "\n")
                        intervals = watchState[state]
                        # print("WTF!", state)
                        stats = Intervals.getIntervalStats(intervals)
                        for stat, val in stats.items():
                            watchResults.write(str(stat) + "\t\t\t" + str(TimeFileUtils.formatTimeValue(val)) + "\n")
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
                    
                    print("LOOK AT THE CLASSIFICATION INTERVALS")
                    for interval, classification in classifications:
                        print(TimeFileUtils.formatTimeInterval(interval), classification)

                    expectedIntervalsDiary = getExpectedIntervals(DIARY_STUDY_FILE)
                    ### Joanna Finish ###
                    checkClassifications(classifications, expectedIntervalsDiary)
                    #####################

                    activatedIntervalsPhone = possessionState.getIntervalsByState()
                    timeSpentByClass = {}

                    for c in intervalsByClass:
                        results.write("----" + str(c) + "-----\n")
                        intervals = intervalsByClass[c]
                        stats = Intervals.getIntervalStats(intervals)
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
                        resultsSummaryWriter.writerow([DATA_DAY, USER_ID, str(c), str(percentage * 100), TimeFileUtils.formatTotalSeconds(time), TimeFileUtils.formatTotalSeconds(totalTime)])

                    results.write("-----Classifications over Time-------\n")
                    for c in classifications:
                        interval = c[0]
                        duration = TimeFileUtils.formatTimeValue(interval[1] - interval[0])
                        classification = c[1]
                        intervalString = "(" + TimeFileUtils.formatTime(interval[0]) + "--" + TimeFileUtils.formatTime(interval[1]) + "); "
                        results.write(intervalString + ' ' + duration + '; ' + str(classification) + "\n")
                except:
                    tb = traceback.format_exc()
                    print(tb)
                    results.write("******EXCEPTION (while computing classifications)*******\n")
                    results.write(tb)
                    results.write("\n")

            if activatedIntervalsPhone != None and CALCULATE_ACTIVATIONS:
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
                        smartUnlockFile.write(TimeFileUtils.formatTimeValue(time) + "\n")

                    smartUnlockFile.write("######## SAVED UNLOCK TIMES (BLE) #######\n")

                    for time in unlockTimesBle:
                        smartUnlockFile.write(TimeFileUtils.formatTimeValue(time) + "\n")

                except:
                    tb = traceback.format_exc()
                    print(tb)
                    smartUnlockFile.write("******EXCEPTION (while computing activations)*******\n")
                    smartUnlockFile.write(tb)
                    smartUnlockFile.write("\n")


            try:
                activatedFile.write("***********" + USER_ID + "*************" + "\n")
                if activatedIntervalsWatch == None or activatedIntervalsPhone == None:
                    activatedFile.write("Check: " + 'watch-testing-results-' + DATA_DAY + NOW_TIME + '.txt')

                else:
                    # print("********Finding both activated**********")
                    # bothActivated = Intervals.findCommonIntervals(activatedIntervalsPhone["activated"], activatedIntervalsWatch["activated"])
                    # print("********Finding both deactivated**********")
                    # bothDeactivated = Intervals.findCommonIntervals(activatedIntervalsPhone["deactivated"], activatedIntervalsWatch["deactivated"])
                    # print("********Finding phone activated**********")
                    # onlyPhoneActivated = Intervals.findCommonIntervals(activatedIntervalsPhone["activated"], activatedIntervalsWatch["deactivated"])
                    # print("********Finding watch activated**********")
                    # onlyWatchActivated = Intervals.findCommonIntervals(activatedIntervalsPhone["deactivated"], activatedIntervalsWatch["activated"])
                    
                    # activatedRow = [DATA_DAY, USER_ID, numUnlocksSaved, numUnlocksTotal]
                    ### CALCULATE UNLOCKS ###

                    unlockData = possessionState.unlockData
                    activatedIntervals = activatedIntervalsPhone["activated"]
                    # pickle.dump(unlockData, open( "unlock_test_data.pkl", "wb" ) )
                    # pickle.dump(activatedIntervals, open( "unlock_test_intervals.pkl", "wb" ) )

                    numUnlocksSaved, numUnlocksTotal, unlockTimes = computeUnlocks(unlockData, activatedIntervals)
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
                            if stateP == "activated" and stateW == "deactivated":
                                print("WTF PHONE ACTIVATED")
                                for interval in activatedIntervalsPhone[stateP]:
                                    print(TimeFileUtils.formatTimeInterval(interval))
                                print("WTF WATCH DEACTIVATED")
                                for interval in activatedIntervalsPhone[stateW]:
                                    print(TimeFileUtils.formatTimeInterval(interval))

                            state = "Phone: " + stateP + " Watch: " + stateW
                            print(state)
                            # activatedFile.write(str(state) + '\n')
                            # print("Phone Intervals:", activatedIntervalsPhone[stateP])
                            # print("Watch Intervals:", activatedIntervalsWatch[stateW])
                            commonIntervals = Intervals.findCommonIntervals(activatedIntervalsPhone[stateP], activatedIntervalsWatch[stateW])
                            # print("COMMON INTERVALS:", commonIntervals)
                            stats = Intervals.getIntervalStats(commonIntervals)
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

                        datum = "Phone " + stateP + "\t" + str(percentage1 * 100)[:6] + '%' + "\t\t" + str(percentage2 * 100)[:6] + '%' + "\n"
                        times = " " * 20 + "\t" + str(stateTimes[state1])[:7] + "\t\t" + str(stateTimes[state2])[:7] + "\n"
                        
                        print(state, "-->", percentage)
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
                        activatedFile.write(TimeFileUtils.formatTimeValue(time) + "\n")

                    activatedFile.write("######## PHONE INTERVALS #######\n")
                    for state, intervals in activatedIntervalsPhone.items():
                        activatedFile.write("#########" + str(state).upper() + "########" + "\n")
                        for interval in intervals:
                            activatedFile.write(TimeFileUtils.formatTimeInterval(interval) + "\n")

                    activatedFile.write("######## WATCH INTERVALS #######\n")
                    for state, intervals in activatedIntervalsWatch.items():
                        activatedFile.write("#########" + str(state).upper() + "########" + "\n")
                        for interval in intervals:
                            activatedFile.write(TimeFileUtils.formatTimeInterval(interval) + "\n")
            except:
                tb = traceback.format_exc()
                print(tb)
                activatedFile.write("******EXCEPTION (while computing activations)*******\n")
                activatedFile.write(tb)
                activatedFile.write("\n")
                    

        if not FULL_STUDY_RUN:
            file.close()
            watchFile.close()
            results.close()
            watchResults.close()
    # print("Start:", start_time)
    # print("Time:", time.time())

    print("--- %s seconds ---" % (TIMER.time() - start_time))
    print("Yay I finished!")

