import csv
import os
import glob
import re
import datetime
import sys
import shutil
# import classifier
import Sensors as sensors
import Classifiers as classifiers 

NOW = datetime.datetime.now()
NOW_DAY = NOW.strftime('%Y_%m_%d')
# NOW_DAY = '2016_09_28'


# Replace with your own
# DATA_DIR = "/home/daw/Dropbox/phone_data/Dashboard_results/"
DASHBOARDDIR = './Classifier_Results/'
DATA_DIR = "../../../Dropbox/phone_data/Sensor_Research/"

USERS_TO_IDS_FILE = "./users_to_ids.csv"
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


def getUserFilesByDayAndInstrument(userID, instrument):
    query = DECRYPTED_DIR + '/' + NOW_DAY + '/' + 'AppMon_' + userID + '*_' + instrument + '_' + NOW_DAY + '*'
    userFiles = glob.glob(query)

    # TODO: Need to filter for sensors that need data files with matching times as other
    # sensors (e.g. accelerometer and step count for Theft Classifier)
    
    return userFiles


def dataFilesToDataList(userFiles):
    dataList = []
    for dataFile in userFiles:
        with open(dataFile) as f:  
            reader = csv.reader(f)
            for row in reader:
                row[0] = convertToDateTime(row[0])
                dataList.append(row)
    return dataList

def getRelevantUserData(userID):
    userData = {}

    for instrument in sensor.RELEVANT_SENSORS:
        dataFiles = getUserFilesByDayAndInstrument(userID, instrument)
        userData[instrument] = dataFilesToDataList(dataFiles)

    return userData 


def runClassifier(classifier, userData):
    windowSize = classifier.getWindowSize()
    instruments = classifier.getRelevantSensors()
    
    numRows = min([len(userData[instrument]) for instrument in instruments])
    
    # {instrument: windowOfRawData}
    results = {}
    for row in range(0, numRows - windowSize):
        windowOfData = {}
        for instrument in instruments:
            data = userData[instrument][row:row + windowSize] 
            windowOfData[instrument] = data
            windowStartTime = getWindowStartTime(data)
        results[windowStartTime] = classifierFunction.classify(windowOfData)
    return results


def runClassifiersOnUser(userID, csvWriter):
    userData = getRelevantUserData(userID)
    csvRow = [userID]

    for classifier in classifiers.CLASSIFIERS:
        results = runClassifier(classifier, userData)
        processResults(results, csvWriter, csvRow)

# TODO:
def processResults(results, writer, csvrow):
    # analyze results
    # write actionable output to writer
    return 0

###### Utilities #######

def filesToTimesToFilesDict(files, userID, instrument):
    timesToFiles = {}
    for f in files:
        time = getTimeFromFile(f, userID, instrument, True)
        timesToFiles[time] = f 
    return timesToFiles


def timeStringsToDateTimes(timeStrings):
    convert = lambda time : datetime.datetime.strptime(time, '%Y_%m_%d_%H_%M_%S')
    return [convert(timeString) for timeString in timeStrings]

def formatTime(dateTime):
    return dateTime.strftime('%H:%M:%S %m-%d-%Y')

def getTimeFromFile(filename, userID, instrument, isDecrypted):
    query = DIRECTORY + 'AppMon' + '_' + userID + '.*_' + instrument + '_' + \
        '(?P<time>.*)' + getFileExtension(isDecrypted)
    match = re.match(query, filename)
    return match.group('time')

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

# TODO: 
def convertToDateTime(timestring, boottime):
    # Converts AppMon timestamp to DateTime
    # Checks if timestamp is epoch or time since boot
    return 0 

# TODO:
def getWindowStartTime(dataRows):
    return windowOfDataRows[0][0] #time of first data row in window



def main():

    now = datetime.datetime.now()
    dashboardFileName = DASHBOARDDIR + "Dashboard-" + now.strftime('%Y_%m_%d_%H_%M') + ".csv"

    dashboardFile = open(dashboardFileName, 'wb')
    dashboardWriter = csv.writer(dashboardFile, delimiter = ',')

    columnHeaders = ["User ID", 
                     ]

    dashboardWriter.writerow(columnHeaders)

    for userID in USERS:
        datarow = [userID]
        runClassifiersOnUser(userID, dashboardWriter)

    print("Dashboard results generated in: " + dashboardFileName)

if __name__ == '__main__':
    main()

