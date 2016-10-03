import csv
import os.path as Path 
import glob
import re
import datetime
import sys
#Assumes this is called in a folder that has that day's decrypted data.
#Also assumes that there exists a CSV file that maps user to uniqueID

USERS_TO_IDS_FILE = '/home/daw/Dropbox/phone_data/Dashboard_results/users_to_ids.csv'
###########################
""" If you have pycrypto installed, uncomment these lines and comment the lines below """
FILES_DECRYPTED = True
DIRECTORY = "/tmp/dashboards/" + datetime.datetime.now().strftime('%Y_%m_%d') + "/"
""" Otherwise leave these uncommented """
# FILES_DECRYPTED = False
# DIRECTORY = "./Dashboard_Results/" + sys.argv[1] + "/" + "encrypted/"
############################

NO_DATA_FOUND = "Instrument Data not Found"
#DIRECTORY = "./0912_pull/"

BLANK_ROWS_THRESHHOLD = 20
USERS = {}
usersToIdsFile = open(USERS_TO_IDS_FILE, 'rU')
try:
    reader = csv.reader(usersToIdsFile)
    for row in reader:
        user = row[0]
        userID = row[1]
        USERS[user] = userID
finally:
    usersToIdsFile.close()

uniqueIDs = set()


#Looks through folder and checks to see there are enough files
def checkFileExist():
    for user in USERS:
        if not userHasMinNumFiles(user, minNum):
            return False
    return True

def userHasMinNumFiles(user, minNum):
    files = getUserFiles(user)
    return len(files) >= minNum

# Gets all times that user submitted data for that day
def getUserFileTimes(userID, instrument, isDecrypted):
    userFiles = getUserFiles(userID, instrument)
    userTimes = [getTimeFromFile(filename, userID, instrument, isDecrypted) for filename in userFiles]
    userDateTimes = timeStringsToDateTimes(userTimes)
    #print(userDateTimes)
    # userDateTimes.sort()

    # print([formatTime(datetime) for datetime in userDateTimes])
    return userDateTimes

def getMostRecentDataTime(userID, instrument):
    userDateTimes = getUserFileTimes(userID, instrument, FILES_DECRYPTED)
    if (len(userDateTimes) == 0):
        return NO_DATA_FOUND
    maxTime = userDateTimes[0]
    for time in userDateTimes:
        if time > maxTime:
            maxTime = time

    return maxTime

### Time Utilities ###
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

#### End Time Utilities ####

# Gets all filenames belonging to certain user 
def getUserFiles(userID, instrument):
    query = DIRECTORY + 'AppMon' + '_' + userID + '*_' + instrument + '_*'
    #print(query)
    userFiles = glob.glob(query)
    # validateFiles(userFiles)
    #print(userFiles)
    return userFiles

def getLastTimeWatchWasWornAndConnected(userID):
    lastBleHrmTime = getMostRecentDataTime(userID, 'BleHrm')
    lastBleConnectedTime = getMostRecentDataTime(userID, 'TriggeredBleConnectedDevices')

    if lastBleHrmTime == NO_DATA_FOUND and lastBleConnectedTime == NO_DATA_FOUND:
        return "No Watch Data found."
    elif lastBleHrmTime == NO_DATA_FOUND:
        return "Watch never worn, no BleHrm data found."
    elif lastBleConnectedTime == NO_DATA_FOUND:
        return "Watch never connected, no TriggeredBleConnectedDevices"

    if lastBleHrmTime == lastBleConnectedTime:
        return lastBleConnectedTime
    elif lastBleHrmTime < lastBleConnectedTime:
        return lastBleHrmTime
    else:
        return "Watch worn but not connected?"

def getMostRecentWatchDataTime(userID):
    watchConnectedTime = getLastTimeWatchConnected(userID)
    watchWornTime = getLastTimeWatchWorn(userID)
    if watchConnectedTime == NO_DATA_FOUND and watchWornTime == NO_DATA_FOUND:
        return "No Watch Data found."
    elif watchConnectedTime == NO_DATA_FOUND:
        return watchWornTime
    elif watchWornTime == NO_DATA_FOUND:
        return watchConnectedTime

    if watchConnectedTime > watchWornTime:
        return watchConnectedTime
    else:
        return watchWornTime

def getLastTimeWatchConnected(userID):
    return getMostRecentDataTime(userID, 'TriggeredBleConnectedDevices')

def getLastTimeWatchWorn(userID):
    return getMostRecentDataTime(userID, 'BleHrm')

def getMostRecentAccelerometerFileTime(userID):
    return getMostRecentDataTime(userID, 'BatchedAccelerometer')


def userToDataFileName(directory, userID, instrument):
    return directory + 'AppMon' + '_' + userID + '_' + instrument + '_'

def fileExists(filename):
    return Path.isfile(filename)


### File Contents validation ###
def validateFiles(userFiles):
    for filename in userFiles:
        f = open(filename, 'r')
        try:
            reader = csv.reader(f)
            blankRows = 0
            for index, row in enumerate(reader):
                if index > 20:
                    break
                if isRowBlank(row):
                    blankRows += 1
            if blankRows < 20:
                userFiles.remove(filename)
        finally:
            f.close()


# Row doesn't just have zeroes
def isRowComplete(row, expectedNumColumns):
    for i in range(expectedNumColumns):
        columnData = row[i]
        if len(columnData) == 0:
            return False
    return True


def findIncompleteRows(logfile, expectedNumColumns):
    f = open(logfile, 'r')
    # expectedNumColumns = 4
    try:
        reader = csv.reader(f)

        incompleteRows = set([])
        count = 0
        for index, row in enumerate(reader):
            if count > 10:
                break
            count += 1
            if not isRowComplete(row, expectedNumColumns):
                incompleteRows.add(index)
            print(incompleteRows)
    finally:
        f.close()

#Checks logs and ensures there is enough data
#Maybe we need a unique function for each kind of log since we're looking for different things???
def lookThroughLog():
    return 0

def getTimeAndTimeSince(time):
    if isinstance(time, str):
        return [time, "N/A"]
    now = datetime.datetime.now()
    return [time, now - time]

def main():
    now = datetime.datetime.now()
    # dashboardDir = "./" + sys.argv[1] + "/Dashboard_Results/"
    dashboardFileName = "/home/daw/Dropbox/phone_data/Dashboard_results/Dashboard-" + sys.argv[1] + ".csv" 

    dashboardFile = open(dashboardFileName, 'wb')
    dashboardWriter = csv.writer(dashboardFile, delimiter = ',')

    columnHeaders = ["USER", "ID", "Most Recent Accel. Data", "Time Since", "Last time Watch was worn", "Time Since", "Last time Watch was connected", "Time Since", "Last time of any Watch Data", "Time Since"]

    dashboardWriter.writerow(columnHeaders)
    for user, userID in USERS.iteritems():
        mostRecentAccelTime = getMostRecentAccelerometerFileTime(userID)
        mostRecentWatchWornTime = getLastTimeWatchWasWornAndConnected(userID)
        mostRecentWatchConnectedTime = getLastTimeWatchConnected(userID)
        mostRecentWatchDataAny = getMostRecentWatchDataTime(userID)
        datarow = [user, userID]
        datarow += getTimeAndTimeSince(mostRecentAccelTime) 
        datarow += getTimeAndTimeSince(mostRecentWatchWornTime)
        datarow += getTimeAndTimeSince(mostRecentWatchConnectedTime)
        datarow += getTimeAndTimeSince(mostRecentWatchDataAny)
        dashboardWriter.writerow(datarow)
    print("Dashboard results generated in: " + dashboardFileName)

if __name__ == '__main__':
    main()

