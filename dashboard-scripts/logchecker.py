import csv
import os.path as Path 
import glob
import re
import datetime
import sys

# where to save output
DASHBOARDDIR = "/home/daw/Dropbox/phone_data/Dashboard_results/"

# assumes that there exists a CSV file that maps user to uniqueID
USERS_TO_IDS_FILE = "/home/daw/Dropbox/phone_data/Dashboard_results/users_to_ids.csv"

# Install pycrypto and pycryptodome to use this script
FILES_DECRYPTED = True

global DIRECTORY
DIRECTORY = None

WATCH_CONNECTED_INSTRUMENT = 'TriggeredBleDevices'
WATCH_WORN_INSTRUMENT = 'BleHrm'
PHONE_INSTRUMENT = 'BatchedAccelerometer'

NO_DATA_FOUND = "Instrument Data not Found"

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

def getMostRecentDataTimeAndTimes(userID, instrument):
    userDateTimes = getUserFileTimes(userID, instrument, FILES_DECRYPTED)
    if (len(userDateTimes) == 0):
        return (NO_DATA_FOUND, [])
    userDateTimes.sort()
  
    return (userDateTimes[len(userDateTimes) - 1], userDateTimes)

def getMostRecentMatchingTime(timesList):
    matchingTimes = set(timesList[0])
    for i in range(1, len(timesList)):
        matchingTimes = matchingTimes.intersection(set(timesList[i]))
    if len(matchingTimes) > 0:
        return min(matchingTimes)
    else:
        return None

def getEmptyTimeStats():
    timesStats = {}
    timesStats["gaps"] = "N/A"
    timesStats["max_gap"] = "N/A"
    timesStats["max_gap_start"] = "N/A"
    timesStats["max_gap_end"] = "N/A"
    timesStats["median_gap"] = "N/A"
    timesStats["earliest_time"] = NO_DATA_FOUND
    timesStats["latest_time"] = NO_DATA_FOUND
    timesStats["period"] = "N/A"
    timesStats["times_count"] = 0
    timesStats["times_per_period"] = "N/A"
    return timesStats

def timesToStats(times):
    if len(times) == 0:
        return getEmptyTimeStats()
    numTimes = len(times)
    numGaps = numTimes - 1
    gapsMap = {}
    largestGap = datetime.timedelta(0)
    medianGap = datetime.timedelta(0)
    earliestTime = times[0]
    latestTime = times[-1] if numTimes > 1 else earliestTime
    period = latestTime - earliestTime
    largestGapStart = times[0]
    largestGapEnd = times[1]

    
    for i in range(numTimes - 1):
        time0 = times[i]
        time1 = times[i + 1]
        gap = time1 - time0
        gapsMap[(time0, time1)] = gap
        if gap > largestGap:
            largestGap = gap
            largestGapStart = time0
            largestGapEnd = time1
        
        if i == numGaps // 2:
            medianGap = gap

    timesStats = {}
    timesStats["gaps"] = gapsMap
    timesStats["max_gap"] = timeDiffToHours(largestGap)
    timesStats["max_gap_start"] = largestGapStart
    timesStats["max_gap_end"] = largestGapEnd
    timesStats["median_gap"] = timeDiffToHours(medianGap)
    timesStats["earliest_time"] = earliestTime
    timesStats["latest_time"] = latestTime
    timesStats["period"] = timeDiffToHours(period)
    timesStats["times_count"] = numTimes
    timesStats["times_per_period"] = float(numTimes) / (period.days * 24 + period.seconds // 3600) if period.seconds >= 3600 else numTimes

    return timesStats

def getTimeStats(userID, instrument):
    times = getUserFileTimes(userID, instrument, FILES_DECRYPTED)
    stats = timesToStats(times)
    return [stats["max_gap"], stats["max_gap_start"], stats["max_gap_end"], stats["median_gap"], stats["earliest_time"],
            stats["latest_time"], stats["period"], stats["times_count"], stats["times_per_period"]]



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

def timeDiffToHours(timeDiff):
    totalSeconds = timeDiff.seconds
    hours = totalSeconds // 3600
    totalSeconds = totalSeconds - hours * 3600
    minutes = totalSeconds // 60
    totalSeconds = totalSeconds - minutes * 60
    seconds = totalSeconds
    if timeDiff.days > 0:
        return "{} days, {} hours, {}:{}".format(timeDiff.days, hours, minutes, seconds)

    return "{} hours, {}:{}".format(hours, minutes, seconds)


def getMaxWatchGapAllowed():
    return datetime.timedelta(hours=MAX_WATCH_GAP_ALLOWED_HOURS)

#### End Time Utilities ####

# Gets all filenames belonging to certain user 
def getUserFiles(userID, instrument):
    query = DIRECTORY + 'AppMon' + '_' + userID + '*_' + instrument + '_*'
    userFiles = glob.glob(query)
    # validateFiles(userFiles)
    
    return userFiles

def getLastTimeWatchWasWornAndConnected(userID):
    data = getMostRecentDataTimeAndTimes(userID, WATCH_WORN_INSTRUMENT)
    lastBleHrmTime, bleHrmTimes = data[0], data[1] 
    data = getMostRecentDataTimeAndTimes(userID, WATCH_CONNECTED_INSTRUMENT)
    lastBleConnectedTime, bleConnectedTimes = data[0], data[1]
    

    if lastBleHrmTime == NO_DATA_FOUND and lastBleConnectedTime == NO_DATA_FOUND:
        return "No Watch Data found, check with participant on watch connection and wearing."
    elif lastBleHrmTime == NO_DATA_FOUND:
        return "No {} data found, participant has been connected, but check if participant has been wearing watch.".format(WATCH_WORN_INSTRUMENT)
    elif lastBleConnectedTime == NO_DATA_FOUND:
        return "No {} data found, check if participant has been connecting to watch.".format(WATCH_CONNECTED_INSTRUMENT)

    if lastBleHrmTime == lastBleConnectedTime:
        return lastBleConnectedTime

    mostRecentMatchingTime = getMostRecentMatchingTime([bleHrmTimes, bleConnectedTimes])
    if mostRecentMatchingTime == None:
        return "{} and {} never uploaded at same time. Something may be wrong with watch, check data.".format(WATCH_WORN_INSTRUMENT, WATCH_CONNECTED_INSTRUMENT)
    else:
        
        return mostRecentMatchingTime


def getMostRecentWatchDataTime(userID):
    watchConnectedTime = getLastTimeWatchConnected(userID)
    watchWornTime = getLastTimeWatchWorn(userID)
    if watchConnectedTime == NO_DATA_FOUND and watchWornTime == NO_DATA_FOUND:
        return "No Watch Data found, check with participant on watch connection and wearing."
    elif watchConnectedTime == NO_DATA_FOUND:
        return watchWornTime
    elif watchWornTime == NO_DATA_FOUND:
        return watchConnectedTime

    if watchConnectedTime > watchWornTime:
        return watchConnectedTime
    else:
        return watchWornTime

def getLastTimeWatchConnected(userID):
    return getMostRecentDataTime(userID, WATCH_CONNECTED_INSTRUMENT)

def getLastTimeWatchWorn(userID):
    return getMostRecentDataTime(userID, WATCH_WORN_INSTRUMENT)

def getLastPhoneDataFileTime(userID):
    return getMostRecentDataTime(userID, PHONE_INSTRUMENT)


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
    since = timeDiffToHours(now - time)
    return [time, since]

def main():
    if len(sys.argv) < 2:
        print("Usage: logchecker.py <directory containing decrypted .csv files>")
        sys.exit()
    global DIRECTORY
    DIRECTORY = sys.argv[1] + "/"
    

    now = datetime.datetime.now()
    dashboardFileName = DASHBOARDDIR + "Dashboard-" + now.strftime('%Y_%m_%d') + ".csv"

    dashboardFile = open(dashboardFileName, 'wb')
    dashboardWriter = csv.writer(dashboardFile, delimiter = ',')

    columnHeaders = ["User ID", 
                     "Most Recent Accel. Data", "Time Since", 
                     "Last time Watch was worn", "Time Since", 
                     "Last time Watch was connected", "Time Since", 
                     "Last time of any Watch Data", "Time Since",
                     "Largest Watch Worn Data Gap", "Start Time of Gap", "End Time of Gap", "Median Watch Worn Data Gap",
                     "Earliest Watch Worn Data Time", "Latest Watch Worn Data Time",
                     "Period of Watch Worn Data", "Number of Data Files", "Data Files per Hour",
                     "Largest Watch Connected Data Gap",  "Start Time of Gap", "End Time of Gap", "Median Watch Connected Data Gap",
                     "Earliest Watch Connected Data Time", "Latest Watch Connected Data Time",
                     "Period of Watch Connected Data", "Number of Data Files", "Data Files per Hour",
                     "Largest Phone Data Gap",  "Start Time of Gap", "End Time of Gap", "Median Phone Data Gap",
                     "Earliest Phone Data Time", "Latest Phone Data Time",
                     "Period of Phone Data", "Number of Data Files", "Data Files per Hour"]

    dashboardWriter.writerow(columnHeaders)
    for userID in USERS:
        mostRecentAccelTime = getMostRecentAccelerometerFileTime(userID)
        mostRecentWatchWornTime = getLastTimeWatchWasWornAndConnected(userID)
        mostRecentWatchConnectedTime = getLastTimeWatchConnected(userID)
        mostRecentWatchDataAny = getMostRecentWatchDataTime(userID)
        datarow = [userID]
        datarow += getTimeAndTimeSince(mostRecentAccelTime) 
        datarow += getTimeAndTimeSince(mostRecentWatchWornTime)
        datarow += getTimeAndTimeSince(mostRecentWatchConnectedTime)
        datarow += getTimeAndTimeSince(mostRecentWatchDataAny)
        datarow += getTimeStats(userID, WATCH_WORN_INSTRUMENT)
        datarow += getTimeStats(userID, WATCH_CONNECTED_INSTRUMENT)
        datarow += getTimeStats(userID, PHONE_INSTRUMENT)
        dashboardWriter.writerow(datarow)
    print("Dashboard results generated in: " + dashboardFileName)

if __name__ == '__main__':
    main()

