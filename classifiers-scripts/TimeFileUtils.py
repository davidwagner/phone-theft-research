import datetime
import glob
import re

NOW = datetime.datetime.now()
# NOW_DAY = NOW.strftime('%Y_%m_%d')

YESTERDAY = (NOW - datetime.timedelta(days=1)).strftime('%Y_%m_%d')
# NOW_DAY = YESTERDAY
NOW_DAY = '2016_11_01'

YEAR_2000 = datetime.date(2000, 1, 1)


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

def formatTimeValue(timeValue):
    if type(timeValue) is str or type(timeValue) is int:
        return str(timeValue) 
    if type(timeValue) is datetime.datetime:
        return formatTime(timeValue)
    elif type(timeValue) is datetime.timedelta:
        return formatTimeDelta(timeValue)
    else:
        # must be an interval
        return formatTimeInterval(timeValue)


def getTimeFromFile(filename, userID, instrument):
    query = '*AppMon' + '_' + userID + '.*_' + instrument + '_' + \
        '(?P<time>.*)' + '_.csv'
    match = re.match(query, filename)
    print("MATCH:", match.group('time'))
    return match.group('time')

def getTimeFromFile(filename):
    query = 'AppMon' + '_*_' + '*_' + \
        '(?P<time>.*)' + '_.csv'
    match = re.match(query, filename[filename.find('AppMon'):])
    time = match.group('time')[-19:]
    print("MATCH:", time)
    
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