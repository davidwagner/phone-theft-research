import csv
import os
import glob
import re
import datetime
import sys
import shutil
# import classifier
import TableFeaturizer


PYTHON = 'python2.7'
THEFT_FEATURIZER = './classifiers/theft_detector.py'
DECRYPTOR_PATH = 'bintocsv.py'


NOW = datetime.datetime.now()
# NOW_DAY = NOW.strftime('%Y_%m_%d')
NOW_DAY = '2016_09_28'
ACCELEROMETER = 'BatchedAccelerometer'
STEP_COUNT = 'BatchedStepCount'

# Replace with your own
# DATA_DIR = "/home/daw/Dropbox/phone_data/Dashboard_results/"
DASHBOARDDIR = './Classifier_Results/'
DATA_DIR = "../../../Dropbox/phone_data/Sensor_Research/"
THEFT_FILES_DIR = ''
DECRYPTED_DIR = './Decrypted_Data'
ENCRYPTED_DIR = './Encrypted_Data'
PRIVATE_KEY_PATH = "./ucb_keypair/ucb.privatekey"

RELEVANT_INSTRUMENTS = [ACCELEROMETER]

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
    # query = DECRYPTED_DIR + '/' + 'AppMon' + '_' + userID + '*_' + instrument + '_' + NOW_DAY +'_*'
    query = DECRYPTED_DIR + '/' + NOW_DAY + '/' + 'AppMon_' + userID + '*_' + instrument + '_' + NOW_DAY + '*'
    print(query)
    userFiles = glob.glob(query)
    print("instrument")
    print(userFiles)
    
    return userFiles



### THEFT CLASSIFIER ####
def runTheftClassifierOnUser(userID, resultsRow):
    removeFilesFromDir(THEFT_FILES_DIR)

    featurizerCommand = PYTHON + ' ' + THEFT_FEATURIZER
    
    pullTheftFiles(userID, THEFT_FILES_DIR)
    # run featurizer
    os.system(featurizerCommand)
    # run classifier
    classifierResults = classifier.classify()

    accuracy = classifierResults[accuracy]
    resultsRow += [accuracy]

def pullTheftFiles(userID, directory):
    theftFiles = getTheftFiles(userID)
    for f in theftFiles:
        shutil.copy2(f, directory)

def getTheftFiles(userID):
    accelerometerFiles = getUserFilesByDayAndInstrument(userID, ACCELEROMETER)
    stepCountFiles = getUserFilesByDayAndInstrument(userID, STEP_COUNT)

    matches = filesToTimesToFilesDict(accelerometerFiles, userID, ACCELEROMETER).viewkeys() & filesToTimesToFilesDict(accelerometerFiles, userID, STEP_COUNT).viewkeys() 
    
    theftFiles = []
    for match in matches:
        accelFile = accelerometerFiles[match]
        stepCountFile = stepCountFiles[match]
        theftFiles.append(accelFile)
        theftFiles.append(stepCountFile)
    return theftFiles

### TABLE CLASSIFIER ###
def runTableClassifierOnUser(userID, resultsRow):
    allTimes = []
    tableFiles = getTableFiles(userID)
    print("########")
    print(userID)
    print("########")
    for f in tableFiles:
        tableTimes = TableFeaturizer.classify(f)
        
        print(tableTimes)
        for time in tableTimes:
            allTimes.append(time)

    resultsRow += [replaceCommasWithSemicolons(str(allTimes))]

def getTableFiles(userID):
    accelerometerFiles = getUserFilesByDayAndInstrument(userID, ACCELEROMETER)

    return accelerometerFiles



### Decryption #####
def decryptRelevantInstrumentData():
    removeFilesFromDir(ENCRYPTED_DIR)
    removeFilesFromDir(DECRYPTED_DIR)
    for instrument in RELEVANT_INSTRUMENTS:
        # query = DATA_DIR + 'AppMon' + '_' + '*_' + instrument + '_' + NOW_DAY +'_*'
        # '../../../Dropbox/phone_data/Sensor_Research/AppMon_*_BatchedAccelerometer' + '_' + '2016_09_28' + '*'
        query = DATA_DIR + 'AppMon_*_' + instrument + '_' + NOW_DAY + '*'
        print(query)
        userFiles = glob.glob(query)
        print("ACCEL")
        print(userFiles)
        
        for f in userFiles:
            shutil.copy2(f, ENCRYPTED_DIR)

    decrypt(ENCRYPTED_DIR, DECRYPTED_DIR + "/" + NOW_DAY)


def decrypt(encryptedPath, destPath):

    command = PYTHON + " " + DECRYPTOR_PATH + " " + PRIVATE_KEY_PATH + " -s " + \
    encryptedPath  + " -d " + destPath + " -l verb"

    print("Running command: " + command)
    os.system(command)

    print("Decryption successful")



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




def main():
    # if len(sys.argv) < 2:
    #     print("Usage: logchecker.py <directory containing decrypted .csv files>")
    #     sys.exit()
    # global DIRECTORY
    # DIRECTORY = sys.argv[1] + "/"



    # decryptRelevantInstrumentData()
    

    now = datetime.datetime.now()
    dashboardFileName = DASHBOARDDIR + "Dashboard-" + now.strftime('%Y_%m_%d') + ".csv"

    dashboardFile = open(dashboardFileName, 'wb')
    dashboardWriter = csv.writer(dashboardFile, delimiter = ',')

    columnHeaders = ["User ID", 
                     "Table Times"
                     ]

    dashboardWriter.writerow(columnHeaders)

    for userID in USERS:
        datarow = [userID]
        runTableClassifierOnUser(userID, datarow)
        # runTheftClassifierOnUser(userID, datarow)

        dashboardWriter.writerow(datarow)
    print("Dashboard results generated in: " + dashboardFileName)

if __name__ == '__main__':
    main()

