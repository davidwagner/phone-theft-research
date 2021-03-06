import csv
##### CONFIGURE FOR OWN MACHINE ###############
# Replace with your own
DASHBOARDDIR = '../data/Classifier_Results_TEST/'
USERS_TO_IDS_FILE = "../joanna_watch.csv"
DIRECTORY = "../data/diary-study-11-14-15"
DIARY_STUDY_FILE = "../data/diary-study-11-14-15/diary_state.txt"
# DIRECTORY = "../data/Decrypted_Data/2016_11_01/"
###############USERS###########################
FILTER_ONLY_CONSISTENT_DATA = False
FULL_STUDY_RUN = False
RUN_WATCH_ONLY = False
RUN_CLASSIFIERS_ONLY = True
DIARY_STUDY = True
READ_USERS_FROM_FILE = False
USERS = []
if READ_USERS_FROM_FILE:
	ids = []
	usersToIdsFile = open(USERS_TO_IDS_FILE, 'rU')
	try:
	    reader = csv.reader(usersToIdsFile)
	    for row in reader:
	        userID = row[1]
	        ids.append(userID)
	finally:
	    usersToIdsFile.close()

	USERS = ids
else:
	USERS = ["d792b61e"]
###############DAYS###########################
DATA_DATES = ["diary_data"]