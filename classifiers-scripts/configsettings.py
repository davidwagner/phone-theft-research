import csv
##### CONFIGURE FOR OWN MACHINE ###############
# Replace with your own
DASHBOARDDIR = '../data/Classifier_Results_TEST/'
USERS_TO_IDS_FILE = "../data/users_r3_watch.csv"
DIRECTORY = "../data/Decrypted_Data/"
# DIRECTORY = "../data/Decrypted_Data/2016_11_01/"
###############USERS###########################
READ_USERS_FROM_FILE = True
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
	USERS = []
###############DAYS###########################
DATA_DATES = ["2016_12_13"]