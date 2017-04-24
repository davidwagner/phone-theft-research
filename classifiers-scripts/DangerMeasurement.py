import ClassifierLog as cl 
import Sensors as sensors
from configsettings import *
import datetime

def stats(intervals):
	firstStart, firstEnd, s = intervals[0]
	lastStart, lastEnd, q = intervals[-1]
	totalTime = lastEnd - firstStart

	timeConnected = datetime.timedelta(seconds=00)
	timeDisconnected = datetime.timedelta(seconds=0)
	prevState = -1
	for start, end, state in intervals:
		timeInBetween = end - start
		timeConnected += timeInBetween
		if prevState != -1:
			timeDisconnected += start - prevState
		prevState = end

	return timeConnected/totalTime , timeDisconnected/totalTime, totalTime

def continousIntervals(userData):
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

if __name__ == '__main__':
	count = 0
	NOW = datetime.datetime.now()
	NOW_TIME = NOW.strftime('_%m_%d_%H_%M')
	summaryFile = open('bt-connected-summary-' + DATA_DATES[0] + '_' + NOW_TIME +'.txt', 'w+')
	summaryWriter = csv.writer(summaryFile)
	summaryWriter.writerow(["User", "BT Connection Changed", "Percent Dropped", "Percent Connected", "Phone Model"])
	allUsers = []
	for USER_ID in USERS:
		count += 1
		print("Number of users processed:", count)
		print("Currently on:", USER_ID)
		cl.DIRECTORY = DIRECTORY + "/" + DATA_DATES[0] + "/"
		try :
			dataFiles = cl.getUserFilesByDayAndInstrument(USER_ID, sensors.CONNECTED_DEVICES)
			userData = cl.dataFilesToDataListAbsTime(dataFiles)
			#print(userData)

			intervals = continousIntervals(userData)
			freq = len(intervals) - 1
			percentConnected, percentDropped, totalTime = stats(intervals)
			phoneModel = ""
			allUsers.append((USER_ID, freq, percentDropped, percentConnected, phoneModel))

		except:
			tb = traceback.format_exc()
			print(tb)
			summaryFile.write("******EXCEPTION (while computing " + USER_ID + " )*******\n")
			summaryFile.write(tb)
	sortedUsers = sorted(allUsers, key=lambda x: x[1])
	for userId, freq, percentDropped, percentConnected, phoneModel in sortedUsers:
		summaryWriter.writerow([userId, freq, percentDropped, percentConnected, phoneModel])
	summaryFile.close()








