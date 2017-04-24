import datetime
# import Classifiers as classifiers
TABLE_CLASSIFIER = "Table Classifier"
POCKET_BAG_CLASSIFIER = "Pocket/Bag Classifier"
THEFT_CLASSIFIER = "Theft Classifier"
HAND_CLASSIFIER = "Hand Classifier"

unlockIndex = 3
PHONE_ACTIVATED = "activated"
PHONE_DEACTIVATED = "deactivated"
SAFE_PERIOD = datetime.timedelta(minutes=3)
BENIGN_CLASSIFIERS = set([POCKET_BAG_CLASSIFIER, HAND_CLASSIFIER])
START_OF_TIME = datetime.datetime.min
# unlock = 0, locked = 1
class PossessionState():
	def __init__(self, allData, sensorData, unlockData, smoothingNum):
		self.activeSensorData = sensorData
		self.unlockData = unlockData
		self.allData = allData

		self.dataIndex = smoothingNum // 2
		self.state = PHONE_DEACTIVATED
		self.lastUnlockedTime = START_OF_TIME
		self.lastBenignTime = START_OF_TIME
		self.lastClassificationTimes = {}
		self.intervals = []
		self.currentInterval = (self.getStateTime(), self.getStateTime())

	def isUnlocked(self):
		if self.dataIndex >= len(self.activeSensorData) - 1:
			return False

		lockValue = self.activeSensorData[self.dataIndex][unlockIndex]
		return True if lockValue == 0 else False

	def getStateTime(self):
		if self.dataIndex >= len(self.activeSensorData) - 1:
			return None

		time = self.activeSensorData[self.dataIndex][0]
		return time

	def updateState(self, time, classification):
		if self.dataIndex >= len(self.activeSensorData) - 1:
			self.state = PHONE_DEACTIVATED
			return

		self.lastClassificationTimes[classification] = time
		if classification in BENIGN_CLASSIFIERS:
			self.lastBenignTime = time

		currentTime = self.getStateTime()
		while currentTime != None and currentTime < time:
			self.dataIndex += 1
			currentTime = self.getStateTime()

		# print("POSESSION TIMES:", currentTime, time)
		if currentTime != time:
			print("DIFFERENT TIMES!", currentTime, time)
		timeSinceLastUnlocked = currentTime - self.lastUnlockedTime
		timeSinceLastBenign = currentTime - self.lastBenignTime

		prevState = self.state
		if self.isUnlocked():
			self.state = PHONE_ACTIVATED
			self.lastUnlockedTime = self.getStateTime()
		elif self.state == PHONE_ACTIVATED: 
			if timeSinceLastBenign <= SAFE_PERIOD:
				self.state = PHONE_ACTIVATED
			else:
				self.state = PHONE_DEACTIVATED
		else:
			self.state = PHONE_DEACTIVATED

		if self.state != prevState:
			interval = (self.currentInterval[0], currentTime)
			self.intervals.append((interval, prevState))
			self.currentInterval = (currentTime, currentTime)

	def getLastBenignTime(self):
		lastBenignTime = START_OF_TIME
		for c in BENIGN_CLASSIFIERS:
			if c in self.lastClassificationTimes:
				time = self.lastClassificationTimes[c]
				if time > lastBenignTime:
					lastBenignTime = time 
		return lastBenignTime

	def getIntervals(self):
		interval = (self.currentInterval[0], self.getStateTime())
		self.intervals.append((interval, self.state))

		return self.intervals

	def getIntervalsByState(self):
		intervalsByState = {PHONE_ACTIVATED : [], PHONE_DEACTIVATED : []}
		for interval in self.getIntervals():
			state = interval[1]
			intervalsByState[state].append(interval[0])

		return intervalsByState






