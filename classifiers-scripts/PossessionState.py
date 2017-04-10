import datetime
import Classifiers as c

unlockIndex = 3
PHONE_ACTIVATED = "activated"
PHONE_DEACTIVATED = "deactivated"
SAFE_PERIOD = datetime.timedelta(minutes=3)
BENIGN_CLASSIFIERS = set([c.POCKET_BAG_CLASSIFIER, c.HAND_CLASSIFIER])
START_OF_TIME = datetime.datetime.min
# unlock = 0, locked = 1
class PossessionState():
	def __init__(self, sensorData, smoothingNum):
		self.activeSensorData = sensorData

		self.dataIndex = smoothingNum // 2
		self.state = PHONE_DEACTIVATED
		self.lastUnlockedTime = START_OF_TIME
		self.lastBenignTime = START_OF_TIME
		self.lastClassificationTimes = {}

		if self.isUnlocked():
			self.state = PHONE_ACTIVATED
			self.lastUnlockedTime = self.getStateTime()

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
		assert(currentTime == time)
		timeSinceLastUnlocked = currentTime - self.lastUnlockedTime
		timeSinceLastBenign = currentTime - self.lastBenignTime

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

		self.dataIndex += 1

	def getLastBenignTime(self):
		lastBenignTime = START_OF_TIME
		for c in BENIGN_CLASSIFIERS:
			if c in self.lastClassificationTimes:
				time = self.lastClassificationTimes[c]:
				if time > lastBenignTime:
					lastBenignTime = time 
		return lastBenignTime





