from datetime import datetime


def computeUnlocks(keyguardData, activationIntervals):

	unlocksSaved = 0
	totalUnlocks = 0
	savedTimes = []

	intervalIndex = 0

	if len(keyguardData) > 0:
		keyGuardState = keyguardData[0][2]

	for event in keyguardData:
		tempState = event[2]
		# print("Keyguard:", keyGuardState, "Temp:", tempState)

		if keyGuardState == "true" and tempState == "false":
			unlockTime = event[0]
			print("Unlocked:", unlockTime)
			totalUnlocks += 1

			while intervalIndex < len(activationIntervals) and unlockTime > activationIntervals[intervalIndex][1]:
				intervalIndex += 1

				if intervalIndex >= len(activationIntervals):
					break
					# return (unlocksSaved, totalUnlocks, savedTimes)

			if intervalIndex < len(activationIntervals) and unlockTime >= activationIntervals[intervalIndex][0] and unlockTime <= activationIntervals[intervalIndex][1]:
				unlocksSaved += 1
				savedTimes.append(unlockTime)

		keyGuardState = tempState
		# print("Updating keyguard state, now:", keyGuardState)


	return (unlocksSaved, totalUnlocks, savedTimes)

def time(hours, minutes, seconds):
	return datetime(2017, 4, 11, hour=hours, minute=minutes, second=seconds)

if __name__ == '__main__':
	test_data = [[time(8, 0, 0), "true"], [time(8, 50, 10), "false"], [time(12, 0, 0), "true"], [time(12, 50, 10), "false"]]
	test_intervals = [(time(8, 40, 0), time(9, 10, 0)), (time(13, 40, 0), time(14, 10, 0))]
	print(computeUnlocks(test_data, test_intervals))
