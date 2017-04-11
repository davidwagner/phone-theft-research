from datetime import datetime


def computeUnlocks(keyguardData, activationIntervals):

	unlocksSaved = 0
	totalUnlocks = 0
	savedTimes = []

	intervalIndex = 0

	keyGuardState = keyguardData[0][1]

	for event in keyguardData:
		tempState = event[1]

		if keyGuardState == "true" and tempState == "false":
			unlockTime = event[0]
			totalUnlocks += 1

			while unlockTime > activationIntervals[intervalIndex][1]:
				intervalIndex += 1

				if intervalIndex >= len(activationIntervals):
					return (unlocksSaved, totalUnlocks, savedTimes)

			if unlockTime >= activationIntervals[intervalIndex][0] and unlockTime <= activationIntervals[intervalIndex][1]
				unlocksSaved += 1
				savedTimes.append(unlockTime)

			keyguardState = tempState


	return (unlocksSaved, totalUnlocks, savedTimes)
