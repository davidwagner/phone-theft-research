import UnlockTimeChecker
import ClassifierLog
import pickle

def main():
	unlockData = pickle.load(open("unlock_test_data.pkl", "rb" ))
	activatedIntervals = pickle.load(open("unlock_test_intervals.pkl", "rb"))
	print(unlockData[-10:])
	for interval in activatedIntervals:
		print(ClassifierLog.formatTimeValue(interval))
	
	numUnlocksSaved, numUnlocksTotal, unlockTimes = UnlockTimeChecker.computeUnlocks(unlockData, activatedIntervals)
	print("UNLOCK DATA:", str(numUnlocksSaved), str(numUnlocksTotal), str(unlockTimes))

if __name__ == '__main__':
	main()