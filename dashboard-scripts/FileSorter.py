import os
import shutil
import sys



"""The path to the directory with all the files: """
sourceDirectory = "./../../../Downloads/0912_pull" #change as needed

"""The path to the directory where you want the files to be moved. 
The subdirectory will automatically create a subdirectory inside with the date


ex: If your directory is "TestSubjects" and you input "2016_01_01", the files will be located in "TestSubjects/2016_01_01"
"""
targetDirectory = "./Dashboard_Results" #change as needed

"""
The path to the decryptor Python script
e.g. "bintocsv.py"
"""
decryptor  = "bintocsv.py"

"""The path to the location of the privatekey"""
privateKeyPath = "./ucb_keypair/ucb.privatekey"


""" If you have pycrypto installed (pip install pycrypto) then set to True """
HAS_PYCRYPTO = False

""" If Windows uncomment and comment Mac line"""
# PYTHON = "python"

""" If Mac """
PYTHON = "python2.7"






"""Invariant: Date MUST BE IN FORM year_month_day"""
def moveFiles(date):
	checkDate = date.split("_")
	if len(checkDate) != 3:
		print("Double check your date")
		return

	newTargetDir = targetDirectory + "/" + date
	encryptedDir = newTargetDir+ "/encrypted"
	if os.path.exists(encryptedDir):
		shutil.rmtree(encryptedDir)
	os.makedirs(encryptedDir)

	for filename in os.listdir(sourceDirectory):
		temp = filename.split("_")
		if checkDate[0] in temp:
			
			yearPos = temp.index(checkDate[0])
			if len(temp) > yearPos + 2:
				if temp[yearPos + 1] == checkDate[1] and temp[yearPos + 2] == checkDate[2]:
					
					shutil.copy2(sourceDirectory + "/" + filename, encryptedDir)

	return encryptedDir, newTargetDir

def decrypt(encryptedPath, destPath):



	command = PYTHON + " " + decryptor + " " + privateKeyPath + " -s " + \
	encryptedPath  + " -d " + destPath + " -l verb"

	print("Running command: " + command)
	os.system(command)

	print("Decryption succesful")



def main():
	if len(sys.argv) < 2:
		print("Enter date in format year_month_day as an argument")
		sys.exit()

	date  = sys.argv[1]


	print("The date you have entered is " + date)
	print("If the path already exists, you will overwrite your past directory")
	print("Enter Y if you wish to continue. Enter N for quit.")
	while True:
		inputString = raw_input()
		if inputString == "Y":
			break
		elif inputString == "N":
			sys.exit()

	print("Please take this moment to make sure your paths are inputted correctly")
	print("Enter Y if you wish to continue. Enter N for quit.")
	while True:
		inputString = raw_input()
		if inputString == "Y":
			break
		elif inputString == "N":
			sys.exit()

	print("Attempting to move files")
	encryptedDir, targetDir = moveFiles(date)
	print("Files moved succesfully")
	if HAS_PYCRYPTO:
		decrypt(encryptedDir, targetDirectory + "/" + date)

if __name__ == '__main__':
	main()
