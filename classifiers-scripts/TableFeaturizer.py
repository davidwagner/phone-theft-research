import csv
import math
import numpy



# numpy-1.11.1+mkl-cp27-cp27m-win_amd64
def getFile(dataFile):
    rows = []

    with open(dataFile) as f:
    # with open('./7_20_Phone_Table_Decrypted/test.csv') as f:    
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows

def getFeatures(rows):
    features = []

    for row in range(0, len(rows)-2):
        windowX = [float(rows[row][1]), float(rows[row+1][1]), float(rows[row+2][1])]
        windowY = [float(rows[row][2]), float(rows[row+1][2]), float(rows[row+2][2])]
        windowZ = [float(rows[row][3]), float(rows[row+1][3]), float(rows[row+2][3])]

        avgX = numpy.mean(windowX)
        avgY = numpy.mean(windowY)
        avgZ = numpy.mean(windowZ)

        t = int(rows[row][0])
        features.append((t,avgX, avgY, avgZ))

    return features
    
def checkFeatures(features):
    tableTimes = []
    temp = []
    tempNegative = []
    toggle = False

    """Edit as Necessary"""
    thresholdX = 0.8
    thresholdY = 0.2
    thresholdZ = 0.65
 
    for featureWindow in features:
        t = float(featureWindow[0])
        x = float(featureWindow[1])
        y = float(featureWindow[2])
        z = float(featureWindow[3])

        if (abs(x-0) < thresholdX and abs(y-0) < thresholdY and abs(z-9) < thresholdZ): #If it matches Threshold
            if len(tempNegative) > 0:
                if t - tempNegative[0] < 1000:  #If tempNegative has values and the difference between t and the first negative is less than 1000
                    for i in tempNegative:
                        tableTimes.append(i)
                tempNegative = []
            temp.append(t)
        else:      #If its above the threshold
            if len(temp) > 0:   #If temp has values
                diff = temp[len(temp) - 1] - temp[0]
                """Play with this number"""
                if diff > 500: #Check for ns!
                    print("yay")
                    for i in temp:
                        tableTimes.append(i)
                else:
                    for i in temp:
                        tempNegative.append(i)
                temp = []
            tempNegative.append(t)
            # print(t)
            # print(abs(x-0) < thresholdX)
            # print(abs(y-0) < thresholdY)
            # print(abs(z-9) < thresholdZ)
    # print(tableTimes)

    if len(temp) > 0:
        diff = temp[len(temp) - 1] - temp[0]
        """Play with this number"""
        if diff > 500: #Check for ns!
            print("yay")
            for i in temp:
                tableTimes.append(i)
    return tableTimes

def plotCheck(rows, tableTimes):

    import matplotlib.pyplot as pyplot

    markers_on = []

    tList = []
    yList = []
    xList = []
    zList = []
    

    toggle = False

    pyplot.figure()
    for a in rows:
        # print(a)
        tList.append(int(a[0]))
        xList.append(float(a[1]))
        yList.append(float(a[2]))
        zList.append(float(a[3]))

    l3 = [x for x in tList if x not in tableTimes]
    # print(l3)
    if len(l3) == 0:
        markers_on.append(tableTimes[0])
        markers_on.append(tableTimes[len(tableTimes)-1])

    else:

    # print(l3)
        for i in range(len(tableTimes)):
            if toggle:
                if len(l3) != 0 and tableTimes[i] > l3[0]:
                    markers_on.append(tableTimes[i-1])
                    toggle = False
                    while len(l3) != 0 and l3[0] < tableTimes[i]:
                        l3.remove(l3[0])
            else:
                markers_on.append(tableTimes[i])
                toggle = True
        if (len(tableTimes) != 0):
            print(tableTimes[len(tableTimes)-1])
            print(rows[len(rows)-1])
            markers_on.append(tableTimes[len(tableTimes)-1])

    # pyplot.plot(tList, xList,'-gD', markevery=markers_on)
    # pyplot.plot(tList, yList,'-gD', markevery=markers_on)
    # pyplot.plot(tList, zList,'-gD',  markevery=markers_on)

    pyplot.plot(tList, xList)
    pyplot.plot(tList, yList)
    pyplot.plot(tList, zList)
    
    for i in markers_on:
        pyplot.axvline(i)

    print(markers_on)
    pyplot.show()


def main():
    rows = getFile()
    features = getFeatures(rows)
    tableTimes = checkFeatures(rows)
    # print(tableTimes)
    plotCheck(rows, tableTimes)

if __name__ == "__main__":
    main()


# r = [math.sqrt(a*a + b*b + c*c) for (a, b, c) in zip(x, y, z)]