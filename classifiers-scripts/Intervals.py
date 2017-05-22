import datetime
import Sensors as sensors
import Classifiers as classifiers 

from configsettings import *
from collections import deque, Counter

def compareIntervals(intervals1, intervals2):
    i1 = 0
    i2 = 0
    interval1 = intervals1[i1][0]
    interval2 = intervals2[i2][0]
    class1 = intervals1[i1][1]
    class2 = intervals1[i2][1]

    startTime = interval1[0] if interval1[0] > interval2[0] else interval2[0]
    endTime = None

    comparedIntervals = []
    matchingIntervals = []
    conflictingIntervals = []
    while i1 < len(intervals1) and i2 < len(intervals2):
        interval1 = intervals1[i1][0]
        interval2 = intervals2[i2][0]
        class1 = intervals1[i1][1]
        class2 = intervals2[i2][1]

        if interval1[1] == interval2[1]:
            endTime = interval1[1]
            i1 += 1
            i2 += 1
        elif interval1[1] < interval2[1]:
            endTime = interval1[1]
            i1 += 1
        else:
            endTime = interval2[1]
            i2 += 1

        comparedClass = None
        matchingClasses = False
        if class1 == class2:
            comparedClass = class1
            matchingClasses = True
        else:
            comparedClass = str(class1) + " | " + str(class2)

        comparedInterval = ((startTime, endTime), comparedClass, matchingClasses)
        comparedIntervals.append(comparedInterval)

        if matchingClasses:
            matchingIntervals.append(comparedInterval)
        else:
            conflictingIntervals.append(comparedInterval)

        startTime = endTime

    return comparedIntervals, matchingIntervals, conflictingIntervals

def mergeAdjacentIntervalsByValue(intervals):
    i = 0
    while i + 1 < len(intervals):
        curr = intervals[i]
        next = intervals[i + 1]
        if curr[1] == next[1]:
            intervals[i] = ((curr[0][0], next[0][1]), curr[1])
            del intervals[i + 1]
        else:
            i += 1 

def mergeAdjacentIntervals(intervals):
    i = 0
    while i + 1 < len(intervals):
        curr = intervals[i]
        next = intervals[i + 1]
        if curr[1] == next[0]:
            intervals[i] = (curr[0], next[1])
            del intervals[i + 1]
        else:
            i += 1

def filterSpikesFromIntervals(intervals, intervalsByValue):
    spikeLength = datetime.timedelta(seconds=1)
    i = 1
    indexAddedToIntervalsByValue = -1
    while i < len(intervals) - 1:
        interval, intervalBefore, intervalAfter = intervals[i], intervals[i - 1], intervals[i + 1]

        timeInterval = interval[0]

        if timeInterval[1] - timeInterval[0] <= spikeLength:
            newTimeInterval = (intervalBefore[0][0], intervalAfter[0][1])
            intervals[i - 1] = (newTimeInterval, intervalBefore[1])
            del intervals[i:i+2]
        else:
            timeIntervalBefore = intervalBefore[0]
            classification = intervalBefore[1]
            intervalsByValue[classification].append(timeIntervalBefore)
            indexAddedToIntervalsByValue = i - 1
            i += 1

    for j in range(indexAddedToIntervalsByValue + 1, len(intervals)):
        interval = intervals[j]
        timeInterval = interval[0]
        classification = interval[1]
        intervalsByValue[classification].append(timeInterval)

def findCommonIntervalsByValue(intervals1, intervals2, value):
    # print("Finding common intervals!")
    # print intervals1
    # print intervals2

    if len(intervals1) == 0 and len(intervals2) == 0:
        return []
    if len(intervals1) == 0:
        return intervals2
    if len(intervals2) == 0:
        return intervals1 

    def advance(intervals, i, value):
        while i < len(intervals) and intervals[i][1] != value:
            # print(i)
            i += 1
        return i 

    i1 = advance(intervals1, 0, value) 
    i2 = advance(intervals2, 0, value)
    # print i1, i2 
    
    commonIntervals = []
    while i1 < len(intervals1) and i2 < len(intervals2):
        interval1 = intervals1[i1][0]
        interval2 = intervals2[i2][0]
        # # print(i1, i2)
        laterStartingInterval, earlierStartingInterval = None, None
        later_i, earlier_i = None, None

        if interval1[0] >= interval2[0]:
            laterStartingInterval, earlierStartingInterval = interval1, interval2
            later_i, earlier_i = i1, i2
        else:
            laterStartingInterval, earlierStartingInterval = interval2, interval1
            later_i, earlier_i = i2, i1

        if laterStartingInterval[0] >= earlierStartingInterval[1]:
            if earlier_i == i1:
                i1 = advance(intervals1, i1, value)
            else:
                i2 = advance(intervals2, i2, value)
        
        else:
            earlierEndingInterval = earlierStartingInterval if earlierStartingInterval[1] <= laterStartingInterval[1] else laterStartingInterval

            commonIntervals.append((laterStartingInterval[0], earlierEndingInterval[1]))
            # print commonIntervals

            if earlierStartingInterval[1] == laterStartingInterval[1]:
                # print "End times are equal"
                i1 = advance(intervals1, i1, value)
                i2 = advance(intervals2, i2, value)

            elif earlierStartingInterval[1] < laterStartingInterval[1]:
                # print "Early start ends earlier, advance early"
                if earlier_i == i1:
                    i1 = advance(intervals1, i1, value)
                else:
                    i2 = advance(intervals2, i2, value)
                # print i1, i2
            else:
                # print "Early start ends later, advance later"
                if later_i == i1:
                    i1 = advance(intervals1, i1, value)
                else:
                    i2 = advance(intervals2, i2, value)
                # print i1, i2

    return commonIntervals

def findCommonIntervals(intervals1, intervals2):
    # print("Finding common intervals!")
    # print intervals1
    # print intervals2

    if len(intervals1) == 0 and len(intervals2) == 0:
        return []
    if len(intervals1) == 0:
        return []
    if len(intervals2) == 0:
        return []

    i1 = 0
    i2 = 0
    # print "Starting"
    # print i1, i2 
    
    commonIntervals = []
    while i1 < len(intervals1) and i2 < len(intervals2):
        interval1 = intervals1[i1]
        interval2 = intervals2[i2]

        laterStartingInterval, earlierStartingInterval = None, None
        later_i, earlier_i = None, None

        if interval1[0] >= interval2[0]:
            # print("Interval1 starts after Interval2")
            laterStartingInterval, earlierStartingInterval = interval1, interval2
            later_i, earlier_i = "i1", "i2"
        else:
            # print("Interval2 starts after Interval1")
            laterStartingInterval, earlierStartingInterval = interval2, interval1
            later_i, earlier_i = "i2", "i1"

        if laterStartingInterval[0] >= earlierStartingInterval[1]:
            # print("GOODBYE")
            # print("Later starting interval starts completely after early interval")
            if earlier_i == "i1":
                i1 += 1
            else:
                i2 += 1
        
        else:
            # print("HELLO")
            earlierEndingInterval = earlierStartingInterval if earlierStartingInterval[1] <= laterStartingInterval[1] else laterStartingInterval
            # print("Earlier ending interval:", TimeFileUtils.formatTimeInterval(earlierEndingInterval))
            
            commonIntervals.append((laterStartingInterval[0], earlierEndingInterval[1]))
            # print("Common Intervals:")
            # for interval in commonIntervals:
            #     print(TimeFileUtils.formatTimeInterval(interval))


            if earlierStartingInterval[1] == laterStartingInterval[1]:
                # print("End times are equal")
                i1 += 1
                i2 += 1

            elif earlierStartingInterval[1] < laterStartingInterval[1]:
                # print("Early start ends earlier, advance early")
                if earlier_i == "i1":
                    i1 += 1
                else:
                    i2 += 1
                # print i1, i2
            else:
                # print("Early start ends later, advance later")
                if later_i == "i1":
                    i1 += 1
                else:
                    i2 += 1
                # print i1, i2

    return commonIntervals

def plotIntervals(intervals):
    times = []
    values = []

    for interval in intervals:
        time = interval[0]
        times.append(time[0])
        times.append(time[1])
        values.append(interval[1])
        values.append(interval[1])

    times = date2num(times)

    seconds = SecondLocator()   # every year
    minutes = MinuteLocator()  # every month
    hours = HourLocator()
    hoursFmt = DateFormatter('%H:%M')
    minutesFmt = DateFormatter('%H:%M:%S')

    fig, ax = plt.subplots()
    ax.plot_date(times, values, '-')

    # format the ticks
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(hoursFmt)
    ax.xaxis.set_minor_locator(minutes)
    ax.autoscale_view()


    # format the coords message box
    ax.fmt_xdata = DateFormatter('%H:%M')
    ax.grid(True)

    axes = plt.gca()
    axes.set_ylim([-0.25, 1.25])

    fig.autofmt_xdate()
    plt.show()

def getIntervalStats(intervals):
    stats = {}
    intervalLengths = [intervalLength(interval) for interval in intervals]
    # print(intervalLengths)
    totalTimeSpent = datetime.timedelta(seconds=0)
    for interval in intervalLengths:
        if type(interval) is int:
            continue
        totalTimeSpent += interval

    medianLength = "N/A"
    avgLength = "N/A"
    longestInterval = "N/A"
    shortestInterval = "N/A"

    if totalTimeSpent.total_seconds() < 0:
        for interval in intervals:
            print(TimeFileUtils.formatTimeInterval(interval))

    if len(intervals) > 0:
        medianLength = intervalLength(intervals[len(intervals) // 2])
        avgLength = totalTimeSpent / len(intervalLengths)
        longestInterval = intervals[-1]
        shortestInterval = intervals[0]


    stats["totalTimeSpent"] = totalTimeSpent
    stats["medianLength"] = medianLength
    stats["avgLength"] = avgLength
    stats["longestInterval"] = longestInterval
    stats["shortestInterval"] = shortestInterval

    return stats

def getIntervalStatHeaders(classifier_name):
    headers = ["% Time Positive", "Total Time Positive", "Median Period Length", "Average Period Length",
               "Longest Positive Period", "Shortest Positive Period", "% Time Negatve", "Total Time Negative", "Median Period Length", "Average Period Length",
               "Longest Negative Period", "Shortest Negative Period",]

    classifier = " (" + classifier_name  + ")"
    return [header + classifier for header in headers]