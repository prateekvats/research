#!/usr/bin/python
import csv
import glob
from collections import defaultdict

import datetime

import os
from matplotlib import pyplot as plt


#This function takes in the csvFile and the hive and gives you the plot for beeCount vs hour of the time for each hive everyday.
def plotHourlyGraphCSV(csvFile,hive):
    countDictionary = {}
    lineCount = 0
    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if lineCount == 0:
                lineCount += 1
                continue
            key = row[0]
            count = int(row[1])
            # temperature = float(row[2])
            hive = key.split('_')[0]
            date = key.split('_')[1]
            hour = key.split('_')[2]
            time = datetime.datetime.strptime(date + '_' + hour, "%m-%d-%y_%H")

            if time.date() not in countDictionary:
                countDictionary[time.date()] = {}

            countDictionary[time.date()][time.hour] = count

    dateKeys = countDictionary.keys()
    dateKeys.sort()
    selectedDays = defaultdict(list)
    plotPath = 'BeeCountsHourlyPlots/HourlyPlots_'+hive+'/'

    if not os.path.exists(plotPath):
        os.mkdir(plotPath)

    for date in dateKeys:
        hourKeys = countDictionary[date].keys()
        hourKeys.sort()
        plt.plot(hourKeys, [countDictionary[date][k] for k in hourKeys])
        plt.savefig(plotPath + date.strftime('%m-%d-%y') + '.png')
        plt.close()

#This function takes in the csvFile and the hive and gives you the plot for beeCount vs time but an averaged count of bees per day(But this plots all the csvFiles together)
def compareCSVsAndPlot(csvFiles):
    hiveCountsDaily = {}
    for csvFile in csvFiles:
        Hive = os.path.basename(csvFile).replace('.csv', '').split('_')[1]
        countDictionary = {}
        lineCount = 0
        with open(csvFile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if lineCount == 0:
                    lineCount += 1
                    continue
                key = row[0]
                count = int(row[1])
                temperature = float(row[2])
                hive = key.split('_')[0]
                date = key.split('_')[1]
                hour = key.split('_')[2]
                time = datetime.datetime.strptime(date + '_' + hour, "%m-%d-%y_%H")
                if time.date().month < 5:
                    continue
                if time.date() not in countDictionary:
                    countDictionary[time.date()] = {}

                countDictionary[time.date()][time.hour] = count

        dateKeys = countDictionary.keys()
        dateKeys.sort()
        for k in dateKeys:
            averageCount = 0
            numberOfCounts = len(countDictionary[k])
            for h in countDictionary[k]:
                averageCount += countDictionary[k][h]

            countDictionary[k] = int(averageCount / numberOfCounts)

        hiveCountsDaily[Hive] = countDictionary
    hivesKeys = hiveCountsDaily.keys()
    hivesKeys.sort()
    for k in hivesKeys:
        dateKeys = hiveCountsDaily[k].keys()
        dateKeys.sort()
        plt.plot(dateKeys, [hiveCountsDaily[k][d] for d in dateKeys])

    plt.legend(hivesKeys)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('Daily_Plot_HiveComparison.png',dpi=100)

#This function takes in the csvFile and the hive and gives you the plot for beeCount vs time but an averaged count of bees per day
def plotDailyAvgGraphCSV(csvFile,hive):
    countDictionary = {}
    lineCount = 0
    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if lineCount == 0:
                lineCount += 1
                continue
            key = row[0]
            count = int(row[1])
            # temperature = float(row[2])
            hive = key.split('_')[0]
            date = key.split('_')[1]
            hour = key.split('_')[2]
            time = datetime.datetime.strptime(date + '_' + hour, "%m-%d-%y_%H")
            if time.date() not in countDictionary:
                countDictionary[time.date()] = {}

            countDictionary[time.date()][time.hour] = count

    dateKeys = countDictionary.keys()
    # dateKeys.sort()
    for k in dateKeys:
        averageCount = 0
        numberOfCounts = len(countDictionary[k])
        for h in countDictionary[k]:
            averageCount += countDictionary[k][h]

        countDictionary[k] = int(averageCount / numberOfCounts)

    # plt.scatter(timeKeys,countList)
    # fig = plt.gcf()
    # fig.set_size_inches(18.5, 10.5)
    with open('/home/prateek/Desktop/My Thesis/Bee count data/BeeCount_'+hive+'.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["date"]+["count"])
        for k in countDictionary:
            wr.writerow([datetime.datetime.strftime(k,"%m-%d-%y"),countDictionary[k]])


    # plt.plot(dateKeys, [countDictionary[k] for k in dateKeys])
    # plt.show()
    # fig.savefig('Daily_Plot_AvgCounts_' + hive + '.png', dpi=100)
    #


#This function takes in the csvFile and the hive and gives you the plot for beeCount vs time but an averaged count of bees per day
def plotDailyGraphCSV(csvFile,hive):
    countDictionary = {}
    lineCount = 0
    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if lineCount == 0:
                lineCount += 1
                continue
            key = row[0]
            count = int(row[1])
            # temperature = float(row[2])
            hive = key.split('_')[0]
            date = key.split('_')[1]
            hour = key.split('_')[2]
            time = datetime.datetime.strptime(date + '_' + hour, "%m-%d-%y_%H")
            countDictionary[time] = count

    timeKeys = countDictionary.keys()
    timeKeys.sort()

    # plt.scatter(timeKeys,countList)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)


    plt.plot(timeKeys, [countDictionary[k] for k in timeKeys])
    # plt.show()
    fig.savefig('Daily_Plot_' + hive + '.png', dpi=100)
    #

def combineTemperatureData(temperatureDictionary,hiveVideosFolder):
    files = glob.glob(hiveVideosFolder+'*/temperatures.txt')
    for file in files:
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            lineCount = 0
            for row in csv_reader:
                #2018-05-26_14-46-59
                time = datetime.datetime.strptime(row[0], "%Y-%m-%d_%H-%M-%S")
                temperature = float(row[1])
                temperatureDictionary[time] = temperature


    return temperatureDictionary

def dictionaryToCSV(dictionaryObject,outputFile):
    keys = sorted(dictionaryObject.keys())
    with open(outputFile, 'w') as f:
        writer = csv.writer(f)
        for k in keys:
            writer.writerow([str(k),str(dictionaryObject[k])])



#This function takes in the csvFile and the hive and gives you the plot for beeCount vs temperature
def plotTemperature(csvFile,hive):
    temperatureDictionary = {}

    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lineCount = 0
        for row in csv_reader:
            if lineCount == 0:
                lineCount += 1
                continue
            count = int(row[1])
            temperature = float(row[2])
            temperatureDictionary[temperature] = count
    temperatures = temperatureDictionary.keys()
    temperatures.sort()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)


    plt.plot(temperatures, [temperatureDictionary[t] for t in temperatureDictionary])
    # plt.show()
    fig.savefig('TemperaturePlot_Onebox_' + hive + '.png', dpi=100)

def createCSVOfBeeCounts(dataDictionary, outputFile='beeCountData.csv'):
    fieldnames = ['Key', 'Bee_Detections', 'Temperature']
    tempDict = getAllTemperatures(args['temperature_directory'])
    with open(outputFile, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for k in dataDictionary:
            if k in tempDict:
                writer.writerow([k, dataDictionary[k], tempDict[k]])
            else:
                writer.writerow([k, dataDictionary[k], '-'])

def combineCSVs(csvFile1,csvFile2,hive):
    import pandas as pd
    output = hive+"_BeeCountData.csv"  # use your path
    allFiles = [csvFile1,csvFile2]

    list_ = []

    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0)
        list_.append(df)

    frame = pd.concat(list_, axis=0, ignore_index=True)

    frame.to_csv(output, sep=',')


temperatureDictionary = {}
combineTemperatureData(temperatureDictionary,"/home/prateek/Desktop/BeeCountCoRelationData/R_4_5/onebox/")
combineTemperatureData(temperatureDictionary,"/home/prateek/Desktop/BeeCountCoRelationData/R_4_5/twobox/")
dictionaryToCSV(temperatureDictionary,"/home/prateek/Desktop/My Thesis/temperatureData/temperature_4.5.csv")

temperatureDictionary = {}
combineTemperatureData(temperatureDictionary,"/home/prateek/Desktop/BeeCountCoRelationData/R_4_7/onebox/")
combineTemperatureData(temperatureDictionary,"/home/prateek/Desktop/BeeCountCoRelationData/R_4_7/twobox/")
dictionaryToCSV(temperatureDictionary,"/home/prateek/Desktop/My Thesis/temperatureData/temperature_4.7.csv")

temperatureDictionary = {}
combineTemperatureData(temperatureDictionary,"/home/prateek/Desktop/BeeCountCoRelationData/R_4_8/onebox/")
combineTemperatureData(temperatureDictionary,"/home/prateek/Desktop/BeeCountCoRelationData/R_4_8/twobox/")
dictionaryToCSV(temperatureDictionary,"/home/prateek/Desktop/My Thesis/temperatureData/temperature_4.8.csv")

temperatureDictionary = {}
combineTemperatureData(temperatureDictionary,"/home/prateek/Desktop/BeeCountCoRelationData/R_4_10/onebox/")
combineTemperatureData(temperatureDictionary,"/home/prateek/Desktop/BeeCountCoRelationData/R_4_10/twobox/")
dictionaryToCSV(temperatureDictionary,"/home/prateek/Desktop/My Thesis/temperatureData/temperature_4.10.csv")

# hive = '4.10'
# csvFile = hive+"_BeeCountData.csv"
# combineCSVs('HiveCountCollectedData/beeCountData_'+hive+'_onebox.csv','beeCountData_'+hive+'_twobox.csv',hive)
# plotDailyGraphCSV(csvFile,hive)
# plotDailyAvgGraphCSV(csvFile,hive)
# plotHourlyGraphCSV(csvFile,hive)
# compareCSVsAndPlot([
#                     'HiveCountCollectedData/beeCountData_4.5_onebox.csv',
#                     'HiveCountCollectedData/beeCountData_4.8_onebox.csv',
#                     'HiveCountCollectedData/beeCountData_4.10_onebox.csv',
#                     # 'HiveCountCollectedData/beeCountData_4.7_onebox.csv'
#                     ])
# plotTemperature(csvFile, hive)