from distutils.command.config import config
from matplotlib import dates
import pandas as pd
import DataMining
import Utils
import backTester
import json

with open("config.json") as configFile:

    configData = json.load(configFile)
    dateStart = configData["dateStart"]
    dateEnd = configData["dateEnd"]
    dataSet = configData["dataSet"]
    minSup = configData["minSup"]
    minDiv = configData["minDiv"]
    granularity = configData["granularity"]

listOfFiles = Utils.generateInputFile(dateStart, dateEnd, granularity, dataSet)

DataMining.mine(listOfFiles, minSup, minDiv)

print("Done with mining, back testing...")

if granularity == "static":
    backTester.backTest(dateStart, dateEnd, dataSet)
else:
    backTester.backTestRollingWindow(dateStart, dateEnd, dataSet, granularity)

# listOfFiles = Utils.generateInputFileKmeans(dateStart, dateEnd, dataSet)
