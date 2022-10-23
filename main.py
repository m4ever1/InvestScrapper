import pandas as pd
import DataMining
import Utils
import backTester
# import weightCalculate

dateStart = {
        "year" : 2020,
        "month" : 9,
        "day"   : 1
    }
dateEnd = {
        "year" : 2021,
            "month" : 9,
            "day"   : 1
        }

dataSet = "nasdaq"

minSup = "8"
minDiv = "70"

listOfFiles = Utils.generateInputFile(dateStart, dateEnd, 'month', dataSet)

# listOfFiles = Utils.generateInputFileKmeans(dateStart, dateEnd, dataSet)


DataMining.mine(listOfFiles, minSup, minDiv)

print("Done with mining, back testing...")

# backTester.backTest(dateStart, dateEnd, dataSet)
backTester.backTestRollingWindow(dateStart, dateEnd, dataSet, 'month')


# dateStart = {
#         "year" : 2006,
#         "month" : 9,
#         "day"   : 1
#     }
# dateEnd = {
#         "year" : 2022,
#             "month" : 9,
#             "day"   : 1
#         }

# listOfFiles = Utils.generateInputFile(dateStart, dateEnd, 'month') 

# DataMining.mine(listOfFiles)
#store = pd.HDFStore('stocks.h5')



# dirname = os.path.dirname(__file__)


# weights = weightCalculate.getWeights(df)

    
