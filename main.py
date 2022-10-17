import pandas as pd
import DataMining
import Utils
# import weightCalculate

dateStart = {
        "year" : 2012,
        "month" : 8,
        "day"   : 1
    }
dateEnd = {
        "year" : 2013,
            "month" : 8,
            "day"   : 1
        }

listOfFiles = Utils.generateInputFile(dateStart, dateEnd, 'yearly', "nasdaq")


# DataMining.mine(listOfFiles)

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

    
