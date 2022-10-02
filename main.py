import pandas as pd
import DataMining
import Utils
# import weightCalculate

dateStart = {
        "year" : 2007,
        "month" : 9,
        "day"   : 1
    }
dateEnd = {
            "year" : 2009,
            "month" : 9,
            "day"   : 1
        }

listOfFiles = Utils.generateInputFile(dateStart, dateEnd, 'year')

DataMining.mine(listOfFiles)
# store = pd.HDFStore('stocks.h5')



# dirname = os.path.dirname(__file__)


# weights = weightCalculate.getWeights(df)

    
