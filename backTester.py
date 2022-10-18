from cProfile import label
from yfinance import download
from Utils import downloadStockData
import inputReader
import matplotlib.pyplot as pl
import matplotlib.dates as mdates

trainingStartDate = {
        "year" : 2013,
        "month" : 8,
        "day"   : 1
    }
trainingEndDate = {
        "year" : 2014,
            "month" : 8,
            "day"   : 1
        }

dataSet = "custom"


filename = '/home/miguel/InvestScrapper/cpp/bin/outputs/output-{2012}-{8}-{1}-To-{2013}-{8}-{1}-{nasdaq}.txt'

pl.figure(figsize=(10,5),dpi=300)
pl.locator_params(axis="x", nbins=12)

#Custom portfolio
tickers = inputReader.buildPortfolio(inputReader.parseInput(filename))

df = downloadStockData(trainingStartDate, trainingEndDate, dataSet, tickers)

((df.iloc[-1] - df)/df.iloc[0])*100

# pl.plot((df.pct_change()*100).cumsum().mean(axis=1), label="portfolio")
pl.plot((df.mean(axis=1)/df.iloc[0].mean() - 1) * 100, label="portfolio")

# set font and rotation for date tick labels
pl.gcf().autofmt_xdate()
#Nasdaq

df = downloadStockData(trainingStartDate, trainingEndDate, "custon", ["NDX"])

((df.iloc[-1] - df)/df.iloc[0])*100

# pl.plot((df.pct_change()*100).cumsum(), label="NASDAQ-100")
pl.plot((df/df.iloc[0] - 1) * 100, label="NASDAQ-100")

#SPY

df = downloadStockData(trainingStartDate, trainingEndDate, "custom", ["^GSPC"])

((df.iloc[-1] - df)/df.iloc[0])*100

pl.plot((df/df.iloc[0] - 1) * 100, label="SPY")

#Dow

df = downloadStockData(trainingStartDate, trainingEndDate, "custom", ["DJI"])

((df.iloc[-1] - df)/df.iloc[0])*100

pl.plot((df/df.iloc[0] - 1) * 100, label="DOW")
ax = pl.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
pl.gcf().autofmt_xdate() # Rotation
##############
pl.legend()
pl.grid(True)
pl.show()