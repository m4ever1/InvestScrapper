from cProfile import label
from yfinance import download
from Utils import downloadStockData
import inputReader
import matplotlib.pyplot as pl
import matplotlib.dates as mdates
import pandas as pd
from matplotlib import lines, markers
from cycler import cycler
import matplotlib.ticker as mtick
import json

filenamePathWin = "C:\\Users\\mglvi\\Documents\\InvestScrapper\\cpp\\bin\\outputs\\"
filenamePathLinux = "/mnt/c/Users/mglvi/Documents/InvestScrapper/cpp/bin/outputs/"
filenamePath = filenamePathLinux

outputFilePathWin = "C:\\Users\\mglvi\\Documents\\InvestScrapper\\"
outputFilePathLinux = "/mnt/c/Users/mglvi/Documents/InvestScrapper/"

markSpacing = 15


def getSharpeRatio(df):
    pd.set_option('use_inf_as_na', True)
    if isinstance(df, pd.DataFrame):
        dailyReturns = df.sum(axis=1).pct_change(1)
    else:
        dailyReturns = df.pct_change(1)
    dailySharp = dailyReturns.mean()/dailyReturns.std()
    yearlySharp = dailySharp * (252**0.5)
    return yearlySharp

def backTest(startDate: dict, endDate: dict, dataSet: str):
    tickerAndSharpe = {}
    testStartDate = {
            "year" : endDate["year"],
            "month" : endDate["month"],
            "day"   : endDate["day"]
        }
    testEndDate = {
            "year" : testStartDate["year"] + 1,
                "month" : testStartDate["month"],
                "day"   : testStartDate["day"]
            }

    tickerAndSharpe["metadata"] = { "Training start date" : startDate,
                                   "Training end date"    : endDate,
                                   "Testing start date"   : testStartDate,
                                   "Testing end date"     : testEndDate,
                                   "Data Set"             : dataSet
                                   }
    
    filename = filenamePath + \
                  f"output-{startDate['year']}-{startDate['month']}-{startDate['day']}-To-{endDate['year']}-{endDate['month']}-{endDate['day']}-{dataSet}.txt"
    filenameGics = filenamePath + \
                    f"output-{startDate['year']}-{startDate['month']}-{startDate['day']}-To-{endDate['year']}-{endDate['month']}-{endDate['day']}-{dataSet+'GICS'}.txt"

    filenameNoELim = filenamePath + \
                    f"output-{startDate['year']}-{startDate['month']}-{startDate['day']}-To-{endDate['year']}-{endDate['month']}-{endDate['day']}-{dataSet+'noelim'}.txt"

    filenameNoELimGics = filenamePath + \
                    f"output-{startDate['year']}-{startDate['month']}-{startDate['day']}-To-{endDate['year']}-{endDate['month']}-{endDate['day']}-{dataSet+'noelimGICS'}.txt"


    pl.figure(1, figsize=(12,7),dpi=250)
    pl.locator_params(axis="x", nbins=12)
    
    #Custom portfolio
    entries = inputReader.parseInput(filename)
    tickers = inputReader.buildPortfolio(entries)

    df = downloadStockData(testStartDate, testEndDate, "custom", tickers)

    print(f"Yearly sharpe ratio of custom = {getSharpeRatio(df)}")
    tickerAndSharpe["Portfolio"] = {"Tickers": tickers, 
                                  "Sharpe Ratio" : getSharpeRatio(df)}
    
    pl.plot((df.mean(axis=1)/df.iloc[0].mean() - 1) * 100, label="Portfolio", marker='o', markevery=markSpacing)

    #Portfolio no outrlier elim   
    # tickers = inputReader.buildPortfolio(inputReader.parseInput(filenameNoELim))

    # df = downloadStockData(testStartDate, testEndDate, "custom", tickers)

    # print(f"Yearly sharpe ratio of Portfolio without elim = {getSharpeRatio(df)}")
    
    # pl.plot((df.mean(axis=1)/df.iloc[0].mean() - 1) * 100, label="Portfolio no elim.", marker='.', markevery=markSpacing) 
    
    #ORIGINAL DISPLAN no outrlier elim   
    entries = inputReader.parseInput(filenameNoELimGics)
    tickers = inputReader.buildPortfolio(entries)

    df = downloadStockData(testStartDate, testEndDate, "custom", tickers)

    print(f"Yearly sharpe ratio of DISPLAN without elim = {getSharpeRatio(df)}")
    tickerAndSharpe["DISPLAN"] = {"Tickers": tickers, 
                                  "Sharpe Ratio" : getSharpeRatio(df)}
    
    pl.plot((df.mean(axis=1)/df.iloc[0].mean() - 1) * 100, label="DISPLAN", marker='d', markevery=markSpacing) 
    
    #Nasdaq
    df = downloadStockData(testStartDate, testEndDate, "custom", ["NDX"])

    print(f"Yearly sharpe ratio of nasdaq = {getSharpeRatio(df)}")
    tickerAndSharpe["NASDAQ-100"] = {"Sharpe Ratio" : getSharpeRatio(df)}
    # pl.plot((df.pct_change()*100).cumsum(), label="NASDAQ-100")
    pl.plot((df/df.iloc[0] - 1) * 100, label="NASDAQ-100", marker='s', markevery=markSpacing)

    # SPY

    df = downloadStockData(testStartDate, testEndDate, "custom", ["^GSPC"])

    print(f"Yearly sharpe ratio of spy = {getSharpeRatio(df)}")
    tickerAndSharpe["SPY"] = {"Sharpe Ratio" : getSharpeRatio(df)}

    pl.plot((df/df.iloc[0] - 1) * 100, label="SPY", marker='+', markevery=markSpacing)

    #Dow

    df = downloadStockData(testStartDate, testEndDate, "custom", ["^DJI"])

    print(f"Yearly sharpe ratio of dow = {getSharpeRatio(df)}")
    tickerAndSharpe["DOW"] = {"Sharpe Ratio" : getSharpeRatio(df)}
    pl.plot((df/df.iloc[0] - 1) * 100, label="DOW", marker="^", markevery=markSpacing)
    ax = pl.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    pl.gcf().autofmt_xdate() # Rotation
    ##############
    pl.legend()
    pl.grid(True)


    outputFilePath = outputFilePathLinux

    outputFile = f"{outputFilePath}/images/results-" \
            f"{testStartDate['year']}-{testStartDate['month']}-{testStartDate['day']}" \
            f"-to-" \
            f"{testEndDate['year']}-{testEndDate['month']}-{testEndDate['day']}" \
            f"-{dataSet}.png"

    print(f"Saving result image to {outputFile}")
    pl.tight_layout()
    ax = pl.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    pl.savefig(outputFile, dpi=250, pad_inches=0)
    pl.show()
    
    outputFile = f"{outputFilePath}/results/results-" \
        f"{testStartDate['year']}-{testStartDate['month']}-{testStartDate['day']}" \
        f"-to-" \
        f"{testEndDate['year']}-{testEndDate['month']}-{testEndDate['day']}" \
        f"-{dataSet}.json"
        
    json_object = json.dumps(tickerAndSharpe, indent=4)
    with open(outputFile, "w") as jsonf:
        jsonf.write(json_object)

    outputFile = f"{outputFilePath}/images/results-" \
        f"{testStartDate['year']}-{testStartDate['month']}-{testStartDate['day']}" \
        f"-to-" \
        f"{testEndDate['year']}-{testEndDate['month']}-{testEndDate['day']}" \
        f"-{dataSet+'SharpeRatioValues'}.png"

    pl.figure(2)
    pl.xlabel("Assets")
    pl.ylabel("Sharpe ratio")
    sharpes = {key : value["Sharpe Ratio"] for key, value in tickerAndSharpe.items() if key != "metadata" }
    pl.bar(list(sharpes.keys()), list(sharpes.values()))
    pl.legend()
    pl.grid(True)
    pl.savefig(outputFile, dpi=200)
    pl.show()
    
def backTestRollingWindow(dateStart: dict, dateEnd: dict, dataSet: str, granularity: str):
    
    auxDateStart = dateStart.copy()
    pl.figure(1, figsize=(12,7),dpi=250)
    pl.locator_params(axis="x", nbins=12)
    
    portfolioDf = pd.DataFrame()
    displanDf = pd.DataFrame()
    nasdaqDf = pd.DataFrame()
    spyDf = pd.DataFrame()
    dowDf = pd.DataFrame() 
    
    dfDict = {
        "Portfolio" : portfolioDf,
        "DISPLAN" : displanDf,
        "NASDAQ-100"    : nasdaqDf,
        "SPY"       : spyDf,
        "DOW"       : dowDf
    }

    
    while(auxDateStart != dateEnd):
        if auxDateStart["month"] == 13:
            auxDateStart["year"] += 1
            auxDateStart["month"] = 1
            
        auxDateEnd = auxDateStart.copy()

        if granularity == "trimester" :
            if auxDateEnd["month"] + 3 > 12:
                auxDateEnd["month"] = (auxDateEnd["month"]+3)%12 if (auxDateEnd["month"]+3)%12 != 0 else 1
                auxDateEnd["year"] += 1
            else:
                auxDateEnd["month"] += 3
        else:
            if auxDateEnd["month"] == 12:
                auxDateEnd["month"] = 1
                auxDateEnd["year"] += 1
            else:
                auxDateEnd[granularity] += 1
            
        filenameWin = f"C:\\Users\\mglvi\\Documents\\InvestScrapper\\cpp\\bin\\outputs\\output-" \
                    f"{auxDateStart['year']}-{auxDateStart['month']}-{auxDateStart['day']}-To-{auxDateEnd['year']}-{auxDateEnd['month']}-{auxDateEnd['day']}-{dataSet}.txt"
        filenamePortfolio = filenamePath + \
                        f"/output-{auxDateStart['year']}-{auxDateStart['month']}-{auxDateStart['day']}-To-{auxDateEnd['year']}-{auxDateEnd['month']}-{auxDateEnd['day']}-{dataSet}.txt"
        filenameNoELimGics = filenamePath + \
                    f"/output-{auxDateStart['year']}-{auxDateStart['month']}-{auxDateStart['day']}-To-{auxDateEnd['year']}-{auxDateEnd['month']}-{auxDateEnd['day']}-{dataSet+'noelimGICS'}.txt"
        
        filenamesDict = {
            "Portfolio" : filenamePortfolio,
            "DISPLAN"   : filenameNoELimGics
        }
        
        testDateStart = {
                "year" : auxDateStart["year"] + 1,
                "month" : auxDateStart["month"],
                "day"   : auxDateStart["day"]
                }       
        testDateEnd = {
                "year" : auxDateEnd["year"] + 1,
                    "month" : auxDateEnd["month"],
                    "day"   : auxDateEnd["day"]
                }
        
        addToDfs(filenamesDict, testDateStart, testDateEnd, dfDict)
        
        if granularity == "trimester":
            if auxDateStart["month"] + 3 > 12:
                auxDateStart["month"] = (auxDateStart["month"]+3)%12 if (auxDateStart["month"]+3)%12 != 0 else 1
                auxDateStart["year"] += 1
            else:
                auxDateStart["month"] += 3
        else:
            auxDateStart[granularity] += 1
            
    ax = pl.gca()
    cm = pl.get_cmap('tab10')
    styles =  cycler('linestyle', ['-', '-', '-', '-','--', '--', '--', '--']) + (cycler('marker',['s','.', 'v', '<','s','.', 'v', '<']) ) + cycler('color', [cm(1.*i/8) for i in range(8)])
    ax.set_prop_cycle(styles)
    pl.figure(1)
    
    buildFigure(dfDict)
    outputFilePath = outputFilePathLinux

    outputFile = f"{outputFilePath}/images/results-" \
            f"{dateStart['year']+1}-{dateStart['month']}-{dateStart['day']}" \
            f"-to-" \
            f"{dateEnd['year']+1}-{dateEnd['month']}-{dateEnd['day']}" \
            f"-{dataSet}-rollingWindow.png"
            
    print(f"Saving result image to {outputFile}")
    pl.tight_layout()
    pl.savefig(outputFile, dpi=250)
    pl.show()
    pl.figure(2)
    outputFile = f"{outputFilePath}/images/results-" \
        f"{dateStart['year']+1}-{dateStart['month']}-{dateStart['day']}" \
        f"-to-" \
        f"{dateEnd['year']+1}-{dateEnd['month']}-{dateEnd['day']}" \
        f"-{dataSet+'-rollingWindow-SharpeRatioValues'}.png"

    pl.xlabel("Assets")
    pl.ylabel("Sharpe ratio")

    pl.legend()
    pl.grid(True)
    pl.savefig(outputFile, dpi=200)
    pl.show()

def addToDfs(filenamesDict: str, testStartDate: dict, testEndDate: dict, dfDict: dict):

    for name, filename in filenamesDict.items():
        #Custom portfolio
        tickers = inputReader.buildPortfolio(inputReader.parseInput(filename))
        df = downloadStockData(testStartDate, testEndDate, "custom", tickers)
        if not dfDict[name].empty:
            df = (df.pct_change().fillna(0).cumsum().mean(axis=1))* 100 + dfDict[name].iloc[-1].values[0]
        else:
            df = (df.pct_change().fillna(0).cumsum().mean(axis=1))* 100
        dfDict[name] = pd.concat([dfDict[name], df])
    

    #Nasdaq
    df = downloadStockData(testStartDate, testEndDate, "custom", ["NDX"])
    dfDict["NASDAQ-100"] = pd.concat([dfDict["NASDAQ-100"], df])

    #SPY
    df = downloadStockData(testStartDate, testEndDate, "custom", ["^GSPC"])
    dfDict["SPY"] = pd.concat([dfDict["SPY"], df])
    
    #Dow
    df = downloadStockData(testStartDate, testEndDate, "custom", ["^DJI"])
    dfDict["DOW"] = pd.concat([dfDict["DOW"], df])

    # return dfDict


def buildFigure(dfDict: dict):

    sharpeRatios = {}
    pl.figure(1)
    for name, df in dfDict.items():
        if name in ['DOW','SPY','NASDAQ-100']:
            pl.plot((df/df.iloc[0] - 1) * 100, label=name, markersize=10, markevery=markSpacing)
        else:
            print(f"Yearly sharpe ratio of {name} = {getSharpeRatio(df)}")
            pl.plot(df, label=name, markersize=10, markevery=markSpacing)
            sharpeRatios[name] = getSharpeRatio(df)
        pl.figure(2)
        pl.bar(name, getSharpeRatio(df), color='b')
        pl.figure(1)
        # pl.plot((df.mean(axis=1)/df.iloc[0].mean() - 1) * 100, label="Portfolio")

    # df = dfDict["DISPLAN"]
    # print(f"Yearly sharpe ratio of custom = {getSharpeRatio(df)}")
    # pl.plot(df, label="DISPLAN")

    # df = dfDict["nasdaq"]
    # print(f"Yearly sharpe ratio of nasdaq = {getSharpeRatio(df)}")
    # pl.plot((df/df.iloc[0] - 1) * 100, label="NASDAQ-100")

    # df = dfDict["spy"]
    # print(f"Yearly sharpe ratio of spy = {getSharpeRatio(df)}")
    # pl.plot((df/df.iloc[0] - 1) * 100, label="SPY")
    
    # df = dfDict["dow"]
    # print(f"Yearly sharpe ratio of dow = {getSharpeRatio(df)}")
    # pl.plot((df/df.iloc[0] - 1) * 100, label="DOW")

    pl.gcf().autofmt_xdate()
    
    ax = pl.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    pl.gcf().autofmt_xdate() # Rotation
    ##############
    pl.legend()
    pl.grid(True)
    
def backTestKmeans(startDate: dict, endDate: dict, dataSet: str):
    markSpacing = 15
    testStartDate = {
            "year" : endDate["year"],
            "month" : endDate["month"],
            "day"   : endDate["day"]
        }
    testEndDate = {
            "year" : testStartDate["year"] + 1,
                "month" : testStartDate["month"],
                "day"   : testStartDate["day"]
            }
    pl.figure(1,figsize=(12,8),dpi=300)
    pl.locator_params(axis="x", nbins=12)
    ax = pl.gca()
    cm = pl.get_cmap('tab10')
    styles =  cycler('linestyle', ['-', '-', '-', '-','--', '--', '--', '--']) + (cycler('marker',['s','.', 'v', '<','s','.', 'v', '<']) ) + cycler('color', [cm(1.*i/8) for i in range(8)])
    ax.set_prop_cycle(styles)
    sharpeRatios = {}
    numStocks = {}
    for k in range(2,20):
        filename = filenamePath + \
                    f"output-{startDate['year']}-{startDate['month']}-{startDate['day']}-To-{endDate['year']}-{endDate['month']}-{endDate['day']}-{dataSet+'kmeans'+str(k)}.txt"
        #Custom portfolio
        tickers = inputReader.buildPortfolio(inputReader.parseInput(filename))

        df = downloadStockData(testStartDate, testEndDate, "custom", tickers)

        print(f"Yearly sharpe ratio of custom = {getSharpeRatio(df)}")
        sharpeRatios[k] = getSharpeRatio(df)
        numStocks[k] = len(tickers)
        pl.figure(1)
        if isinstance(df, pd.DataFrame):
            if((df.mean(axis=1)/df.iloc[0].mean() - 1) * 100)[-1] < -5:
                continue
            pl.plot((df.mean(axis=1)/df.iloc[0].mean() - 1) * 100, label=f"k={k}", markersize=10, markevery=markSpacing)
        else:
            if((df/df.iloc[0] - 1) * 100)[-1] < -5:
                continue
            pl.plot((df/df.iloc[0] - 1) * 100, label=f"k={k}", markersize=10, markevery=markSpacing)
            
            

    pl.figure(1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    pl.gcf().autofmt_xdate() # Rotation
    ##############
        
    pl.legend()
    pl.grid(True)


    outputFilePath = outputFilePathLinux

    outputFile = f"{outputFilePath}/images/results-" \
            f"{testStartDate['year']}-{testStartDate['month']}-{testStartDate['day']}" \
            f"-to-" \
            f"{testEndDate['year']}-{testEndDate['month']}-{testEndDate['day']}" \
            f"-{dataSet+'-kmeans'}.png"

    print(f"Saving result image to {outputFile}")
    pl.savefig(outputFile, dpi=300)
    
    
    pl.figure(2)
    pl.xlabel('Number of clusters')
    pl.ylabel('Sharpe Ratio')
    ax = pl.gca()
    ax.set_xticks(list(sharpeRatios.keys()))
    pl.plot(sharpeRatios.keys(), sharpeRatios.values(),marker='o')
    
    outputFile = f"{outputFilePath}/results-" \
            f"{testStartDate['year']}-{testStartDate['month']}-{testStartDate['day']}" \
            f"-to-" \
            f"{testEndDate['year']}-{testEndDate['month']}-{testEndDate['day']}" \
            f"-{dataSet+'-SharpeRatio'}.png"
    
    pl.savefig(outputFile, dpi=300)
    
    pl.figure(3)
    pl.xlabel('Number of clusters')
    pl.ylabel('Number of stocks in portfolio')
    ax = pl.gca()
    ax.set_xticks(list(numStocks.keys()))
    pl.plot(numStocks.keys(),numStocks.values(),marker='o')
    outputFile = f"{outputFilePath}/results-" \
            f"{testStartDate['year']}-{testStartDate['month']}-{testStartDate['day']}" \
            f"-to-" \
            f"{testEndDate['year']}-{testEndDate['month']}-{testEndDate['day']}" \
            f"-{dataSet+'-NumStocks'}.png"
    
    pl.savefig(outputFile, dpi=300)
    pl.show()    