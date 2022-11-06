import datetime, requests, re, sys, os
import bs4 as bs
import yfinance as yf
from collections import defaultdict
import pandas as pd
import numpy as np
from kmeans import getSectorsDict
from math import sqrt
import  pylab as pl
from collections import OrderedDict
from scipy import stats
import json

def getFileDir():
    if getattr(sys, 'frozen', False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)

    return datadir

def getTable():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    return table

def compareSectors(df: pd.DataFrame):
    gicsSects = getGICSSectors(df)
    kmeansSects = getSectorsDict(df)[0]
    countDict = {}

    print("GICS SECTORS")

# reads tickers from a text file in a comma separated list e.g. (A,AAP,APPL,AAL) 
# DOES NOT SANITIZE: SPACES IN BETWEEN, BRACKETS OR COMMAS, it only eliminates duplicates and sorts the tickers lexicographycally
def readTickersFromFile(fileName: str = None) -> list:
    if fileName is None:
        fileName = "nasdaqTickers2013-2014.txt" 
    with open(fileName, "r") as file:
        text = file.read()
        tickers = text.split(',')
        tickers = sorted(set(tickers)) # set() filters non-uniques, and sorted... sorts.

    return tickers

def scrapeTickerList(index: str = "spy"):
    url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tickerColumn = 0
    if index != "spy":
        if index == "nasdaq":
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tickerColumn = 1
        elif index == "dow":
            url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            tickerColumn = 1
            
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable', 'id' : 'constituents'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[tickerColumn].text
        ticker = ticker.strip()
        tickers.append(ticker)

    tickers = [s.replace('\n', '') for s in tickers]

    return tickers

def downloadStockData(dateStart: dict, dateEnd: dict, dataSet: str = "spy", tickerList_in:list = None) -> pd.DataFrame:
    if dataSet == "spy":
        tickerList = scrapeTickerList("spy")
    elif dataSet == "nasdaq":
        if dateEnd["year"] < 2000:
            tickerList = readTickersFromFile()
        else:
            tickerList = scrapeTickerList("nasdaq")
    elif dataSet == "dow":
        tickerList = scrapeTickerList("dow")
    else: 
        if tickerList_in is None:
            raise Exception("Invalid Ticker List")
        else:
            tickerList = tickerList_in

    start = datetime.datetime(dateStart["year"], dateStart["month"], dateStart["day"])
    end = datetime.datetime(dateEnd["year"], dateEnd["month"], dateEnd["day"])
    
    data = yf.download(tickerList, start=start, end=end)
    data["Adj Close"].to_csv("stocks.csv")
    data = data["Adj Close"]
    data = data.dropna(how="all")
    if not isinstance(data,pd.Series):
        data = data.dropna(axis=1)
    data = data.interpolate()
    
    return data

def readDataFromCsv() -> pd.DataFrame:
    df = pd.read_csv("stocks.csv")
    df = df.set_index("Date")
    df = df.sort_index()
    
    return df


def filterStocksByDate(df: pd.DataFrame, dateStart: dict, dateEnd: dict) -> pd.DataFrame:
    start = datetime.datetime(dateStart["year"], dateStart["month"], dateStart["day"])
    end = datetime.datetime(dateEnd["year"], dateEnd["month"], dateEnd["day"])
    
    dfOut = df.loc[str(start):str(end)]
    dfOut = dfOut.dropna(how="all")
    dfOut = dfOut.dropna(axis=1)
    dfOut = dfOut.interpolate()
    return dfOut

def convertToInputFile(df: pd.DataFrame, dateStart: dict, dateEnd: dict, dataSet: str, gics: bool = False, outlierElim: bool = False) -> list:

    #Outlier elimination
    auxDf = pd.DataFrame()
    if outlierElim:
        auxDf = df[df.columns[(np.abs(stats.zscore(df,axis=0)) < 3).all(axis=0)]].copy()
    else:
        auxDf = df.copy()

    stockToSector = {}
    if not gics:
        stockToSector = getSectorsDict(df) 
    else:
        stockToSector = getGICSSectors(df)
        
    pctChangeDf = auxDf.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))
    transactions = pctChangeDf

    listOfTransLists = '\r\n'.join(f"{' '.join(map(str, transactions.columns))}:{' '.join(map(str, [stockToSector[index] for index in transactions.iloc[i].index]))}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))
    listOfTransLists = re.sub("  +", " ", listOfTransLists)
    
    dirname = getFileDir()

    outputFile = f'input-{dateStart["year"]}-{dateStart["month"]}-{dateStart["day"]}-To-{dateEnd["year"]}-{dateEnd["month"]}-{dateEnd["day"]}-{dataSet}-.txt'
    filePath = os.path.join(dirname, "cpp/bin/inputs")
    absolutePath = os.path.join(filePath, outputFile)

    with open(absolutePath, "w") as f:
        f.write(listOfTransLists)
        print(f"Saved stocks to file: {absolutePath}")
    
    return absolutePath

def convertToInputFilesKmeans(df: pd.DataFrame, dateStart: dict, dateEnd: dict, dataSet: str) -> list:

    #Outlier elimination
    auxDf = df.copy()
    returnList = []
    for k in range(2,20):
        stockToSector = {}

        stockToSector = getSectorsDict(df, k) 

            
        pctChangeDf = auxDf.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))
        transactions = pctChangeDf

        listOfTransLists = '\r\n'.join(f"{' '.join(map(str, transactions.columns))}:{' '.join(map(str, [stockToSector[index] for index in transactions.iloc[i].index]))}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))
        listOfTransLists = re.sub("  +", " ", listOfTransLists)
        
        dirname = getFileDir()

        outputFile = f'input-{dateStart["year"]}-{dateStart["month"]}-{dateStart["day"]}-To-{dateEnd["year"]}-{dateEnd["month"]}-{dateEnd["day"]}-{dataSet+"kmeans"+str(k)}-.txt'
        filePath = os.path.join(dirname, "cpp/bin/inputs")
        absolutePath = os.path.join(filePath, outputFile)

        returnList.append(absolutePath)

        with open(absolutePath, "w") as f:
            f.write(listOfTransLists)
            print(f"Saved stocks to file: {absolutePath}")
    
    return returnList

def getGICSSectors(df: pd.DataFrame) -> defaultdict(list):
    # sectors_dict = defaultdict(list)
    sectors_dict = {}
    tickersToDrop = []
    
    print("Getting GICS sectors from file...")
    with open("sectors.json") as jsonf:
        sectors_dict = json.load(jsonf)

        
    sectorList = []

    for ticker in df.columns:
        try:
            sectorList.append(sectors_dict[ticker])
        except:
            sectors_dict[ticker] = yf.Ticker(ticker).info["sector"]
            sectorList.append(sectors_dict[ticker])

    sectorListNr = [{val: key for key, val in enumerate(
        OrderedDict.fromkeys(sectorList))}
        [ele] for ele in sectorList]
    
    
    return {df.columns[i]:sectorListNr[i] for i in range(len(df.columns))}


def generateInputFile(dateStart: dict, dateEnd: dict, granularity: str = "static", dataSet: str = "spy") -> list:
    assert(dateStart["day"] == dateEnd["day"])
    if granularity == "yearly":
        assert(dateStart["month"] == dateStart["month"])
    
    listOfFiles = []
    df = downloadStockData(dateStart, dateEnd, dataSet)

    auxDateStart = dateStart.copy()
    
    if granularity == "static":
        auxDf = df
        fname = convertToInputFile(auxDf, dateStart, dateEnd, dataSet, False, True)
        listOfFiles.append(fname)
        fname = convertToInputFile(auxDf, dateStart, dateEnd, dataSet+"noelimGICS", True, False)
        listOfFiles.append(fname)
        return listOfFiles
    
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
        
        # auxDf = filterStocksByDate(df, auxDateStart, auxDateEnd)
        auxDf = df
        fname = convertToInputFile(auxDf, auxDateStart, auxDateEnd, dataSet, False, True)
        listOfFiles.append(fname)
        fname = convertToInputFile(auxDf, auxDateStart, auxDateEnd, dataSet+"noelimGICS", True, False)
        listOfFiles.append(fname)
        
        if granularity == "trimester":
            if auxDateStart["month"] + 3 > 12:
                auxDateStart["month"] = (auxDateStart["month"]+3)%12 if (auxDateStart["month"]+3)%12 != 0 else 1
                auxDateStart["year"] += 1
            else:
                auxDateStart["month"] += 3
        else:
            auxDateStart[granularity] += 1
            
            
    # fname = convertToInputFile(df, dateStart, dateEnd, dataSet)
    # listOfFiles.append(fname)
        
    return listOfFiles

def generateInputFileKmeans(dateStart: dict, dateEnd: dict, dataSet: str = "spy") -> list:
    
    df = downloadStockData(dateStart, dateEnd, dataSet)
    listOfFiles = convertToInputFilesKmeans(df, dateStart, dateEnd, dataSet)
        
    return listOfFiles

def getScoresFromFile() -> dict:
    with open('images/score.txt') as scoresFile:
        scores = {}
        for line in scoresFile.readlines():
            scores[int((line.split()[3]))] = float(line.split(":")[1][0:-1])
        return scores

def showScoresFig():
    scores = getScoresFromFile()
    # pl.ylim([0,1])
    pl.xlim([2,19])
    pl.plot(list(scores.keys()),list(scores.values()))
    pl.show()