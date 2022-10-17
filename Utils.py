import datetime, requests, re, sys, os
from multiprocessing.sharedctypes import Value
from logging import exception
from email.policy import default
import bs4 as bs
import yfinance as yf
from collections import defaultdict
import pandas as pd
import numpy as np
from kmeans import getSectorsDict
from math import sqrt
import  pylab as pl
from collections import OrderedDict

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

    # for gicsSect in gicsSects.values():
    #     tickerList = gicsSect
    #     for ticker in tickerList:
    #         try:
    #             if kmeansSects[ticker] in countDict:
    #                 countDict[kmeansSects[ticker]] += 1
    #             else:
    #                 countDict[kmeansSects[ticker]] = 1
    #         except:
    #             pass

    # return countDict

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

def downloadStockData(dateStart: dict, dateEnd: dict, dataSet: str = "spy") -> pd.DataFrame:
    if dataSet == "spy":
        tickerList = scrapeTickerList("spy")
    elif dataSet == "nasdaq":
        if dateEnd["year"] < 2015:
            tickerList = readTickersFromFile()
        else:
            tickerList = scrapeTickerList("nasdaq")
    else: 
        raise Exception("Invalid data set")

    start = datetime.datetime(dateStart["year"], dateStart["month"], dateStart["day"])
    end = datetime.datetime(dateEnd["year"], dateEnd["month"], dateEnd["day"])
    
    data = yf.download(tickerList, start=start, end=end)
    data["Adj Close"].to_csv("stocks.csv")
    data = data["Adj Close"]
    data = data.dropna(how="all")
    data = data.dropna(axis=1)
    data = data.interpolate()
    
    return data

def readDataFromCsv() -> pd.DataFrame:
    df = pd.read_csv("stocks.csv")
    df = df.set_index("Date")
    df = df.sort_index()
    
    return df
    # df = df.dropna(how="all")
    # df = df.dropna(axis=1)
    # df = df.interpolate()
    # df = df.sort_index()

def filterStocksByDate(df: pd.DataFrame, dateStart: dict, dateEnd: dict) -> pd.DataFrame:
    start = datetime.datetime(dateStart["year"], dateStart["month"], dateStart["day"])
    end = datetime.datetime(dateEnd["year"], dateEnd["month"], dateEnd["day"])
    
    dfOut = df.loc[str(start):str(end)]
    dfOut = dfOut.dropna(how="all")
    dfOut = dfOut.dropna(axis=1)
    dfOut = dfOut.interpolate()
    return dfOut

def convertToInputFile(df: pd.DataFrame, dateStart: dict, dateEnd: dict) -> str:
    G = pd.DataFrame()
    G = np.log(df).shift(-1) - np.log(df)

    G = G.dropna(how='all')

    # rows = []
    # for ticker_i in G:
    #     column = []
    #     for ticker_j in G:
    #         column.append((np.mean(G[ticker_i].multiply(G[ticker_j])) - np.mean(G[ticker_i])*np.mean(G[ticker_j]))/(G[ticker_i].std()*G[ticker_j].std()))
    #     rows.append(column)

    # C = pd.DataFrame(rows, columns=G.columns, index=G.columns)

    # t = G.shape[0]
    # n = G.shape[1]

    # Q = t/n

    # lMax = 1 + 1/Q + 2*np.sqrt(1/Q)
    # lMin = 1 + 1/Q - 2*np.sqrt(1/Q)

    pctChangeDf = df.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))

    # stocksToTransact = []
    # for stck in list(sectors_dict.values()):
    #     stock_name = stck
    #     stocksToTransact += stock_name 

    transactions = pctChangeDf
    # stockToSector = {stock: index for index, tuple in enumerate(sectors_dict.items()) for stock in tuple[1]}
    stockToSector, droppedTicker = getSectorsDict(df)
    # listOfTransLists = '\r\n'.join(f"{' '.join(map(str, range(1, len(transactions.columns) + 1)))}:{transactions.iloc[i].sum()}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))
    transactions = transactions.drop(droppedTicker, axis=1, errors="ignore")

    listOfTransLists = '\r\n'.join(f"{' '.join(map(str, transactions.columns))}:{' '.join(map(str, [stockToSector[index] for index in transactions.iloc[i].index]))}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))
    listOfTransLists = re.sub("  +", " ", listOfTransLists)
    
    dirname = getFileDir()

    outputFile = f'input-{dateStart["year"]}-{dateStart["month"]}-{dateStart["day"]}-To-{dateEnd["year"]}-{dateEnd["month"]}-{dateEnd["day"]}-.txt'
    filePath = os.path.join(dirname, "")
    absolutePath = os.path.join(filePath, outputFile)

    with open(absolutePath, "w") as f:
        f.write(listOfTransLists)
        print(f"Saved stocks to file: {absolutePath}")
    
    return absolutePath

def getGICSSectors(df: pd.DataFrame) -> defaultdict(list):
    # sectors_dict = defaultdict(list)
    sectors_dict = {}
    table = getTable()
    for ticker in df.columns:
        try:
            row = table.find("a", string=ticker).parent.parent
            sector = row.findAll('td')[3].text
            sectors_dict[ticker] = sector
        except:
            pass

    sectorList = []
    tickersToDrop = []

    for ticker in df.columns:
        try:
            sectorList.append(sectors_dict[ticker])
        except:
            tickersToDrop.append(ticker)

    sectorListNr = [{val: key for key, val in enumerate(
        OrderedDict.fromkeys(sectorList))}
        [ele] for ele in sectorList]

    returns = df.drop(tickersToDrop, axis=1).pct_change().mean() * 252
    variance = df.drop(tickersToDrop, axis=1).pct_change().std() * sqrt(252)
    returns.columns = ["Returns"]

    variance.columns = ["Variance"]
    sectors = pd.DataFrame(sectorListNr, index=returns.index)

    #Concatenating the returns and variances into a single data-frame
    ret_var = pd.concat([returns, variance, sectors], axis = 1).dropna()
    ret_var.columns = ["Returns","Variance", "Sector"]

    X =  ret_var.values #Converting ret_var into nummpy array
    # X = ((df-df.mean())/(df.std())).values #Converting ret_var into nummpy array
    
    pl.scatter(X[:,0],X[:,1], c = X[:,2], cmap ="rainbow")
    pl.show()
    
    # ticker = pd.DataFrame({'ticker' : ret_var.index})
    # cluster_labels = pd.DataFrame({'sector' : kmeans.labels_})
    # zip_iterator = zip(ret_var.index, kmeans.labels_)
    # dataOut = dict(zip_iterator)
    # dataOut = pd.concat([ticker, cluster_labels],axis = 1)
    # dataOut = dataOut.set_index('ticker')
    
    # return dataOut

def generateInputFile(dateStart: dict, dateEnd: dict, granularity: str, dataSet: str = "spy") -> list:
    assert(dateStart["day"] == dateEnd["day"])
    if granularity == "yearly":
        assert(dateStart["month"] == dateStart["month"])
    
    listOfFiles = []
    df = downloadStockData(dateStart, dateEnd, dataSet)

    auxDateStart = dateStart.copy()
    while(auxDateStart != dateEnd):
        if auxDateStart["month"] == 12:
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
            auxDateEnd[granularity] += 1
        
        auxDf = filterStocksByDate(df, auxDateStart, auxDateEnd)
        fname = convertToInputFile(auxDf, auxDateStart, auxDateEnd)
        
        if granularity == "trimester":
            if auxDateStart["month"] + 3 > 12:
                auxDateStart["month"] = (auxDateStart["month"]+3)%12 if (auxDateStart["month"]+3)%12 != 0 else 1
                auxDateStart["year"] += 1
            else:
                auxDateStart["month"] += 3
        else:
            auxDateStart[granularity] += 1

        listOfFiles.append(fname)
        
    return listOfFiles
