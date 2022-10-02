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


def getFileDir():
    if getattr(sys, 'frozen', False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)

    return datadir

def downloadStockData(dateStart: dict, dateEnd: dict) -> pd.DataFrame:
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker.strip()
        tickers.append(ticker)

    tickers = [s.replace('\n', '') for s in tickers]

    start = datetime.datetime(dateStart["year"], dateStart["month"], dateStart["day"])
    end = datetime.datetime(dateEnd["year"], dateEnd["month"], dateEnd["day"])
    
    data = yf.download(tickers, start=start, end=end)
    data["Adj Close"].to_csv("stocks.csv")
    
    return data["Adj Close"]

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

    transactions = pctChangeDf.iloc[: , 0:20]
    # stockToSector = {stock: index for index, tuple in enumerate(sectors_dict.items()) for stock in tuple[1]}
    stockToSector, droppedTicker = getSectorsDict(df)
    # listOfTransLists = '\r\n'.join(f"{' '.join(map(str, range(1, len(transactions.columns) + 1)))}:{transactions.iloc[i].sum()}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))
    transactions = transactions.drop(droppedTicker, axis=1, errors="ignore")

    listOfTransLists = '\r\n'.join(f"{' '.join(map(str, transactions.columns))}:{' '.join(map(str, [stockToSector[index] for index in transactions.iloc[i].index]))}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))
    listOfTransLists = re.sub("  +", " ", listOfTransLists)
    
    dirname = getFileDir()

    outputFile = f'input-{dateStart["year"]}-{dateStart["month"]}-{dateStart["day"]}-To-{dateEnd["year"]}-{dateEnd["month"]}-{dateEnd["day"]}-.txt'
    filePath = os.path.join(dirname, "cpp/bin/inputs")
    absolutePath = os.path.join(filePath, outputFile)

    with open(absolutePath, "w") as f:
        f.write(listOfTransLists)
        print(f"Saved stocks to file: {absolutePath}")
    
    return absolutePath

def getGICSSectors(G: pd.DataFrame, table) -> defaultdict(list):
    sectors_dict = defaultdict(list)
    
    for ticker in G.columns:
        row = table.find("a", string=ticker).parent.parent
        sector = row.findAll('td')[3].text
        sectors_dict[sector].append(ticker)
        
    return sectors_dict

def generateInputFile(dateStart: dict, dateEnd: dict, granularity: str) -> list(str):
    assert(dateStart["day"] == dateEnd["day"])
    if granularity == "yearly":
        assert(dateStart["month"] == dateStart["month"])
    
    listOfFiles = []
    df = readDataFromCsv()

    auxDateStart = dateStart.copy()
    while(auxDateStart != dateEnd):
        auxDateEnd = auxDateStart.copy()
        auxDateEnd[granularity] += 1
        
        auxDf = filterStocksByDate(df, auxDateStart, auxDateEnd)
        fname = convertToInputFile(auxDf, auxDateStart, auxDateEnd)
        
        auxDateStart[granularity] += 1
        listOfFiles.append(fname)
        
    return listOfFiles
    