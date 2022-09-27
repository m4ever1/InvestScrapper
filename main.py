from operator import attrgetter
import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from itemset_mining.two_phase_huim import TwoPhase
import random
import inputReader
# from spmf import Spmf
import re
import os
# import weightCalculate


def whitespace_remover(dataframe):
   
    # iterating over the columns
    for i in dataframe.columns:
         
        # checking datatype of each columns
        if dataframe[i].dtype == 'object':
             
            # applying strip function on column
            dataframe[i] = dataframe[i].map(str.strip)
        else:
             
            # if condn. is False then it will do nothing.
            pass

# store = pd.HDFStore('stocks.h5')

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
sectors_dict = defaultdict(list)
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    # sector = row.findAll('td')[4].text
    ticker = ticker.strip()
    tickers.append(ticker)
    # sectors_dict[sector].append(ticker)

tickers = [s.replace('\n', '') for s in tickers]
dateStart = {
            "year" : 2022,
            "month" : 8,
            "day"   : 1
        }
dateEnd = {
            "year" : 2022,
            "month" : 9,
            "day"   : 1
        }
start = datetime.datetime(dateStart["year"], dateStart["month"], dateStart["day"])
end = datetime.datetime(dateEnd["year"], dateEnd["month"], dateEnd["day"])
data = yf.download(tickers, start=start, end=end)

data["Adj Close"].to_csv("stocks.csv")

df = pd.read_csv("stocks.csv")
df = df.set_index("Date")

df = df.dropna(how="all")
df = df.dropna(axis=1)
df = df.interpolate()
df = df.sort_index()
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

for ticker in G.columns:
    row = table.find("a", string=ticker).parent.parent
    sector = row.findAll('td')[3].text
    sectors_dict[sector].append(ticker)


pctChangeDf = df.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))

stocksToTransact = []
for stck in list(sectors_dict.values()):
    stock_name = stck
    stocksToTransact += stock_name 

transactions = pctChangeDf
# .iloc[: , 0:20]
stockToSector = {stock: index for index, tuple in enumerate(sectors_dict.items()) for stock in tuple[1]}

# listOfTransLists = '\r\n'.join(f"{' '.join(map(str, range(1, len(transactions.columns) + 1)))}:{transactions.iloc[i].sum()}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))

listOfTransLists = '\r\n'.join(f"{' '.join(map(str, transactions.columns))}:{' '.join(map(str, [stockToSector[index] for index in transactions.iloc[i].index]))}:{np.array2string(transactions.iloc[i].values, max_line_width=float('inf'), floatmode='fixed', sign='-')[1:-1].strip()}" for i in range(len(transactions.index)))
listOfTransLists = re.sub("  +", " ", listOfTransLists)

dirname = os.path.dirname(__file__)

outputFile = f'input-{dateStart["year"]}-{dateStart["month"]}-{dateStart["day"]}-To-{dateEnd["year"]}-{dateEnd["month"]}-{dateEnd["day"]}-.txt'
filePath = os.path.join(dirname, "cpp/bin/inputs")
relativePath = os.path.join(filePath, outputFile)

with open(relativePath, "w") as f:
    f.write(listOfTransLists)

# weights = weightCalculate.getWeights(df)