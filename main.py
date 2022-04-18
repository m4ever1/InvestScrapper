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

# tickers = [s.replace('\n', '') for s in tickers]
# start = datetime.datetime(2011, 1, 1)
# end = datetime.datetime(2021, 1, 1)
# data = yf.download(tickers, start=start, end=end)

# data["Adj Close"].to_csv("stocks.csv")
# store["data"] = data

# data = store['data']
df = pd.read_csv("stocks.csv")
df = df.set_index("Date")

# new_header = data.iloc[0] #grab the first row for the header
# data = data[1:] #take the data less the header row
# data.set_index(data.columns[0])
# data.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'))
# data.rename(columns = {"Unnamed: 0_level_0" : "Stat", "Unnamed: 0_level_1" : "Ticker"})
# df = data["Adj Close"]
df = df.dropna(how="all")
df = df.dropna(axis=1)
df = df.interpolate()
df = df.sort_index()
G = pd.DataFrame()


# Iterate through every ticker
G = np.log(df).shift(-1) - np.log(df)

G = G.dropna(how='all')

# C = np.mean(G.multiply)

rows = []
for ticker_i in G:
    column = []
    for ticker_j in G:
        column.append((np.mean(G[ticker_i].multiply(G[ticker_j])) - np.mean(G[ticker_i])*np.mean(G[ticker_j]))/(G[ticker_i].std()*G[ticker_j].std()))
    rows.append(column)

C = pd.DataFrame(rows, columns=G.columns, index=G.columns)

t = G.shape[0]
n = G.shape[1]

Q = t/n

lMax = 1 + 1/Q + 2*np.sqrt(1/Q)
lMin = 1 + 1/Q - 2*np.sqrt(1/Q)

for ticker in G.columns:
    row = table.find("a", string=ticker).parent.parent
    sector = row.findAll('td')[3].text
    sectors_dict[sector].append(ticker)
# for row in table.findAll('tr')[1:]:
#     ticker = row.findAll('td')[0].text
#     sector = row.findAll('td')[3].text
#     ticker = ticker.strip()
#     if ticker in G.columns:
#         sectors_dict[sector].append(ticker)

# for ticker in G.columns:
#     sector = yf.Ticker(ticker).info["sector"]
#     sectors_dict[sector].append(ticker)


# setOfSectors = set(sectors_dict.keys())

# Pmatrix = np.zeros([len(setOfSectors), len(G.columns)])
# P = pd.DataFrame(Pmatrix, columns=G.columns)



# for sectorId, stocks in enumerate(sectors_dict.values()):
#     for stock in stocks:
#         Nl = len(stocks)
#         P[stock][sectorId] = 1/Nl

    

# w, v = np.linalg.eig(C)

# # eigVectors = np.delete(v, 0, axis=0)

# lastVector = 0
# for i, value in enumerate(w):
#     if value < lMax:
#         lastVector = i
#         break

# eigValues = w #[1:(lastVector)]
# eigVectors = v #[1:(lastVector)]

# eigVectorsSquared = np.square(eigVectors)

# X = np.dot(eigVectorsSquared, P.T)
# # X = np.zeros([442, 120])
# # for k, eigVector in enumerate(eigVectors):
# #     for l, pVector in enumerate(np.matrix(P)):
# #         X[k][l] = np.sum(np.multiply(pVector, np.square(eigVector)))

# print(X)


pctChangeDf = df.apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))

stocksToTransact = []
for stck in list(sectors_dict.values()):
    stock_name = stck[0]
    stocksToTransact.append(stock_name)

transactions = pctChangeDf[pctChangeDf.columns[pctChangeDf.columns.isin(stocksToTransact)]]
# transactions = transactions.clip(0)
# transactions = transactions.round(0).astype(int)
# wSupport = transactions.min(axis=1).mean()

listOfTransLists = [[(tick, transactions.iloc[1][j]) for j, tick in enumerate(transactions.columns) if transactions.iloc[1][j] != 0] for i in range(len(transactions.index)) if i < 3]

# listOfTransLists = [(column, random.randint(1,10)) for column in transactions.columns[0:6]]
    
# listOfTransLists = [list(transactions.iloc[i].items()) for i in range(len(transactions.index))]

profits = dict.fromkeys(transactions.columns, 1)

hui = TwoPhase(listOfTransLists, profits, -1)
result = hui.get_hui()
print(sorted(result, key=attrgetter('itemset_utility'), reverse=True))
